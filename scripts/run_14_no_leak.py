#!/usr/bin/env python3
"""14 — Leakage-free re-run of the equal-baseline experiment.

The run_10 feature sets contain inputs that (a) leak the label and (b) are never
perturbed by the noise injection, which floors every noise curve:

  - `redshift` (spectroscopic) is a feature in BOTH surveys. A depth-6 tree on
    redshift alone scores 0.947 on the SDSS split — above the reported KAN
    (0.9347) and MLP-Reg (0.9355) on all six features. It is column 5, while
    inject_real_noise only touches columns :n_noisy (=5), so it stays clean.
  - DESI `bp`/`rp` are 0.0 for 56.6% of rows (Gaia non-match sentinel, kept
    because `between(0,40)` accepts 0.0). Zero-rate by class: GALAXY 96.7%,
    QSO 67.2%, STAR 5.9%. They are columns 3-4, also never perturbed.

This script re-runs the equal-baseline on PHOTOMETRY ONLY, so every input feature
is a magnitude and every input feature is perturbed:

    SDSS: u,g,r,i,z      (5 features, 5 noised)
    DESI: g,r,z          (3 features, 3 noised)

It also fixes the weight-decay selection: run_10's tune_mlp_reg matches KAN's
accuracy on the TEST set and then reports on that same test set. Here the wd grid
is scored on a validation split carved out of train; test is touched only to report.

Outputs:
    output/results/noleak_variantB_{sdss,desi}.csv
    output/results/noleak_summary.csv
    output/results/noleak_jacobian.csv

Usage:
    python scripts/run_14_no_leak.py
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import run_10_real_noise as R

RESULTS_DIR = R.RESULTS_DIR
ALPHAS = R.ALPHAS
N_TRIALS = R.N_TRIALS
SEED = 42


def build_photometry_only(name):
    """Photometry-only feature set: every feature is a magnitude with a sigma."""
    if name == 'sdss':
        df = pd.read_csv(R.DATA_DIR / "star_classification.csv")
        err = pd.read_csv(R.DATA_DIR / "sdss_errors.csv")
        mag_cols = ['u', 'g', 'r', 'i', 'z']
        err_cols = ['psfMagErr_' + b for b in 'ugriz']
        df = df.merge(err, left_on='obj_ID', right_on='objID', how='inner')
    else:
        df = pd.read_csv(R.DATA_DIR / "desi_dr1_sample_with_errors.csv")
        mag_cols = ['g', 'r', 'z_mag']
        err_cols = ['g_err', 'r_err', 'z_err']
    good = np.ones(len(df), dtype=bool)
    for c in mag_cols:
        good &= df[c].between(0, 40)
    for c in err_cols:
        good &= df[c] > 0
    df = df[good].reset_index(drop=True)

    X = df[mag_cols].values.astype(np.float64)
    E = df[err_cols].values.astype(np.float64)
    y = LabelEncoder().fit(df['class']).transform(df['class'])
    cls = sorted(df['class'].unique())

    idx = np.arange(len(df))
    itr_full, ite = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    itr, iva = train_test_split(itr_full, test_size=0.2, random_state=SEED,
                                stratify=y[itr_full])
    sc = StandardScaler().fit(X[itr])
    print(f"  [{name}] {len(df)} objs | feats {mag_cols} (all noised) | "
          f"train {len(itr)} val {len(iva)} test {len(ite)}")
    return dict(name=name, mag_cols=mag_cols, X=X, E=E, y=y, cls=cls,
                itr=itr, iva=iva, ite=ite, sc=sc)


def tune_mlp_reg_val(Xtr, ytr, Xva, yva, n_feat, n_cls, target_val_acc):
    """Pick weight decay on VALIDATION (not test) to match KAN's val accuracy."""
    best, best_acc, best_wd = None, -1.0, None
    for wd in R.WD_GRID:
        m = R.train_mlp(Xtr, ytr, n_feat, n_cls, wd=wd)
        acc = accuracy_score(yva, R.predict_torch(m, Xva))
        if best is None or abs(acc - target_val_acc) < abs(best_acc - target_val_acc):
            best, best_acc, best_wd = m, acc, wd
    return best, best_wd, best_acc


def jac_mean(model, X, n_cls):
    Xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    logits = model(Xt)
    sq = torch.zeros(Xt.shape[0])
    for c in range(n_cls):
        g, = torch.autograd.grad(logits[:, c].sum(), Xt, retain_graph=True)
        sq += (g ** 2).sum(dim=1)
    return float(torch.sqrt(sq).detach().numpy().mean())


def run(D):
    name = D['name']
    sc, X, E, y = D['sc'], D['X'], D['E'], D['y']
    itr, iva, ite = D['itr'], D['iva'], D['ite']
    n_cls, n_feat = len(D['cls']), X.shape[1]
    Xtr, Xva, Xte = sc.transform(X[itr]), sc.transform(X[iva]), sc.transform(X[ite])
    ytr, yva, yte = y[itr], y[iva], y[ite]

    print(f"\n{'='*64}\n{name.upper()} — photometry only, all features noised\n{'='*64}")
    kan = R.train_kan(Xtr, ytr, Xva, yva, n_feat, n_cls, f"noleak_{name}")
    kan_val = accuracy_score(yva, R.predict_torch(kan, Xva))
    kan_test = accuracy_score(yte, R.predict_torch(kan, Xte))
    mlp = R.train_mlp(Xtr, ytr, n_feat, n_cls)                   # unregularized ref
    mlpreg, wd, mr_val = tune_mlp_reg_val(Xtr, ytr, Xva, yva, n_feat, n_cls, kan_val)
    mr_test = accuracy_score(yte, R.predict_torch(mlpreg, Xte))
    mlp_test = accuracy_score(yte, R.predict_torch(mlp, Xte))
    print(f"  clean: KAN test={kan_test:.4f} | MLP-Reg test={mr_test:.4f} (wd={wd}, "
          f"val match {kan_val:.4f} vs {mr_val:.4f}) | MLP(noWD) test={mlp_test:.4f}")

    rows = []
    for alpha in ALPHAS:
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(trial)
            Xn = sc.transform(R.inject_real_noise(X[ite], E[ite], n_feat, alpha, rng))
            for label, m in [('KAN', kan), ('MLP-Reg', mlpreg)]:
                p = R.predict_torch(m, Xn)
                rows.append({'alpha': alpha, 'trial': trial, 'Model': label,
                             'Accuracy': accuracy_score(yte, p),
                             'F1': f1_score(yte, p, average='weighted')})
        print(f"    alpha={alpha}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"noleak_variantB_{name}.csv", index=False)

    piv = df.groupby(['alpha', 'Model'])['Accuracy'].mean().unstack()
    piv['gap_pp'] = (piv['KAN'] - piv['MLP-Reg']) * 100
    print(f"\n  Variant B [{name}] leakage-free (acc vs alpha):")
    print(piv.round(4).to_string())

    jrows = [{'dataset': name, 'Model': l, 'jac_mean': round(jac_mean(m, Xte[:2000], n_cls), 4),
              'clean_test_acc': round(a, 4)}
             for l, m, a in [('KAN', kan, kan_test), ('MLP-Reg', mlpreg, mr_test),
                             ('MLP', mlp, mlp_test)]]
    return piv.reset_index().assign(dataset=name), pd.DataFrame(jrows)


def main():
    pivs, jacs = [], []
    for name in ('sdss', 'desi'):
        p, j = run(build_photometry_only(name))
        pivs.append(p); jacs.append(j)
    summary = pd.concat(pivs, ignore_index=True)
    summary.to_csv(RESULTS_DIR / "noleak_summary.csv", index=False)
    jac = pd.concat(jacs, ignore_index=True)
    jac.to_csv(RESULTS_DIR / "noleak_jacobian.csv", index=False)
    print("\n===== LEAKAGE-FREE SUMMARY (KAN - MLP-Reg, p.p.) =====")
    print(summary.round(4).to_string(index=False))
    print("\n===== Jacobian (photometry only) =====")
    print(jac.to_string(index=False))


if __name__ == "__main__":
    main()
