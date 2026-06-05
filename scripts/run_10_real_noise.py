#!/usr/bin/env python3
"""10 — Real photometric-noise extension (Variants A / B / C).

Tests whether the equal-baseline finding (KAN ≈ MLP-Reg under noise) survives
when the noise is *real* per-object catalogue uncertainty rather than synthetic
uniform-SNR noise. Runs on both SDSS DR17 and DESI DR1.

Requires Phase-0 products:
    data/sdss_errors.csv                  (from fetch_sdss_errors.py)
    data/desi_dr1_sample_with_errors.csv  (from extract_desi_errors.py)

Variants:
    A — uncertainties as extra input features (replicates LSST DP1, arXiv:2603.25262):
        train on base vs base+errors, compare clean + noisy accuracy per model.
    B — per-object real-noise injection: perturb each object by its own catalogue
        sigma scaled by alpha in {0.5,1,2,5,10}; recompute equal-baseline curve.
    C — magnitude-binned degradation under the real per-object noise.

Outputs:
    output/results/real_noise_variantA_<ds>.csv
    output/results/real_noise_variantB_<ds>.csv
    output/results/real_noise_variantC_<ds>.csv
    output/figures/real_noise_variantB_<ds>.png
    output/figures/real_noise_variantC_<ds>.png
Trained models cached under output/models/real_noise/ (resumable).

Env:
    SMOKE=1   reduce epochs/trials/grid for a fast end-to-end smoke test.

Usage:
    python scripts/run_10_real_noise.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from kan import KAN
from xgboost import XGBClassifier
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"
RESULTS_DIR = PROJECT_ROOT / "output" / "results"
MODELS_DIR = PROJECT_ROOT / "output" / "models" / "real_noise"
PYKAN_CACHE = str(PROJECT_ROOT / "output" / "pykan_cache")
for d in (FIGURES_DIR, RESULTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
sns.set_theme(style="whitegrid")
SMOKE = os.environ.get("SMOKE") == "1"

MLP_EPOCHS = 5 if SMOKE else 50
KAN_STEPS = 20 if SMOKE else 200
N_TRIALS = 2 if SMOKE else 5
ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0]
WD_GRID = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]
SEED = 42


# ── models ───────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def predict_torch(model, X_np):
    with torch.no_grad():
        return model(torch.FloatTensor(X_np).to(DEVICE)).argmax(dim=1).cpu().numpy()


def train_mlp(Xtr, ytr, n_feat, n_cls, wd=0.0, epochs=MLP_EPOCHS, seed=SEED):
    torch.manual_seed(seed)
    model = MLP(n_feat, 64, n_cls).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xtr), torch.LongTensor(ytr)),
        batch_size=512, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            opt.step()
    model.eval()
    return model


def train_kan(Xtr, ytr, Xte, yte, n_feat, n_cls, tag):
    """Train (or load cached) a KAN; cached by tag for resumability."""
    ckpt = MODELS_DIR / f"kan_{tag}.pt"
    model = KAN(width=[n_feat, 24, 12, n_cls], grid=5, k=3, seed=SEED,
                device=str(DEVICE), ckpt_path=PYKAN_CACHE, auto_save=False)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        model.eval()
        print(f"    [cached] KAN {tag}")
        return model
    ds = {
        'train_input': torch.FloatTensor(Xtr).to(DEVICE),
        'train_label': torch.LongTensor(ytr).to(DEVICE),
        'test_input': torch.FloatTensor(Xte).to(DEVICE),
        'test_label': torch.LongTensor(yte).to(DEVICE),
    }
    model.fit(ds, opt="Adam", lr=1e-3, steps=KAN_STEPS, lamb=0.001,
              loss_fn=nn.CrossEntropyLoss())
    model.eval()
    torch.save(model.state_dict(), ckpt)
    return model


def tune_mlp_reg(Xtr, ytr, Xte, yte, n_feat, n_cls, target_acc, tag):
    """Grid-search weight decay so MLP clean accuracy matches `target_acc`."""
    best, best_acc, best_wd = None, 0.0, 0.0
    grid = WD_GRID[:3] if SMOKE else WD_GRID
    for wd in grid:
        m = train_mlp(Xtr, ytr, n_feat, n_cls, wd=wd)
        acc = accuracy_score(yte, predict_torch(m, Xte))
        if abs(acc - target_acc) < abs(best_acc - target_acc):
            best, best_acc, best_wd = m, acc, wd
    print(f"    MLP-Reg[{tag}] wd={best_wd} acc={best_acc:.4f} (target {target_acc:.4f})")
    return best, best_wd, best_acc


# ── data builders ─────────────────────────────────────────────────────────────
def build_sdss():
    """Return dict with raw mags, errors, labels, scaler, column indices."""
    df = pd.read_csv(DATA_DIR / "star_classification.csv")
    err = pd.read_csv(DATA_DIR / "sdss_errors.csv")
    mag_cols = ['u', 'g', 'r', 'i', 'z']
    err_cols = ['psfMagErr_u', 'psfMagErr_g', 'psfMagErr_r', 'psfMagErr_i', 'psfMagErr_z']
    df = df.merge(err, left_on='obj_ID', right_on='objID', how='inner')
    # drop bad magnitudes (Kaggle -9999 sentinels) and bad/missing errors
    good = np.ones(len(df), dtype=bool)
    for c in mag_cols:
        good &= df[c].between(0, 40)
    for c in err_cols:
        good &= df[c] > 0
    df = df[good].reset_index(drop=True)
    return _assemble(df, mag_cols, err_cols, other_cols=['redshift'],
                     class_col='class', name='sdss')


def build_desi():
    df = pd.read_csv(DATA_DIR / "desi_dr1_sample_with_errors.csv")
    mag_cols = ['g', 'r', 'z_mag']
    err_cols = ['g_err', 'r_err', 'z_err']
    good = np.ones(len(df), dtype=bool)
    for c in mag_cols + ['bp', 'rp']:
        good &= df[c].between(0, 40)
    for c in err_cols:
        good &= df[c] > 0
    df = df[good].reset_index(drop=True)
    return _assemble(df, mag_cols, err_cols, other_cols=['bp', 'rp', 'redshift'],
                     class_col='class', name='desi')


def _assemble(df, mag_cols, err_cols, other_cols, class_col, name):
    base_cols = mag_cols + other_cols                # noisy bands first, then the rest
    X_base = df[base_cols].values.astype(np.float64)
    X_err = df[err_cols].values.astype(np.float64)   # aligned to mag_cols (first len(mag_cols))
    le = LabelEncoder()
    y = le.fit_transform(df[class_col])

    idx = np.arange(len(df))
    itr, ite = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_base[itr])
    print(f"  [{name}] {len(df)} objs, base feats {base_cols}, "
          f"errors on {mag_cols}, classes {list(le.classes_)}")
    return {
        'name': name, 'base_cols': base_cols, 'mag_cols': mag_cols,
        'n_noisy': len(mag_cols), 'class_names': list(le.classes_),
        'X_base': X_base, 'X_err': X_err, 'y': y,
        'itr': itr, 'ite': ite, 'scaler': scaler,
    }


def inject_real_noise(X_base, X_err, n_noisy, alpha, rng):
    """Perturb each object's noisy bands by alpha * its own catalogue sigma."""
    X = X_base.copy()
    sigma = X_err[:, :n_noisy] * alpha
    X[:, :n_noisy] += rng.standard_normal((X.shape[0], n_noisy)) * sigma
    return X


# ── experiment per dataset ─────────────────────────────────────────────────────
def run_dataset(D):
    name = D['name']
    cn = D['class_names']
    itr, ite = D['itr'], D['ite']
    sc = D['scaler']
    Xb, Xe, y = D['X_base'], D['X_err'], D['y']
    n_cls = len(cn)
    n_base = Xb.shape[1]
    n_noisy = D['n_noisy']

    # standardized base train/test
    Xtr = sc.transform(Xb[itr]); Xte = sc.transform(Xb[ite])
    ytr, yte = y[itr], y[ite]

    # augmented feature set (base + errors), its own scaler
    Xb_aug = np.hstack([Xb, Xe])
    sc_aug = StandardScaler().fit(Xb_aug[itr])
    Xtr_a = sc_aug.transform(Xb_aug[itr]); Xte_a = sc_aug.transform(Xb_aug[ite])
    n_aug = Xb_aug.shape[1]

    print(f"\n{'='*64}\nDATASET: {name.upper()}  base={n_base}f  aug={n_aug}f\n{'='*64}")

    # ── train base-feature models ──
    print("  training base-feature models ...")
    kan_b = train_kan(Xtr, ytr, Xte, yte, n_base, n_cls, f"{name}_base")
    kan_b_acc = accuracy_score(yte, predict_torch(kan_b, Xte))
    mlp_b = train_mlp(Xtr, ytr, n_base, n_cls)
    xgb_b = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          random_state=SEED, eval_metric='mlogloss').fit(Xtr, ytr)
    mlpreg_b, wd_b, _ = tune_mlp_reg(Xtr, ytr, Xte, yte, n_base, n_cls, kan_b_acc, f"{name}_base")

    # ── train augmented-feature models ──
    print("  training augmented-feature (base+errors) models ...")
    kan_a = train_kan(Xtr_a, ytr, Xte_a, yte, n_aug, n_cls, f"{name}_aug")
    kan_a_acc = accuracy_score(yte, predict_torch(kan_a, Xte_a))
    mlp_a = train_mlp(Xtr_a, ytr, n_aug, n_cls)
    xgb_a = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          random_state=SEED, eval_metric='mlogloss').fit(Xtr_a, ytr)
    mlpreg_a, wd_a, _ = tune_mlp_reg(Xtr_a, ytr, Xte_a, yte, n_aug, n_cls, kan_a_acc, f"{name}_aug")

    # ════════════════════════════════════════════════════════════════════
    # VARIANT A — uncertainties as features: clean + real-noise(alpha=1) acc
    # ════════════════════════════════════════════════════════════════════
    def acc_on(model, Xstd, is_torch=True):
        p = predict_torch(model, Xstd) if is_torch else model.predict(Xstd)
        return accuracy_score(yte, p), f1_score(yte, p, average='weighted')

    rng = np.random.default_rng(0)
    Xn_base = inject_real_noise(Xb[ite], Xe[ite], n_noisy, 1.0, rng)
    Xn_base_std = sc.transform(Xn_base)
    rng = np.random.default_rng(0)
    Xn_aug_phys = inject_real_noise(Xb_aug[ite], Xe[ite], n_noisy, 1.0, rng)
    Xn_aug_std = sc_aug.transform(Xn_aug_phys)

    rowsA = []
    for label, model, is_t, Xc, Xn in [
        ('KAN', kan_b, True, Xte, Xn_base_std),
        ('MLP', mlp_b, True, Xte, Xn_base_std),
        ('MLP-Reg', mlpreg_b, True, Xte, Xn_base_std),
        ('XGBoost', xgb_b, False, Xte, Xn_base_std),
    ]:
        ca, cf = acc_on(model, Xc, is_t); na, nf = acc_on(model, Xn, is_t)
        rowsA.append({'featureset': 'base', 'Model': label,
                      'clean_acc': ca, 'clean_f1': cf, 'noisy_acc': na, 'noisy_f1': nf})
    for label, model, is_t, Xc, Xn in [
        ('KAN', kan_a, True, Xte_a, Xn_aug_std),
        ('MLP', mlp_a, True, Xte_a, Xn_aug_std),
        ('MLP-Reg', mlpreg_a, True, Xte_a, Xn_aug_std),
        ('XGBoost', xgb_a, False, Xte_a, Xn_aug_std),
    ]:
        ca, cf = acc_on(model, Xc, is_t); na, nf = acc_on(model, Xn, is_t)
        rowsA.append({'featureset': 'base+errors', 'Model': label,
                      'clean_acc': ca, 'clean_f1': cf, 'noisy_acc': na, 'noisy_f1': nf})
    dfA = pd.DataFrame(rowsA)
    dfA.to_csv(RESULTS_DIR / f"real_noise_variantA_{name}.csv", index=False)
    print(f"\n  Variant A [{name}] (clean / noisy@alpha=1):")
    print(dfA.round(4).to_string(index=False))

    # ════════════════════════════════════════════════════════════════════
    # VARIANT B — equal-baseline under per-object real noise, scaled by alpha
    # ════════════════════════════════════════════════════════════════════
    rowsB = []
    models_b = {'KAN': (kan_b, True), 'MLP-Reg': (mlpreg_b, True),
                'XGBoost': (xgb_b, False)}
    for alpha in ALPHAS:
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(trial)
            Xn = sc.transform(inject_real_noise(Xb[ite], Xe[ite], n_noisy, alpha, rng))
            for label, (model, is_t) in models_b.items():
                p = predict_torch(model, Xn) if is_t else model.predict(Xn)
                rowsB.append({'alpha': alpha, 'trial': trial, 'Model': label,
                              'Accuracy': accuracy_score(yte, p)})
        print(f"    Variant B alpha={alpha}: done")
    dfB = pd.DataFrame(rowsB)
    dfB.to_csv(RESULTS_DIR / f"real_noise_variantB_{name}.csv", index=False)
    pivB = dfB.groupby(['Model', 'alpha'])['Accuracy'].mean().unstack('Model').round(4)
    print(f"\n  Variant B [{name}] equal-baseline (acc vs alpha):")
    print(pivB.to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'KAN': '#1E88E5', 'MLP-Reg': '#E53935', 'XGBoost': '#43A047'}
    for label, color in colors.items():
        md = dfB[dfB['Model'] == label].groupby('alpha')['Accuracy']
        m, s = md.mean(), md.std()
        ax.plot(m.index, m.values, 'o-', color=color, label=label, lw=2)
        ax.fill_between(m.index, m.values - s.values, m.values + s.values, alpha=0.15, color=color)
    ax.set_xlabel(r"Noise scale $\alpha$ (× catalogue $\sigma$)")
    ax.set_ylabel("Accuracy")
    ax.set_xscale('log')
    ax.set_title(f"Equal-baseline under real per-object noise — {name.upper()}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"real_noise_variantB_{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════════
    # VARIANT C — magnitude-binned degradation (real noise, alpha=1)
    # ════════════════════════════════════════════════════════════════════
    # bin by r-band magnitude (index 2 in SDSS mag_cols; 1 in DESI -> use 'r')
    r_idx = D['mag_cols'].index('r')
    r_mag = Xb[ite][:, r_idx]
    edges = [0, 18, 20, 22, 99]
    rowsC = []
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(trial)
        Xn = sc.transform(inject_real_noise(Xb[ite], Xe[ite], n_noisy, 1.0, rng))
        for label, (model, is_t) in models_b.items():
            p = predict_torch(model, Xn) if is_t else model.predict(Xn)
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (r_mag >= lo) & (r_mag < hi)
                if m.sum() == 0:
                    continue
                rowsC.append({'bin': f"{lo}-{hi}", 'trial': trial, 'Model': label,
                              'n': int(m.sum()),
                              'Accuracy': accuracy_score(yte[m], p[m])})
    dfC = pd.DataFrame(rowsC)
    dfC.to_csv(RESULTS_DIR / f"real_noise_variantC_{name}.csv", index=False)
    pivC = dfC.groupby(['Model', 'bin'])['Accuracy'].mean().unstack('Model').round(4)
    print(f"\n  Variant C [{name}] accuracy by r-mag bin (noisy@alpha=1):")
    print(pivC.to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    order = [b for b in ['0-18', '18-20', '20-22', '22-99'] if b in pivC.index]
    for label, color in colors.items():
        if label in pivC.columns:
            ax.plot(order, pivC.loc[order, label].values, 'o-', color=color, label=label, lw=2)
    ax.set_xlabel("r-band magnitude bin (fainter →)")
    ax.set_ylabel("Accuracy (real noise, α=1)")
    ax.set_title(f"Magnitude-binned degradation — {name.upper()}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"real_noise_variantC_{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {'kan_base_acc': kan_b_acc, 'kan_aug_acc': kan_a_acc, 'wd_base': wd_b, 'wd_aug': wd_a}


def main():
    print(f"SMOKE={SMOKE}  MLP_EPOCHS={MLP_EPOCHS} KAN_STEPS={KAN_STEPS} N_TRIALS={N_TRIALS}")
    summary = {}
    for builder in (build_sdss, build_desi):
        D = builder()
        summary[D['name']] = run_dataset(D)
    print("\n" + "=" * 64)
    print("SUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("Done.")


if __name__ == "__main__":
    main()
