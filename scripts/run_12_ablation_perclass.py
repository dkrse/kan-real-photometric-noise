#!/usr/bin/env python3
"""12 — Band-ablation (A) and per-class degradation (C) under real noise.

Adds two analyses on top of run_10:

A. Band-ablation (SDSS). The paper attributes KAN's extreme-noise edge to the
   strongly heteroscedastic high-sigma u band. This tests it causally: re-run the
   equal-baseline (KAN vs MLP-Reg) injecting real noise into different band
   subsets and check where the KAN edge appears.
     all5  : u,g,r,i,z   (the main result)
     grz   : g,r,z       (DESI-like 3 low-sigma bands; controls SDSS/DESI confound)
     u_only: u           (the high-sigma band alone)
     no_u  : g,r,i,z     (everything except u)
   Prediction: the KAN edge is present for all5 and u_only, absent for grz/no_u.

C. Per-class degradation (SDSS + DESI). Per-class F1 (GALAXY/QSO/STAR) under real
   noise at alpha in {1,5,10}, to show which class drives the KAN edge.

Reuses run_10's data builders, cached KAN models, MLP-Reg tuning, and noise.

Outputs:
    output/results/ablation_bands_sdss.csv
    output/results/perclass_realnoise_{sdss,desi}.csv
    output/figures/ablation_bands_sdss.png
    output/figures/perclass_realnoise.png

Usage:
    python scripts/run_12_ablation_perclass.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

import run_10_real_noise as R

RESULTS_DIR = R.RESULTS_DIR
FIGURES_DIR = R.FIGURES_DIR
ALPHAS = R.ALPHAS
N_TRIALS = R.N_TRIALS


def inject_bands(X_base, X_err, band_idx, alpha, rng):
    """Real-noise injection restricted to the given magnitude-band indices.

    Columns of X_err are aligned to the magnitude bands (first n_noisy columns of
    the base feature set), so `band_idx` indexes both X_base and X_err.
    """
    X = X_base.copy()
    sig = X_err[:, band_idx] * alpha
    X[:, band_idx] += rng.standard_normal((X.shape[0], len(band_idx))) * sig
    return X


def fit_pair(D):
    """Return (kan, mlpreg, scaler, Xb, Xe, y, itr, ite) for a dataset dict."""
    itr, ite = D['itr'], D['ite']
    sc = D['scaler']
    Xb, Xe, y = D['X_base'], D['X_err'], D['y']
    n_cls = len(D['class_names'])
    n_base = Xb.shape[1]
    Xtr, Xte = sc.transform(Xb[itr]), sc.transform(Xb[ite])
    ytr, yte = y[itr], y[ite]
    kan = R.train_kan(Xtr, ytr, Xte, yte, n_base, n_cls, f"{D['name']}_base")
    kan_acc = accuracy_score(yte, R.predict_torch(kan, Xte))
    mlpreg, wd, _ = R.tune_mlp_reg(Xtr, ytr, Xte, yte, n_base, n_cls, kan_acc, D['name'])
    return kan, mlpreg, sc, Xb, Xe, y, itr, ite


# ════════════════════════════════════════════════════════════════════════════
# A. Band-ablation on SDSS
# ════════════════════════════════════════════════════════════════════════════
def ablation_sdss():
    print("\n" + "=" * 64 + "\nA. BAND-ABLATION (SDSS)\n" + "=" * 64)
    D = R.build_sdss()                       # mag_cols = u,g,r,i,z (idx 0..4)
    kan, mlpreg, sc, Xb, Xe, y, itr, ite = fit_pair(D)
    yte = y[ite]
    band_sets = {
        'all5':   [0, 1, 2, 3, 4],
        'grz':    [1, 2, 4],
        'u_only': [0],
        'no_u':   [1, 2, 3, 4],
    }
    rows = []
    for name, idx in band_sets.items():
        for alpha in ALPHAS:
            for trial in range(N_TRIALS):
                rng = np.random.default_rng(trial)
                Xn = sc.transform(inject_bands(Xb[ite], Xe[ite], idx, alpha, rng))
                ak = accuracy_score(yte, R.predict_torch(kan, Xn))
                am = accuracy_score(yte, R.predict_torch(mlpreg, Xn))
                rows.append({'bands': name, 'alpha': alpha, 'trial': trial,
                             'kan': ak, 'mlpreg': am, 'gap': ak - am})
        print(f"  bands={name}: done")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "ablation_bands_sdss.csv", index=False)

    gap = df.groupby(['bands', 'alpha'])['gap'].agg(['mean', 'std']).reset_index()
    print("\nKAN - MLP-Reg accuracy gap (mean over trials), by band subset:")
    piv = gap.pivot(index='alpha', columns='bands', values='mean').round(4)
    print(piv[['all5', 'u_only', 'no_u', 'grz']].to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'all5': '#1E88E5', 'u_only': '#E53935', 'no_u': '#43A047', 'grz': '#8E24AA'}
    for b, c in colors.items():
        g = gap[gap['bands'] == b]
        ax.errorbar(g['alpha'], g['mean'], yerr=g['std'], marker='o', color=c,
                    label=b, lw=2, capsize=3)
    ax.axhline(0, color='gray', ls=':', alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel(r"Noise scale $\alpha$")
    ax.set_ylabel("KAN $-$ MLP-Reg accuracy gap")
    ax.set_title("SDSS band-ablation: where the KAN edge comes from")
    ax.legend(title="noisy bands")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ablation_bands_sdss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return piv


# ════════════════════════════════════════════════════════════════════════════
# C. Per-class degradation under real noise
# ════════════════════════════════════════════════════════════════════════════
def perclass(D, alphas=(1.0, 5.0, 10.0)):
    name = D['name']
    cn = D['class_names']
    n_noisy = D['n_noisy']
    kan, mlpreg, sc, Xb, Xe, y, itr, ite = fit_pair(D)
    yte = y[ite]
    rows = []
    for alpha in alphas:
        for trial in range(N_TRIALS):
            rng = np.random.default_rng(trial)
            Xn = sc.transform(R.inject_real_noise(Xb[ite], Xe[ite], n_noisy, alpha, rng))
            for mname, model in [('KAN', kan), ('MLP-Reg', mlpreg)]:
                pred = R.predict_torch(model, Xn)
                f1 = f1_score(yte, pred, average=None, labels=list(range(len(cn))))
                for ci, cls in enumerate(cn):
                    rows.append({'dataset': name, 'alpha': alpha, 'trial': trial,
                                 'Model': mname, 'class': cls, 'f1': f1[ci]})
        print(f"  [{name}] per-class alpha={alpha}: done")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"perclass_realnoise_{name}.csv", index=False)
    piv = df.groupby(['Model', 'class', 'alpha'])['f1'].mean().unstack('alpha').round(4)
    print(f"\nPer-class F1 [{name}] (mean over trials):")
    print(piv.to_string())
    return df


def main():
    ablation_sdss()
    dfs = [perclass(R.build_sdss()), perclass(R.build_desi())]

    # combined per-class figure: F1 vs alpha, per class, KAN vs MLP-Reg, two surveys
    allc = pd.concat(dfs, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cls_color = {'GALAXY': '#1E88E5', 'QSO': '#FB8C00', 'STAR': '#43A047'}
    for ax, ds in zip(axes, ['sdss', 'desi']):
        sub = allc[allc['dataset'] == ds]
        for cls, c in cls_color.items():
            for mdl, ls in [('KAN', '-'), ('MLP-Reg', '--')]:
                g = sub[(sub['class'] == cls) & (sub['Model'] == mdl)]
                m = g.groupby('alpha')['f1'].mean()
                ax.plot(m.index, m.values, ls, color=c, lw=2,
                        label=f"{cls} {mdl}")
        ax.set_xscale('log')
        ax.set_xlabel(r"Noise scale $\alpha$")
        ax.set_title(ds.upper())
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Per-class F1")
    axes[1].legend(fontsize=7, ncol=1)
    plt.suptitle("Per-class degradation under real noise (solid KAN, dashed MLP-Reg)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "perclass_realnoise.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
