#!/usr/bin/env python3
"""13 — Statistical robustness of the equal-baseline gap (multiple training seeds).

The headline KAN edge at extreme noise rests, in run_10, on a single training seed
and 5 noise trials. This script hardens it: it retrains KAN and the matched MLP-Reg
under several independent training seeds, runs more noise trials, and reports the
KAN - MLP-Reg accuracy gap with seed-level uncertainty plus a significance test.

For each survey and each training seed:
  - train a fresh KAN (its own seed) on the base feature set;
  - tune MLP-Reg (weight decay) to that KAN's clean accuracy;
  - over N_TRIALS_B noise realizations, measure the gap at each alpha.

Statistics per alpha:
  - seed-level mean gap +/- std (each seed = one independent model-level replicate);
  - one-sample t-test on the 5 seed-mean gaps vs 0 (robustness to training seed);
  - pooled Wilcoxon signed-rank over all seed x trial pairs (reported for context).

Outputs:
    output/results/seed_stats_gap_{sdss,desi}.csv   (raw: seed, alpha, trial, gap)
    output/results/seed_stats_summary.csv           (per survey/alpha summary + p-values)
    output/figures/seed_stats_gap.png

Env: SMOKE=1 -> 2 seeds, 2 trials, 20-step KAN (fast check).

Usage:
    python scripts/run_13_seed_stats.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from scipy import stats
from kan import KAN

import run_10_real_noise as R

RESULTS_DIR = R.RESULTS_DIR
FIGURES_DIR = R.FIGURES_DIR
SMOKE = R.SMOKE
SEEDS = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
N_TRIALS_B = 2 if SMOKE else 20
ALPHAS = R.ALPHAS                       # {0.5,1,2,5,10}
KEY_ALPHAS = [5.0, 10.0]                # where the gap is tested


def train_kan_seed(Xtr, ytr, Xte, yte, n_feat, n_cls, tag, seed):
    ckpt = R.MODELS_DIR / f"kan_{tag}.pt"
    model = KAN(width=[n_feat, 24, 12, n_cls], grid=5, k=3, seed=seed,
                device="cpu", ckpt_path=R.PYKAN_CACHE, auto_save=False)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.eval()
        print(f"    [cached] KAN {tag}")
        return model
    ds = {'train_input': torch.FloatTensor(Xtr), 'train_label': torch.LongTensor(ytr),
          'test_input': torch.FloatTensor(Xte), 'test_label': torch.LongTensor(yte)}
    model.fit(ds, opt="Adam", lr=1e-3, steps=R.KAN_STEPS, lamb=0.001,
              loss_fn=nn.CrossEntropyLoss())
    model.eval()
    torch.save(model.state_dict(), ckpt)
    return model


def tune_mlp_reg_seed(Xtr, ytr, Xte, yte, n_feat, n_cls, target, seed):
    grid = R.WD_GRID[:3] if SMOKE else R.WD_GRID
    best, best_acc, best_wd = None, 0.0, 0.0
    for wd in grid:
        m = R.train_mlp(Xtr, ytr, n_feat, n_cls, wd=wd, seed=seed)
        acc = accuracy_score(yte, R.predict_torch(m, Xte))
        if abs(acc - target) < abs(best_acc - target):
            best, best_acc, best_wd = m, acc, wd
    return best, best_wd, best_acc


def run_survey(D):
    name = D['name']
    itr, ite = D['itr'], D['ite']
    sc = D['scaler']
    Xb, Xe, y = D['X_base'], D['X_err'], D['y']
    n_cls = len(D['class_names'])
    n_base = Xb.shape[1]
    n_noisy = D['n_noisy']
    Xtr, Xte = sc.transform(Xb[itr]), sc.transform(Xb[ite])
    ytr, yte = y[itr], y[ite]

    rows = []
    for seed in SEEDS:
        kan = train_kan_seed(Xtr, ytr, Xte, yte, n_base, n_cls, f"{name}_seed{seed}", seed)
        kan_acc = accuracy_score(yte, R.predict_torch(kan, Xte))
        mlpreg, wd, mr_acc = tune_mlp_reg_seed(Xtr, ytr, Xte, yte, n_base, n_cls, kan_acc, seed)
        print(f"  [{name}] seed={seed}: KAN={kan_acc:.4f} MLP-Reg={mr_acc:.4f} (wd={wd})")
        for alpha in ALPHAS:
            for trial in range(N_TRIALS_B):
                rng = np.random.default_rng(1000 * seed + trial)
                Xn = sc.transform(R.inject_real_noise(Xb[ite], Xe[ite], n_noisy, alpha, rng))
                ak = accuracy_score(yte, R.predict_torch(kan, Xn))
                am = accuracy_score(yte, R.predict_torch(mlpreg, Xn))
                rows.append({'seed': seed, 'alpha': alpha, 'trial': trial,
                             'kan': ak, 'mlpreg': am, 'gap': ak - am})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"seed_stats_gap_{name}.csv", index=False)
    return df


def summarize(name, df):
    out = []
    for alpha in ALPHAS:
        sub = df[df['alpha'] == alpha]
        seed_means = sub.groupby('seed')['gap'].mean().values     # one per seed
        gap_mean = seed_means.mean()
        gap_std = seed_means.std(ddof=1) if len(seed_means) > 1 else 0.0
        # robustness to training seed: is the seed-mean gap != 0 ?
        if len(seed_means) > 1 and np.ptp(seed_means) > 0:
            t_p = stats.ttest_1samp(seed_means, 0.0).pvalue
        else:
            t_p = np.nan
        # pooled paired test over all seed x trial pairs (context only)
        try:
            w_p = stats.wilcoxon(sub['kan'], sub['mlpreg']).pvalue
        except ValueError:
            w_p = np.nan
        out.append({'dataset': name, 'alpha': alpha,
                    'gap_mean': round(gap_mean, 4), 'gap_std': round(gap_std, 4),
                    'ttest_p_seedmeans': round(t_p, 4) if t_p == t_p else np.nan,
                    'wilcoxon_p_pooled': round(w_p, 6) if w_p == w_p else np.nan})
    return pd.DataFrame(out)


def main():
    print(f"SMOKE={SMOKE} SEEDS={SEEDS} N_TRIALS_B={N_TRIALS_B} KAN_STEPS={R.KAN_STEPS}")
    summaries, raws = [], {}
    for builder in (R.build_sdss, R.build_desi):
        D = builder()
        df = run_survey(D)
        raws[D['name']] = df
        summaries.append(summarize(D['name'], df))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(RESULTS_DIR / "seed_stats_summary.csv", index=False)
    print("\n===== Seed-level gap summary (KAN - MLP-Reg) =====")
    print(summary.to_string(index=False))
    print("\nKey claim test (SDSS, alpha>=5): seed-mean gap and one-sample t-test p:")
    print(summary[(summary.dataset == 'sdss') & (summary.alpha.isin(KEY_ALPHAS))]
          [['alpha', 'gap_mean', 'gap_std', 'ttest_p_seedmeans']].to_string(index=False))

    # figure: gap mean +/- std vs alpha, both surveys
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'sdss': '#1E88E5', 'desi': '#E53935'}
    for ds, c in colors.items():
        s = summary[summary.dataset == ds]
        ax.errorbar(s['alpha'], s['gap_mean'], yerr=s['gap_std'], marker='o',
                    color=c, lw=2, capsize=3, label=ds.upper())
    ax.axhline(0, color='gray', ls=':', alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel(r"Noise scale $\alpha$")
    ax.set_ylabel(r"KAN $-$ MLP-Reg gap (mean $\pm$ std over seeds)")
    ax.set_title(f"Equal-baseline gap across {len(SEEDS)} training seeds")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "seed_stats_gap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
