#!/usr/bin/env python3
"""11 — Lipschitz / Jacobian smoothness of the decision boundary (Extension #2).

Variant B (run_10) shows KAN gains a real edge over MLP-Reg only under extreme,
strongly heteroscedastic multi-band noise (SDSS), while on DESI the equal-baseline
holds at every level. This script tests the *mechanism* behind that: a smoother
input->logit map (lower Jacobian / Lipschitz constant) should degrade more
gracefully under input perturbation.

For each dataset we compare, at matched clean accuracy:
    KAN (cached from run_10)  vs  MLP-Reg (weight-decay tuned)  vs  MLP (no WD).

Metrics on the standardized test set (subsampled):
    - mean & max Frobenius norm of the input->logit Jacobian  ||df/dx||_F
    - local empirical Lipschitz estimate  max ||f(x+d)-f(x)|| / ||d||  over small d

Hypothesis: KAN ≈ MLP-Reg < MLP(default). If the SDSS KAN also shows a markedly
lower norm than MLP-Reg, that is the mechanistic signature of its extreme-noise edge.

Outputs:
    output/results/lipschitz_metrics.csv

Usage:
    python scripts/run_11_lipschitz.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

import run_10_real_noise as R  # reuse data builders + model trainers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "output" / "results"
DEVICE = torch.device("cpu")
N_JAC = 2000          # test points for Jacobian estimate
N_LIP = 4000          # points for local Lipschitz estimate
DELTA = 1e-2          # perturbation scale (standardized units)
SEED = 42


def jacobian_norms(model, X, n_cls):
    """Mean & max Frobenius norm of the input->logit Jacobian over rows of X."""
    Xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    logits = model(Xt)                       # (N, C)
    sq = torch.zeros(Xt.shape[0])
    for c in range(n_cls):
        g, = torch.autograd.grad(logits[:, c].sum(), Xt, retain_graph=True)
        sq += (g ** 2).sum(dim=1)
    fro = torch.sqrt(sq).detach().numpy()    # ||J||_F per sample
    return float(fro.mean()), float(fro.max())


def local_lipschitz(model, X, rng):
    """max ||f(x+d)-f(x)|| / ||d|| over small random perturbations d."""
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32)
        f0 = model(Xt)
        d = torch.tensor(rng.standard_normal(X.shape) * DELTA, dtype=torch.float32)
        f1 = model(Xt + d)
        num = torch.linalg.norm(f1 - f0, dim=1)
        den = torch.linalg.norm(d, dim=1)
        ratio = (num / den).numpy()
    return float(ratio.mean()), float(ratio.max())


def run_dataset(D):
    name = D['name']
    itr, ite = D['itr'], D['ite']
    sc = D['scaler']
    Xb, y = D['X_base'], D['y']
    n_cls = len(D['class_names'])
    n_base = Xb.shape[1]
    Xtr, Xte = sc.transform(Xb[itr]), sc.transform(Xb[ite])
    ytr, yte = y[itr], y[ite]

    # KAN cached from run_10; MLP-Reg tuned to KAN clean acc; MLP default
    kan = R.train_kan(Xtr, ytr, Xte, yte, n_base, n_cls, f"{name}_base")
    kan_acc = accuracy_score(yte, R.predict_torch(kan, Xte))
    mlp = R.train_mlp(Xtr, ytr, n_base, n_cls)
    mlpreg, wd, _ = R.tune_mlp_reg(Xtr, ytr, Xte, yte, n_base, n_cls, kan_acc, name)

    rng = np.random.default_rng(SEED)
    sub_j = rng.choice(len(Xte), size=min(N_JAC, len(Xte)), replace=False)
    sub_l = rng.choice(len(Xte), size=min(N_LIP, len(Xte)), replace=False)

    rows = []
    for label, model in [('KAN', kan), ('MLP-Reg', mlpreg), ('MLP', mlp)]:
        jm, jx = jacobian_norms(model, Xte[sub_j], n_cls)
        lm, lx = local_lipschitz(model, Xte[sub_l], np.random.default_rng(SEED))
        acc = accuracy_score(yte, R.predict_torch(model, Xte))
        rows.append({'dataset': name, 'Model': label, 'clean_acc': round(acc, 4),
                     'jac_mean': round(jm, 4), 'jac_max': round(jx, 4),
                     'lip_mean': round(lm, 4), 'lip_max': round(lx, 4)})
    return rows


def main():
    all_rows = []
    for builder in (R.build_sdss, R.build_desi):
        all_rows += run_dataset(builder())
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "lipschitz_metrics.csv", index=False)
    print("\n===== Lipschitz / Jacobian smoothness =====")
    print(df.to_string(index=False))
    print("\nHypothesis check: KAN jac_mean vs MLP-Reg vs MLP (default).")


if __name__ == "__main__":
    main()
