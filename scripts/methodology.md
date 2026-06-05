# Methodology: real photometric noise and Lipschitz mechanism

Companion methodology to `paper/paper.tex`. Kept in sync with the paper.

## 1. Question

The companion paper (arXiv:2605.29039) showed that KAN's noise-robustness
advantage over MLPs in stellar classification is an *implicit-regularization*
effect: once an MLP is given enough weight decay to match KAN's clean accuracy
(the *equal-baseline* protocol), the gap under synthetic uniform-SNR noise
vanishes. This experiment asks whether that holds under **real, per-object
photometric uncertainties** from the survey catalogues, and measures the
mechanism directly.

## 2. Data and uncertainties (Phase 0)

| Survey | N usable | Base features | Real per-object errors |
|--------|----------|---------------|------------------------|
| SDSS DR17 | 73,302 | u, g, r, i, z, redshift | psfMagErr_{u..z} (SkyServer `PhotoObjAll`, joined on `obj_ID`; 76.8% match) |
| DESI DR1 | 99,999 | g, r, z, BP, RP, redshift | σ_{g,r,z} from FLUX_IVAR |

DESI magnitude errors: σ_mag = 1.0857 · (FLUX_IVAR)^(−1/2) / FLUX (g, r, z only;
Gaia BP/RP carry no per-object error). Median errors: SDSS r≈0.08, u≈0.49 mag;
DESI g,r,z≈0.008 mag. Features standardized; 80/20 stratified split, seed 42.

## 3. Models

- **KAN 2.0** `[n,24,12,3]`, cubic B-splines (k=3, grid=5), L1 λ=0.001, Adam, 200 steps.
- **MLP** `[n,64,64,3]` SiLU, 50 epochs.
- **MLP-Reg**: the MLP with weight decay grid-searched so its clean accuracy
  matches KAN's (the control isolating regularization from architecture).
- **XGBoost**: 200 trees, depth 6.

## 4. Experiments (Phase 1, `run_10_real_noise.py`)

- **Variant A: uncertainties as features.** Train each model on base vs
  base+errors; compare clean and noisy accuracy (replicates the LSST DP1
  augmentation, now including KAN).
- **Variant B: per-object real-noise injection.** Perturb each object's bands
  by its own catalogue σ scaled by α ∈ {0.5,1,2,5,10}:
  x ← x + 𝒩(0, (α·σ_i)²). α=1 reproduces catalogue noise; α>1 is a controlled
  heteroscedastic stress-test. Recompute the equal-baseline curve (5 trials/α).
- **Variant C: magnitude-binned degradation.** At α=1, accuracy in r-band bins
  (<18, 18 to 20, 20 to 22, >22).

## 5. Mechanism (Phase 2, `run_11_lipschitz.py`)

At matched clean accuracy, measure on the standardized test set:
- mean & max Frobenius norm of the input→logit Jacobian ‖∂f/∂x‖_F (2000 points);
- a local empirical Lipschitz estimate max ‖f(x+δ)−f(x)‖/‖δ‖ over small random δ.
Compared across KAN, MLP-Reg, and an unregularized MLP.

## 6. Headline results

- **Variant A:** uncertainties-as-features do not help (neutral/slightly worse), a
  demarcation from the LSST DP1 Random-Forest result.
- **Variant B:** equal-baseline holds at realistic noise (α≤2) on both surveys
  (KAN ≈ MLP-Reg within 0.1 p.p.); a KAN edge emerges only in the SDSS extreme
  (α≥5, +4.6 p.p. at α=10), absent on DESI.
- **Lipschitz:** unregularized MLP jac_mean ~58 to 68 vs KAN ~3.1 vs MLP-Reg ~5.6 to 7.0
  → implicit-regularizer confirmed; KAN's mean Jacobian ~45% below MLP-Reg
  predicts its extreme-noise edge.
- **Band-ablation (`run_12`):** the SDSS edge is caused by the high-σ u band
  (gap @α=10: u_only +8.6, all5 +4.6, no_u −1.7, grz −0.8 p.p.); the grz subset
  reproduces the DESI null on SDSS objects, removing the SDSS/DESI confound.
- **Per-class (`run_12`):** the SDSS edge is a quasar effect (QSO F1 +13.3 p.p. at
  α=10), with a galaxy gain (+3.3) offset by a star loss (−3.3); DESI within ±0.5 p.p.
- **Seed-robustness (`run_13`):** 5 training seeds × 20 noise trials. SDSS gap
  +4.41 ± 1.67 p.p. at α=10 (p=0.004, one-sample t-test on seed means); DESI gap
  −0.52 ± 0.20 p.p. (p=0.004). The edge is seed-robust and the DESI null is confirmed.

## 7. References

- Companion paper: arXiv:2605.29039.
- KAN or MLP: A Fairer Comparison, arXiv:2407.16674 (demarcation).
- LSST DP1 uncertainties-as-features, arXiv:2603.25262.
- KAN / KAN 2.0: arXiv:2404.19756, arXiv:2408.10205.
- 21cmKAN (KAN in astrophysics): Dorigo Jones et al. 2025, ApJ 991, 152.
