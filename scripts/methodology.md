# ⚠️ RETRACTED. This methodology describes an invalid experiment.

> The study is withdrawn. The design documented below contains the error that
> invalidates it: **the noise injection never perturbed the features that carried
> the label** (see §2 and §4, marked below).
>
> This file is retained to document what was actually done, not as guidance.
> **Read [`../RETRACTION.md`](../RETRACTION.md) first.**

# Methodology: real photometric noise and Lipschitz mechanism

Methodology of the retracted study. The LaTeX source is not committed to this
repository (see `.gitignore`).

## 1. Question

The companion paper (arXiv:2605.29039) showed that KAN's noise-robustness
advantage over MLPs in stellar classification is an *implicit-regularization*
effect: once an MLP is given enough weight decay to match KAN's clean accuracy
(the *equal-baseline* protocol), the gap under synthetic uniform-SNR noise
vanishes. This experiment asked whether that holds under **real, per-object
photometric uncertainties** from the survey catalogues, and measured the
mechanism directly.

## 2. Data and uncertainties (Phase 0)

| Survey | N usable | Base features | Real per-object errors |
|--------|----------|---------------|------------------------|
| SDSS DR17 | 73,302 | u, g, r, i, z, redshift | psfMagErr_{u..z} (SkyServer `PhotoObjAll`, joined on `obj_ID`; 76.8% match) |
| DESI DR1 | 99,999 | g, r, z, BP, RP, redshift | σ_{g,r,z} from FLUX_IVAR |

> **⚠️ This table is the defect.** `redshift` is spectroscopic and nearly
> determines the class (median: STAR −0.0001, GALAXY 0.456, QSO 1.617). A depth-6
> tree on redshift alone scores 0.9470 on the SDSS split, above the reported KAN
> (0.9347) on all six features. Photometry alone scores 0.8043. DESI `BP`/`RP` are
> 0.0 for 56.6% of rows (Gaia non-match sentinel, kept because the validity filter
> accepts 0.0), and that zero is class-correlated: GALAXY 96.7%, QSO 67.2%, STAR
> 5.9%. Only the magnitude bands carry a σ, so only they were ever perturbed:
> **SDSS kept 1 of 6 features clean at every noise level, DESI kept 3 of 6.**
>
> The join is also one-to-many: `obj_ID` is not unique (100,000 rows, 78,053
> unique), so 36.6% of rows receive an error vector belonging to a different
> object. The "76.8% match" is a row fraction, not an object fraction.

DESI magnitude errors: σ_mag = 1.0857 · (FLUX_IVAR)^(−1/2) / FLUX (g, r, z only;
Gaia BP/RP carry no per-object error). Median errors: SDSS r≈0.08, u≈0.48 mag;
DESI g,r,z≈0.008 mag. Features standardized; 80/20 stratified split, seed 42.

## 3. Models

- **KAN 2.0** `[n,24,12,3]`, cubic B-splines (k=3, grid=5), L1 λ=0.001, Adam, 200 steps.
- **MLP** `[n,64,64,3]` SiLU, 50 epochs.
- **MLP-Reg**: the MLP with weight decay grid-searched so its clean accuracy
  matches KAN's (the control isolating regularization from architecture).
- **XGBoost**: 200 trees, depth 6.

> **⚠️ The MLP-Reg grid search scored weight decay on the TEST set**
> (`run_10_real_noise.py:141`), and the same test set was then used to report every
> result. The equal-baseline control was fitted on test data. Only
> `run_14_no_leak.py` selects weight decay on a validation split.
>
> The KAN checkpoint cache keys on `tag` alone, not on `SMOKE`/`KAN_STEPS`/feature
> set, so a smoke run silently poisons later full runs.

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

> **⚠️ Variant B is where the defect operates.** The implementation perturbs only
> the leading magnitude columns, `X[:, :n_noisy]`, while the feature vector is
> `mag_cols + other_cols`. Redshift and BP/RP therefore survive every α intact, so
> every noise curve is floored by clean, label-leaking inputs. At α=10 the study
> reports KAN 0.8200 and MLP-Reg 0.7741, both below what the untouched redshift
> column alone yields (0.9470).

## 5. Mechanism (Phase 2, `run_11_lipschitz.py`)

At matched clean accuracy, measure on the standardized test set:
- mean & max Frobenius norm of the input→logit Jacobian ‖∂f/∂x‖_F (2000 points);
- a local empirical Lipschitz estimate max ‖f(x+δ)−f(x)‖/‖δ‖ over small random δ.
Compared across KAN, MLP-Reg, and an unregularized MLP.

> **⚠️ The Lipschitz estimate is not an independent measurement.** For small δ,
> ‖f(x+δ)−f(x)‖/‖δ‖ ≈ ‖J‖_F/√n, and with n=6 inputs √6 = 2.449. Measured,
> `jac_mean/lip_mean` is 2.82 to 2.98 across all six rows: it is a rescaling of
> the Jacobian, presented as corroboration.

## 6. Results

**Withdrawn.** The headline results previously listed here are invalid. See
[`../RETRACTION.md`](../RETRACTION.md) for the claim-by-claim verdict and for what
the corrected computation (`run_14_no_leak.py`) gives instead.

## 7. References

- Companion paper: arXiv:2605.29039 (a separate work, not affected by this retraction).
- KAN or MLP: A Fairer Comparison, arXiv:2407.16674 (demarcation).
- LSST DP1 uncertainties-as-features, arXiv:2603.25262.
- KAN / KAN 2.0: arXiv:2404.19756, arXiv:2408.10205.
- 21cmKAN (KAN in astrophysics): Dorigo Jones et al. 2025, ApJ 991, 152.
