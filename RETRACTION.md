# RETRACTION. The paper is INVALID. Do not cite or reuse its results.

**Status: withdrawn. All quantitative claims of the study are invalid.**

The LaTeX source and PDF are not committed to this repository (see `.gitignore`).
The retracted paper exists only as its published record, which carries the same
retraction notice. This document is the reason.

The headline result does not survive a corrected computation. This is not a
presentation or rounding problem: the numbers in the paper are faithfully
reproduced by the code in `scripts/`, and the tables match the CSVs in
`output/results/` exactly. The computation itself was wrong.

---

## Root cause

**The noise injection never perturbed the features that carried the label.**

`inject_real_noise` (`scripts/run_10_real_noise.py:201-206`) perturbs only the
leading magnitude columns:

```python
X[:, :n_noisy] += rng.standard_normal((X.shape[0], n_noisy)) * sigma
```

The feature vector is built as `base_cols = mag_cols + other_cols`
(`run_10_real_noise.py:182`), so everything after the magnitude bands is left
untouched at every noise level. Those trailing features are the ones that
determine the class.

### 1. Spectroscopic redshift is an input feature (both surveys)

`redshift` is column 5 of 6 and is never perturbed. It nearly determines the
label. Median by class: STAR −0.0001, GALAXY 0.456, QSO 1.617.

On the paper's own SDSS split (73,302 objects, seed 42, 80/20):

| Features | Accuracy (depth-6 tree) |
|---|---|
| **`redshift` alone** | **0.9470** |
| `u,g,r,i,z` (photometry only) | 0.8043 |
| all six | 0.9651 |

A depth-6 tree on that single feature scores **0.9470, above the paper's reported
KAN (0.9347) and MLP-Reg (0.9355) using all six features**. The task is described
throughout as *photometric* classification; it is not.

Because redshift is never noised, every noise curve in the paper is floored by a
clean feature that alone beats the models being compared. At α = 10 the paper
reports KAN 0.8200 and MLP-Reg 0.7741, both far *below* what the untouched
redshift column yields on its own.

### 2. DESI `bp`/`rp` are a class-correlated sentinel, also never perturbed

`extract_desi_errors.py:78` applies `dropna` to `g,r,z` only. In the DESI
zcatalog a missing Gaia magnitude is stored as **0.0**, not NaN, so the
`between(0, 40)` filter in `build_desi` (`run_10_real_noise.py:172-173`) accepts
it as a valid magnitude.

- `bp == 0` for **56.6%** of rows.
- Zero-rate by class: **GALAXY 96.7%, QSO 67.2%, STAR 5.9%.**
- The rule "`bp != 0` ⇒ STAR" alone scores accuracy 0.86, F1(STAR) 0.82.

`bp`/`rp` are columns 3-4, so they too survive every noise level intact.

### 3. The consequence: the SDSS/DESI contrast is an artifact

At any α, **SDSS kept 1 of 6 features clean; DESI kept 3 of 6** (redshift + bp +
rp). Those three alone score 0.9058, against the paper's DESI KAN of 0.9148 at
α = 10. DESI's reported robustness was almost entirely unperturbed leaked signal,
not architecture and not photometry.

The entire paper is built on the contrast "KAN edge on SDSS, null on DESI". That
contrast is a measurement artifact of this asymmetry.

---

## What the corrected computation gives

`scripts/run_14_no_leak.py` re-runs the equal-baseline on photometry only:
SDSS `u,g,r,i,z` (5 features), DESI `g,r,z` (3 features), so every input is a
magnitude and **every input is perturbed**. It also selects weight decay on a
validation split carved out of train, instead of on the test set (see defect S1).

| | published (leaky) | corrected |
|---|---|---|
| SDSS KAN clean acc | 0.9347 | **0.7813** |
| DESI KAN clean acc | 0.9306 | **0.8204** |
| **KAN − MLP-Reg gap @ α=10, SDSS** | **+4.41 p.p.** | **+1.53 p.p.** |
| **KAN − MLP-Reg gap @ α=10, DESI** | **−0.52 p.p.** | **+1.06 p.p.** |

15.3 p.p. (SDSS) and 11.0 p.p. (DESI) of the published "clean accuracy" was the
unperturbed leaked features.

**The DESI null does not merely shrink, it changes sign.** Corrected, DESI shows
a KAN edge at *every* α (+0.46 rising to +1.06, growing monotonically). Published,
the contrast was +4.4 vs −0.5 (opposite signs); corrected it is +1.5 vs +1.1 (same
sign, comparable size). With the leak removed DESI actually degrades under noise
(0.8204 to 0.7884) instead of barely moving (0.9305 to 0.9148).

### Claim-by-claim

| Paper claim | Verdict |
|---|---|
| (i) Equal-baseline equivalence survives real noise at α ≤ 2 | **Survives.** Corrected SDSS gaps at α ≤ 2 are within ±0.07 p.p. (std ~0.1), indistinguishable. This is the one honest result, and it is cleaner than published, since it now holds on genuine photometry. |
| (ii) Genuine KAN architectural advantage, SDSS-only extreme, absent on DESI | **Falsified as stated.** +4.4/−0.5 becomes +1.5/+1.1. |
| (iii) Uncertainties as features don't help | **Untested** in the corrected setting. |
| (iv) Jacobian mechanism | **Partly falsified.** Order-of-magnitude suppression by both weight decay and KAN survives qualitatively but is ~5.5 to 11.8×, not 10 to 22×. The predictive part fails: corrected, DESI has the *larger* relative KAN Jacobian advantage (3.12/6.66, 53% lower) but the *smaller* noise-driven edge, than SDSS (1.75/2.41, 27% lower), the opposite of the paper's logic. |
| §4.4 band-ablation: the `u` band *causes* the edge | **Not re-run; unsafe.** `u` is not special for being ultraviolet, it has the largest σ (median 0.48 mag, vs r 0.076, vs DESI ~0.008). The defensible statement is that the edge scales with perturbation amplitude, not that the `u` band causes it. |
| §4.6 per-class: the edge is a quasar effect (+13.3 p.p. QSO F1) | **Not re-run; unsafe.** Computed with redshift present, and redshift *is* the quasar discriminator (median z = 1.62). |

---

## Secondary defects (independent of the above)

- **S1, weight decay tuned on the test set.** `tune_mlp_reg`
  (`run_10_real_noise.py:135-145`) selects wd by `accuracy_score(yte, ...)`, and
  the same test set is then used to report every result. The "equal baseline" was
  fitted on test data. Fixed only in `run_14_no_leak.py`.
- **S2, `obj_ID` is not unique.** `data/star_classification.csv` has 100,000 rows
  but 78,053 unique `obj_ID`, and rows sharing an `obj_ID` have different
  photometry, redshift, and sometimes class. The inner join to `sdss_errors.csv`
  is one-to-many, so **26,800 of 73,302 usable rows (36.6%)** receive an error
  vector belonging to a different object (median within-group spread: 0.67 mag in
  r, 0.00 in the assigned σ_r). "Per-object" uncertainty does not hold for over a
  third of the SDSS sample.
- **S3, model cache ignores training configuration.** `train_kan`
  (`run_10_real_noise.py:112-132`) keys the checkpoint on `tag` only, not on
  `SMOKE`/`KAN_STEPS`/feature set. A `SMOKE=1` run silently poisons every
  subsequent full run, and a cached KAN cannot be verified after the fact.
- **S4, the Lipschitz column is not independent evidence.** In
  `output/results/lipschitz_metrics.csv`, `jac_mean/lip_mean` is 2.82 to 2.98
  across all six rows: for small δ, ‖f(x+δ)−f(x)‖/‖δ‖ ≈ ‖J‖_F/√n, and √6 = 2.449.
  The Lipschitz estimate is a rescaling of the Jacobian, but §3.3 and Table 6
  present it as a second, corroborating measurement.
- **S5**, the paper states median u-band error 0.49; the actual median is 0.479
  (0.48).

---

## Repository status, read before reusing anything here

**The original scripts are NOT fixed and the original results are NOT corrected.**

| Path | Status |
|---|---|
| `scripts/run_10_real_noise.py` | **Unfixed.** Contains the leaky feature builders, the never-perturbed non-photometric columns, S1 and S3. |
| `scripts/run_11_lipschitz.py`, `run_12_ablation_perclass.py`, `run_13_seed_stats.py` | **Unfixed.** All import the leaky builders from `run_10`. |
| `scripts/extract_desi_errors.py` | **Unfixed.** Emits the 0.0 bp/rp sentinel. |
| `output/results/*.csv` (all pre-existing) | **Invalid.** Superseded, not regenerated. |
| `scripts/run_14_no_leak.py` | **New, leakage-free.** The only trustworthy script here. |
| `output/results/noleak_*.csv` | **New.** Corrected Variant B + Jacobian only. |

### Limits of the corrected numbers

They are reported as-is, and they are not sufficient to support a replacement claim:

- **One training seed, 5 noise trials.** The published headline used 5 seeds with
  std ±1.67 p.p. at α = 10. The corrected SDSS +1.53 p.p. sits inside that
  seed-level spread and **cannot yet be distinguished from zero.** The ±0.21 std
  in the corrected run measures noise realizations of one fixed model, not seed
  variance.
- **The DESI baseline is not equalized.** KAN clean 0.8204 vs MLP-Reg 0.8161, a
  +0.43 p.p. offset, because the wd grid is too coarse to match. So of DESI's
  +1.06 p.p. at α = 10, only ~+0.6 p.p. is noise-driven (SDSS: ~+1.6 p.p.).
- **`bp`/`rp` were dropped entirely** rather than restricted to the Gaia-matched
  subset. That is one defensible choice, not the only one.
- Ablation, per-class, and seed statistics were **not** re-run.

Establishing anything beyond claim (i) requires, at minimum, a seed-replicated
leakage-free run. That has not been done.

## Reproducing the diagnosis

```bash
python scripts/run_14_no_leak.py     # leakage-free equal-baseline + Jacobian
```

The two checks that expose the root cause directly:

```python
# 1. redshift alone beats the paper's models on the paper's own split
from sklearn.tree import DecisionTreeClassifier   # -> 0.9470 vs published KAN 0.9347

# 2. the DESI Gaia sentinel
import pandas as pd
d = pd.read_csv('data/desi_dr1_sample_with_errors.csv')
d.groupby('class')['bp'].apply(lambda s: (s == 0).mean())
# GALAXY 0.967 | QSO 0.672 | STAR 0.059
```

## Lesson

Every table in the paper matches its CSV, and `run_11_lipschitz.py` reproduces to
four decimals. Checking outputs against outputs cannot detect this class of error.
The defect was visible only from reconstructing the input matrix, asking what each
column physically is, and running a trivial single-feature baseline, which takes
seconds and would have caught it before any model was trained.
