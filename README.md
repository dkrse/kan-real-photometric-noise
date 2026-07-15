# ⚠️ RETRACTED, this study is invalid. Do not cite or reuse its results.

> **The paper and every result in this repository are withdrawn.** The headline
> finding does not survive a corrected computation.
>
> This is **not** a presentation or rounding problem. Every table in the paper is
> faithfully reproduced by the code in `scripts/`, and `run_11_lipschitz.py`
> reproduces to four decimals. The computation itself was wrong.
>
> **Full analysis: [`RETRACTION.md`](RETRACTION.md).**

Repository: <https://github.com/dkrse/kan-real-photometric-noise>

The LaTeX source and PDF are not committed to this repository (see `.gitignore`).
The retracted paper exists only as its published record, which carries the same
retraction notice.

---

## Root cause, in one paragraph

**The noise injection never perturbed the features that carried the label.**
`inject_real_noise` (`scripts/run_10_real_noise.py:201-206`) perturbs only the
leading magnitude columns (`X[:, :n_noisy]`), while the feature vector is built as
`mag_cols + other_cols`. Everything after the magnitude bands survives every noise
level intact, and those are the features that determine the class:

1. **Spectroscopic `redshift` is an input feature in both surveys**, and it nearly
   determines the label (median by class: STAR −0.0001, GALAXY 0.456, QSO 1.617).
   On the paper's own SDSS split, a depth-6 tree on **redshift alone scores 0.9470,
   above the published KAN (0.9347) and MLP-Reg (0.9355) using all six features.**
   Photometry alone (`u,g,r,i,z`) scores 0.8043. The task is described throughout
   as *photometric* classification; it is not.
2. **DESI `bp`/`rp` are a class-correlated sentinel.** A missing Gaia magnitude is
   stored as `0.0`, not NaN, and `between(0, 40)` accepts it. `bp == 0` for 56.6%
   of rows: GALAXY 96.7%, QSO 67.2%, **STAR 5.9%**. The rule "`bp != 0` ⇒ STAR"
   alone scores 0.86 accuracy.

At any noise level, **SDSS kept 1 of 6 features clean; DESI kept 3 of 6.** The
entire paper is built on the contrast "KAN edge on SDSS, null on DESI", that
contrast is an artifact of this asymmetry.

## What the corrected computation gives

`scripts/run_14_no_leak.py` re-runs the equal-baseline on photometry only, so every
input is a magnitude and **every input is perturbed** (and weight decay is tuned on
a validation split, not on test; see `RETRACTION.md`, defect S1).

| | published (invalid) | corrected |
|---|---|---|
| SDSS KAN clean accuracy | 0.9347 | **0.7813** |
| DESI KAN clean accuracy | 0.9306 | **0.8204** |
| **KAN − MLP-Reg gap @ α=10, SDSS** | **+4.41 p.p.** | **+1.53 p.p.** |
| **KAN − MLP-Reg gap @ α=10, DESI** | **−0.52 p.p.** | **+1.06 p.p.** |

15.3 p.p. (SDSS) and 11.0 p.p. (DESI) of the published "clean accuracy" was the
unperturbed leaked features. **The DESI null does not merely shrink, it changes
sign.** Published, the contrast was +4.4 vs −0.5 (opposite signs); corrected, it is
+1.5 vs +1.1 (same sign, comparable size).

### Claim-by-claim

| Published claim | Verdict |
|---|---|
| Equal-baseline equivalence survives real noise at α ≤ 2 | **Survives.** Corrected SDSS gaps at α ≤ 2 are within ±0.07 p.p., indistinguishable. The one honest result, and cleaner than published, since it now holds on genuine photometry. |
| KAN architectural advantage, SDSS-only extreme, absent on DESI | **Falsified as stated.** |
| Band-ablation pins the cause to the high-σ `u` band | **Not re-run; unsafe.** `u` is not special for being ultraviolet, it simply has the largest σ (0.48 mag, vs r 0.076, vs DESI ~0.008). |
| The edge is a quasar effect (+13 p.p. QSO F1) | **Not re-run; unsafe.** Computed with redshift present, and redshift *is* the quasar discriminator (median z = 1.62). |
| Jacobian mechanism (~10 to 22× suppression) | **Partly falsified.** Suppression survives qualitatively (~5.5 to 11.8×), but the predictive part fails: corrected, DESI has the *larger* relative KAN Jacobian advantage yet the *smaller* edge, the opposite of the paper's logic. |

**The corrected numbers are one training seed, 5 noise trials, and are not
sufficient to support a replacement claim.** The published headline used 5 seeds
with std ±1.67 p.p. at α = 10; the corrected SDSS +1.53 p.p. sits inside that
spread and cannot yet be distinguished from zero. See `RETRACTION.md` for the full
limits.

## Repository status, read before reusing anything

**The original scripts are NOT fixed and the original results are NOT corrected.**

| Path | Status |
|---|---|
| `scripts/run_10_real_noise.py` | **Unfixed.** Leaky feature builders; never-perturbed non-photometric columns; test-set wd tuning; broken cache key. |
| `scripts/run_11_lipschitz.py`, `run_12_ablation_perclass.py`, `run_13_seed_stats.py` | **Unfixed.** All import the leaky builders from `run_10`. |
| `scripts/extract_desi_errors.py` | **Unfixed.** Emits the `0.0` bp/rp sentinel. |
| `output/results/*.csv` (pre-existing) | **Invalid.** Superseded, not regenerated. |
| `scripts/run_14_no_leak.py` | **New, leakage-free.** The only trustworthy script here. |
| `output/results/noleak_*.csv` | **New.** Corrected Variant B + Jacobian only. |

Secondary defects, independent of the root cause, are documented in
[`RETRACTION.md`](RETRACTION.md): test-set weight-decay tuning, non-unique `obj_ID`
(36.6% of SDSS rows share an error vector with a different object), a model cache
that ignores training configuration, and a Lipschitz column that is a rescaling of
the Jacobian rather than independent evidence.

## Reproducing the diagnosis

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python scripts/run_14_no_leak.py     # leakage-free equal-baseline + Jacobian
```

The two checks that expose the root cause directly:

```python
# 1. redshift alone beats the published models, on the published split
from sklearn.tree import DecisionTreeClassifier   # -> 0.9470 vs published KAN 0.9347

# 2. the DESI Gaia sentinel
import pandas as pd
d = pd.read_csv('data/desi_dr1_sample_with_errors.csv')
d.groupby('class')['bp'].apply(lambda s: (s == 0).mean())
# GALAXY 0.967 | QSO 0.672 | STAR 0.059
```

## Repository layout

```
data/        SDSS/DESI inputs + per-object uncertainties (+ checksums)
scripts/     pipeline (fetch/extract + run_10..run_13, all unfixed) + run_14_no_leak.py
output/      results/ (CSV), figures/ (PNG), models/ (cached KANs, gitignored: >100MB)
INSTALL.md   environment setup
requirements.txt
```

## Lesson

Checking outputs against outputs cannot detect this class of error, the tables all
matched. The defect was visible only from reconstructing the input matrix, asking
what each column physically is, and running a trivial single-feature baseline. That
takes seconds and would have caught it before any model was trained.

## Author

krse

## Citation

**Do not cite this work.** The companion study *KANs as Implicit Regularizers*
(arXiv:2605.29039) is a separate work and is not affected by this retraction; note
only that it relies on synthetic uniform-SNR noise, not on the pipeline retracted
here.

## License

See [`LICENSE`](LICENSE). SDSS DR17 and DESI DR1 data are public under their
respective collaboration policies.
