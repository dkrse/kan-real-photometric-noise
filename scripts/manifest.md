# ⚠️ RETRACTED. The outputs mapped here are invalid.

> The study is withdrawn. Every pre-existing CSV and figure listed below was
> produced by a pipeline whose noise injection never perturbed the features that
> carried the label. The files are superseded, not regenerated.
>
> This file is retained as a record of what produced what.
> **Read [`../RETRACTION.md`](../RETRACTION.md) first.**

# Manifest: real-photometric-noise experiment

Data → script → output map of the retracted study. The LaTeX source is not
committed to this repository (see `.gitignore`).

## Datasets

| File | Source | Rows | Role |
|------|--------|------|------|
| `data/star_classification.csv` | SDSS DR17 (Kaggle/fedesoriano) | 100,000 | SDSS magnitudes + `obj_ID` join key |
| `data/sdss_errors.csv` | SkyServer `PhotoObjAll` | 60,411 | per-object `psfMagErr_{u..z}` |
| `data/desi_dr1_sample_with_errors.csv` | DESI DR1 `zpix` FITS | 99,999 | DESI mags + `g/r/z` σ (balanced sample) |
| `data/zpix-main-{bright,dark}.fits` | DESI DR1 | n/a | raw catalogues (8.8 + 10.5 GB, external) |

> **⚠️ Data caveats.** `obj_ID` is not unique in `star_classification.csv` (100,000
> rows, 78,053 unique), so the join to `sdss_errors.csv` is one-to-many and 36.6%
> of usable rows carry an error vector belonging to a different object.
> `desi_dr1_sample_with_errors.csv` stores a missing Gaia magnitude as `0.0`, not
> NaN: `bp == 0` for 56.6% of rows, class-correlated (GALAXY 96.7%, QSO 67.2%,
> STAR 5.9%).

## Pipeline

| Script | Phase | Status | Output |
|--------|-------|--------|--------|
| `fetch_sdss_errors.py` | 0 | ok | `data/sdss_errors.csv` |
| `extract_desi_errors.py` | 0 | **unfixed** (emits the 0.0 bp/rp sentinel) | `data/desi_dr1_sample_with_errors.csv` |
| `run_10_real_noise.py` | 1 | **unfixed, outputs invalid** | `output/results/real_noise_variant{A,B,C}_{sdss,desi}.csv`, `output/figures/real_noise_variant{B,C}_{sdss,desi}.png` |
| `run_11_lipschitz.py` | 2 | **unfixed, outputs invalid** | `output/results/lipschitz_metrics.csv` |
| `run_12_ablation_perclass.py` | 3 | **unfixed, outputs invalid** | `output/results/ablation_bands_sdss.csv`, `output/results/perclass_realnoise_{sdss,desi}.csv`, `output/figures/{ablation_bands_sdss,perclass_realnoise}.png` |
| `run_13_seed_stats.py` | 4 | **unfixed, outputs invalid** | `output/results/seed_stats_gap_{sdss,desi}.csv`, `output/results/seed_stats_summary.csv`, `output/figures/seed_stats_gap.png` |
| `run_14_no_leak.py` | 5 | **leakage-free, the only trustworthy script here** | `output/results/noleak_variantB_{sdss,desi}.csv`, `output/results/noleak_summary.csv`, `output/results/noleak_jacobian.csv` |

`run_11` through `run_13` all import the feature builders from `run_10`, so they
inherit its defect.

## Models (per `run_10`/`run_11`)

- **KAN 2.0** `[n,24,12,3]`, cubic B-splines, grid 5, L1 λ=0.001, 200 steps.
- **MLP** `[n,64,64,3]` SiLU; **MLP-Reg** = MLP with weight decay grid-searched to
  match KAN clean accuracy (equal-baseline control); **XGBoost** 200 trees, depth 6.
- Real-noise injection: per-object σ scaled by α ∈ {0.5,1,2,5,10} on the magnitude bands.

> **⚠️ "on the magnitude bands" is the defect.** Only the magnitude bands carry a
> σ, so only they were perturbed. The remaining features (`redshift` in both
> surveys, `bp`/`rp` on DESI) stayed clean at every α and are precisely the ones
> that determine the class. In `run_10` the MLP-Reg weight decay was also
> grid-searched against accuracy on the test set, which was then used to report
> every result.

## Results

**Withdrawn.** The headline results table previously here is invalid. See
[`../RETRACTION.md`](../RETRACTION.md) for the claim-by-claim verdict and for what
`run_14_no_leak.py` gives instead.

## Integrity

`data/checksums.sha256` covers the three CSV data files. Verify with
`sha256sum -c data/checksums.sha256` (run from `data/`). The checksums attest that
the input files are unmodified; they say nothing about whether the analysis built
on them was correct.

## Data license

SDSS data: public under [SDSS DR policies](https://www.sdss.org/collaboration/citing-sdss/);
Kaggle compilation by [fedesoriano](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17).
DESI DR1: public at <https://data.desi.lbl.gov/public/dr1/>.
