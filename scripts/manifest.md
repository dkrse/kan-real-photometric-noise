# Manifest: real-photometric-noise experiment

Data → script → output map for `paper/paper.tex`.

## Datasets

| File | Source | Rows | Role |
|------|--------|------|------|
| `data/star_classification.csv` | SDSS DR17 (Kaggle/fedesoriano) | 100,000 | SDSS magnitudes + `obj_ID` join key |
| `data/sdss_errors.csv` | SkyServer `PhotoObjAll` | 60,411 | per-object `psfMagErr_{u..z}` |
| `data/desi_dr1_sample_with_errors.csv` | DESI DR1 `zpix` FITS | 99,999 | DESI mags + `g/r/z` σ (balanced sample) |
| `data/zpix-main-{bright,dark}.fits` | DESI DR1 | n/a | raw catalogues (8.8 + 10.5 GB, external) |

## Pipeline

| Script | Phase | Output |
|--------|-------|--------|
| `fetch_sdss_errors.py` | 0 | `data/sdss_errors.csv` |
| `extract_desi_errors.py` | 0 | `data/desi_dr1_sample_with_errors.csv` |
| `run_10_real_noise.py` | 1 | `output/results/real_noise_variant{A,B,C}_{sdss,desi}.csv`, `output/figures/real_noise_variant{B,C}_{sdss,desi}.png` |
| `run_11_lipschitz.py` | 2 | `output/results/lipschitz_metrics.csv` |
| `run_12_ablation_perclass.py` | 3 | `output/results/ablation_bands_sdss.csv`, `output/results/perclass_realnoise_{sdss,desi}.csv`, `output/figures/{ablation_bands_sdss,perclass_realnoise}.png` |
| `run_13_seed_stats.py` | 4 | `output/results/seed_stats_gap_{sdss,desi}.csv`, `output/results/seed_stats_summary.csv`, `output/figures/seed_stats_gap.png` |

## Models (per `run_10`/`run_11`)

- **KAN 2.0** `[n,24,12,3]`, cubic B-splines, grid 5, L1 λ=0.001, 200 steps.
- **MLP** `[n,64,64,3]` SiLU; **MLP-Reg** = MLP with weight decay grid-searched to
  match KAN clean accuracy (equal-baseline control); **XGBoost** 200 trees, depth 6.
- Real-noise injection: per-object σ scaled by α ∈ {0.5,1,2,5,10} on the magnitude bands.

## Headline results

| Experiment | SDSS | DESI | Conclusion |
|-----------|------|------|------------|
| Variant A (uncertainties as features) | neutral/slightly worse | neutral | no extra signal; demarcation from LSST-RF |
| Variant B equal-baseline, α=1 | KAN 0.9313 ≈ MLP-Reg 0.9315 | KAN 0.9305 ≈ MLP-Reg 0.9303 | equivalence survives real noise |
| Variant B extreme, α=10 | KAN 0.8200 > MLP-Reg 0.7741 | KAN ≈ MLP-Reg | KAN edge only in SDSS heteroscedastic extreme |
| Lipschitz (jac_mean) | KAN 3.06 / MLP-Reg 5.58 / MLP 57.7 | KAN 3.11 / MLP-Reg 6.97 / MLP 68.3 | implicit-regularizer confirmed mechanistically |
| Band-ablation gap @α=10 | u_only +8.6 / all5 +4.6 / no_u −1.7 / grz −0.8 p.p. | (grz reproduces DESI null) | u band causes the KAN edge; SDSS/DESI confound removed |
| Per-class edge @α=10 | QSO +13.3 / GALAXY +3.3 / STAR −3.3 p.p. | all within ±0.5 p.p. | SDSS edge is a quasar effect |
| Seed-robustness gap @α=10 | +4.41 ± 1.67 p.p. (p=0.004, 5 seeds) | −0.52 ± 0.20 p.p. (p=0.004) | SDSS edge seed-robust; DESI null confirmed |

## Integrity

`data/checksums.sha256` covers the three CSV data files. Verify with
`sha256sum -c data/checksums.sha256` (run from `data/`).

## Data license

SDSS data: public under [SDSS DR policies](https://www.sdss.org/collaboration/citing-sdss/);
Kaggle compilation by [fedesoriano](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17).
DESI DR1: public at <https://data.desi.lbl.gov/public/dr1/>.
