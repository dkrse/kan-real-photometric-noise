# Installation & Reproduction

Real-photometric-noise + Lipschitz experiment for
***Does the Implicit-Regularizer View of Kolmogorov–Arnold Networks Survive Real
Photometric Noise?*** (`paper/paper.tex`).

Everything runs from a project-local virtual environment; no dependency outside
this directory is required.

---

## 1. Requirements

- Linux, Python **3.12**
- ~25 GB free disk (the two DESI FITS catalogues are 8.8 + 10.5 GB)
- Internet access for: pip install, the SDSS SkyServer query (Phase 0), and the
  one-time DESI FITS download
- CPU is sufficient (no GPU needed; KAN training is CPU-bound)

## 2. Create the environment

```bash
cd /opt/apps/jupyter/work/p06
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Verify the install is fully self-contained (all imports resolve inside `.venv`):

```bash
.venv/bin/python -c "import torch, kan, xgboost, astropy, sklearn, pandas; \
print('env OK', torch.__version__)"
```

## 3. Obtain the input data

| File | How |
|------|-----|
| `data/star_classification.csv` | Already present (SDSS DR17, Kaggle). 100k objects with `obj_ID`. |
| `data/zpix-main-bright.fits` (8.8 GB) | Download once from DESI DR1 (URL below) into `data/`. |
| `data/zpix-main-dark.fits` (10.5 GB) | Download once from DESI DR1 (URL below) into `data/`. |

```bash
base=https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/zcatalog/v1
curl -o data/zpix-main-bright.fits $base/zpix-main-bright.fits
curl -o data/zpix-main-dark.fits  $base/zpix-main-dark.fits
```

Integrity of the CSV products can be checked any time:

```bash
cd data && sha256sum -c checksums.sha256 && cd ..
```

## 4. Run the pipeline

Run from the project root, in order:

```bash
# Phase 0: fetch real per-object photometric uncertainties
.venv/bin/python scripts/fetch_sdss_errors.py     # SDSS psfMagErr via SkyServer (~9 min, network; resumable)
.venv/bin/python scripts/extract_desi_errors.py   # DESI sigma from FLUX_IVAR (~2 min)

# Phase 1: Variants A/B/C on SDSS + DESI
.venv/bin/python scripts/run_10_real_noise.py     # ~1 to 1.5 h (8 KAN trainings on CPU)

# Phase 2: Jacobian / Lipschitz mechanism
.venv/bin/python scripts/run_11_lipschitz.py      # minutes (reuses cached KANs)

# Phase 3: band-ablation + per-class breakdown
.venv/bin/python scripts/run_12_ablation_perclass.py  # reuses cached KANs

# Phase 4: multi-seed statistics (5 seeds/survey, 20 noise trials each)
.venv/bin/python scripts/run_13_seed_stats.py     # trains 5 seeds/survey
```

### Quick smoke test

To validate the full pipeline end-to-end in a few minutes (reduced epochs/trials;
results are not publication-grade):

```bash
SMOKE=1 .venv/bin/python scripts/run_10_real_noise.py
SMOKE=1 .venv/bin/python scripts/run_11_lipschitz.py
```

### Notes & resumability

- `fetch_sdss_errors.py` is resumable: re-running skips `obj_ID`s already in
  `data/sdss_errors.csv`.
- `run_10_real_noise.py` caches trained KANs in `output/models/real_noise/`; a
  re-run loads them instead of retraining. Delete those `.pt` files to force a
  fresh training run.
- `run_11_lipschitz.py` imports `run_10_real_noise` for its data builders and
  trainers, and reuses the cached KANs.

## 5. Outputs

| Location | Contents |
|----------|----------|
| `output/results/real_noise_variant{A,B,C}_{sdss,desi}.csv` | Variant A/B/C tables |
| `output/results/lipschitz_metrics.csv` | Jacobian / Lipschitz metrics |
| `output/results/ablation_bands_sdss.csv` | Band-ablation gaps (run_12) |
| `output/results/perclass_realnoise_{sdss,desi}.csv` | Per-class F1 (run_12) |
| `output/results/seed_stats_gap_{sdss,desi}.csv`, `seed_stats_summary.csv` | Multi-seed statistics (run_13) |
| `output/figures/real_noise_variant{B,C}_{sdss,desi}.png` | Degradation & magnitude-bin figures |
| `output/figures/{ablation_bands_sdss,perclass_realnoise,seed_stats_gap}.png` | Ablation / per-class / seed figures |
| `output/models/real_noise/kan_{sdss,desi}_{base,aug}.pt` | Cached KAN models (gitignored: >100MB) |


See `scripts/README.md`, `scripts/manifest.md`, and `scripts/methodology.md` for
the full data→script→output map, model details, and headline results.
