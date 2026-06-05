# Scripts: real-photometric-noise experiment

Pipeline for `paper/paper.tex`. Run with the project-local venv
(`../.venv/bin/python`). Only scripts used by this experiment are kept.

| Script | Phase | Input | Output |
|--------|-------|-------|--------|
| `fetch_sdss_errors.py` | 0 | `data/star_classification.csv` (obj_ID) | `data/sdss_errors.csv` (psfMagErr via SkyServer `PhotoObjAll`) |
| `extract_desi_errors.py` | 0 | `data/zpix-main-{bright,dark}.fits` | `data/desi_dr1_sample_with_errors.csv` (σ from FLUX_IVAR) |
| `run_10_real_noise.py` | 1 | both data products above | `output/results/real_noise_variant{A,B,C}_{sdss,desi}.csv`, `output/figures/real_noise_variant{B,C}_{sdss,desi}.png` |
| `run_11_lipschitz.py` | 2 | imports `run_10` (data builders + trainers); cached KAN models | `output/results/lipschitz_metrics.csv` |
| `run_12_ablation_perclass.py` | 3 | imports `run_10`; cached KAN models | `output/results/ablation_bands_sdss.csv`, `output/results/perclass_realnoise_{sdss,desi}.csv`, `output/figures/{ablation_bands_sdss,perclass_realnoise}.png` |
| `run_13_seed_stats.py` | 4 | imports `run_10`; trains 5 seeds/survey | `output/results/seed_stats_gap_{sdss,desi}.csv`, `output/results/seed_stats_summary.csv`, `output/figures/seed_stats_gap.png` |

`run_11` imports `run_10_real_noise` for its data builders and model trainers;
trained KANs are cached under `output/models/real_noise/` so re-runs are fast.

## Run order

```bash
cd ..
.venv/bin/python scripts/fetch_sdss_errors.py     # needs internet (SkyServer)
.venv/bin/python scripts/extract_desi_errors.py   # needs cached DESI FITS
.venv/bin/python scripts/run_10_real_noise.py     # Variants A/B/C, SDSS + DESI
.venv/bin/python scripts/run_11_lipschitz.py      # Jacobian / Lipschitz mechanism
```

Set `SMOKE=1` for a fast reduced-epoch end-to-end check of `run_10`.

## Notes

- KAN training is CPU-bound (~460 s per fit); `run_10` trains 8 KANs (4 per
  survey) plus the MLP-Reg weight-decay grids; budget ~1 to 1.5 h for the full run.
- `run_10`/`run_11` rebuild all features from the CSV/FITS sources; no `.npz`
  preprocessing artifact is needed.

See `manifest.md` for the full data→script→output map and headline results, and
`methodology.md` for the detailed methodology (kept in sync with the paper).
