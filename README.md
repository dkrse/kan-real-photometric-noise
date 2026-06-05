# Does the Implicit-Regularizer View of KANs Survive Real Photometric Noise?

A heteroscedastic stress-test of Kolmogorov–Arnold Networks (KAN 2.0) against
regularized MLPs for stellar classification under **real per-object photometric
noise** on SDSS DR17 and DESI DR1.

Repository: <https://github.com/dkrse/kan-real-photometric-noise>

Companion to *KANs as Implicit Regularizers* (arXiv:2605.29039). The LaTeX
source under `paper/` is not committed (see `.gitignore`).

## TL;DR

A prior study showed that KAN's noise-robustness advantage over MLPs is an
*implicit-regularization* effect: an MLP given enough weight decay to match KAN's
clean accuracy (the *equal-baseline* protocol) closes the gap under synthetic noise.
Here we test that under **real catalogue uncertainties**.

- The equal-baseline equivalence **holds under realistic noise** (α ≤ 2) on both
  surveys; the regularized MLP is KAN's equal to within a few tenths of a point.
- A genuine KAN architectural advantage emerges **only in the SDSS heteroscedastic
  extreme** (+4.4 ± 1.7 p.p. at α = 10, p = 0.004 over 5 seeds); it is absent on DESI.
- A band-ablation pins the cause causally to the **high-σ u band** (excluding u
  removes the edge; a g,r,z subset reproduces the DESI null on SDSS objects).
- Per class, the edge is almost entirely a **quasar effect** (+13 p.p. QSO F1).
- A direct Jacobian measurement gives the mechanism: weight decay and the KAN
  architecture both cut the input→logit Jacobian ~10–22× versus an unregularized MLP.

## Repository layout

```
data/        SDSS/DESI inputs + Phase-0 per-object uncertainties (+ checksums)
scripts/     pipeline (fetch/extract + run_10..run_13) + README/manifest/methodology
paper/       paper.tex / .pdf (LaTeX source; not committed, see .gitignore)
output/      results/ (CSV), figures/ (PNG), models/ (cached KANs, gitignored: >100MB)
INSTALL.md   environment setup + step-by-step reproduction
requirements.txt
```

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Phase 0: real per-object uncertainties
.venv/bin/python scripts/fetch_sdss_errors.py     # SDSS psfMagErr via SkyServer
.venv/bin/python scripts/extract_desi_errors.py   # DESI sigma from FLUX_IVAR

# Phase 1-4: experiments
.venv/bin/python scripts/run_10_real_noise.py        # Variants A/B/C
.venv/bin/python scripts/run_11_lipschitz.py         # Jacobian/Lipschitz mechanism
.venv/bin/python scripts/run_12_ablation_perclass.py # band-ablation + per-class
.venv/bin/python scripts/run_13_seed_stats.py        # multi-seed statistics
```

Full instructions, data download, and notes are in [`INSTALL.md`](INSTALL.md).
See `scripts/manifest.md` for the data→script→output map and headline numbers.

## Headline results

| Experiment | SDSS | DESI |
|-----------|------|------|
| Equal-baseline gap @ α=1 (KAN − MLP-Reg) | ≈ 0 | ≈ 0 |
| Gap @ α=10 (5 seeds) | **+4.4 ± 1.7 p.p.** (p=0.004) | −0.5 ± 0.2 p.p. (p=0.004) |
| Band-ablation gap @ α=10 | u_only +8.6 / all5 +4.6 / no_u −1.7 / grz −0.8 p.p. | (grz reproduces DESI null) |
| Per-class edge @ α=10 | QSO +13.3 / GALAXY +3.3 / STAR −3.3 p.p. | within ±0.5 p.p. |
| Jacobian mean (KAN / MLP-Reg / MLP) | 3.06 / 5.58 / 57.7 | 3.11 / 6.97 / 68.3 |

## Author

krse

## Citation

If you use this code or data, please cite the paper (preprint forthcoming) and the
companion *KANs as Implicit Regularizers* (arXiv:2605.29039).

## License

See [`LICENSE`](LICENSE). SDSS DR17 and DESI DR1 data are public under their
respective collaboration policies.
