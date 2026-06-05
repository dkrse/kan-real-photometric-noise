#!/usr/bin/env python3
"""extract_desi_errors — build a DESI DR1 sample with per-object photometric errors.

The companion paper's DESI sample carried magnitudes but no uncertainties and no
join key back to the FITS catalogue. This script extracts a balanced sample
directly from the cached `zpix-main-{bright,dark}.fits`, keeping the per-band
magnitude uncertainties derived from the inverse-variance columns:

    flux  [nanomaggies]      = FLUX_<band>
    sigma_flux               = 1 / sqrt(FLUX_IVAR_<band>)
    mag                      = 22.5 - 2.5*log10(flux)
    sigma_mag                = 1.0857 * sigma_flux / flux

Uncertainties are available for the Legacy Surveys g, r, z bands. Gaia BP/RP
magnitudes have no per-object error in the zcatalog, so they are kept without a
sigma column (documented limitation).

Output: `data/desi_dr1_sample_with_errors.csv`
        (the original sample is left untouched.)

Phase 0 of the real-photometric-noise extension.

Usage:
    python scripts/extract_desi_errors.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_CSV = DATA_DIR / "desi_dr1_sample_with_errors.csv"

FITS_FILES = [DATA_DIR / "zpix-main-bright.fits", DATA_DIR / "zpix-main-dark.fits"]
PER_CLASS = 33333          # balanced sample, matches the original
SEED = 42
SPECTYPE_MAP = {"GALAXY": "GALAXY", "QSO": "QSO", "STAR": "STAR"}


def mag_and_err(flux, ivar):
    """Vector flux/ivar -> (mag, sigma_mag); invalid entries become NaN."""
    flux = np.asarray(flux, dtype=np.float64)
    ivar = np.asarray(ivar, dtype=np.float64)
    good = (flux > 0) & (ivar > 0)
    mag = np.full(flux.shape, np.nan)
    err = np.full(flux.shape, np.nan)
    mag[good] = 22.5 - 2.5 * np.log10(flux[good])
    err[good] = 1.0857 / (np.sqrt(ivar[good]) * flux[good])
    return mag, err


def load_fits(path: Path) -> pd.DataFrame:
    print(f"reading {path.name} ...", flush=True)
    with fits.open(path, memmap=True) as hdul:
        d = hdul[1].data
        spectype = np.char.strip(d["SPECTYPE"].astype(str))
        zwarn = d["ZWARN"]
        primary = d["ZCAT_PRIMARY"]
        g, ge = mag_and_err(d["FLUX_G"], d["FLUX_IVAR_G"])
        r, re = mag_and_err(d["FLUX_R"], d["FLUX_IVAR_R"])
        z, ze = mag_and_err(d["FLUX_Z"], d["FLUX_IVAR_Z"])
        df = pd.DataFrame({
            "class": spectype,
            "redshift": d["Z"].astype(np.float64),
            "g": g, "r": r, "z_mag": z,
            "g_err": ge, "r_err": re, "z_err": ze,
            "bp": d["GAIA_PHOT_BP_MEAN_MAG"].astype(np.float64),
            "rp": d["GAIA_PHOT_RP_MEAN_MAG"].astype(np.float64),
            "zwarn": zwarn.astype(np.int64),
            "primary": primary.astype(bool),
        })
    # quality cuts: primary spectrum, no redshift warning, valid g/r/z mags+errs
    df = df[df["primary"] & (df["zwarn"] == 0)]
    df = df[df["class"].isin(SPECTYPE_MAP)]
    df = df.dropna(subset=["g", "r", "z_mag", "g_err", "r_err", "z_err"])
    return df.drop(columns=["zwarn", "primary"])


def main() -> None:
    frames = [load_fits(p) for p in FITS_FILES]
    cat = pd.concat(frames, ignore_index=True)
    print(f"combined clean rows: {len(cat)}")
    print(cat["class"].value_counts().to_string())

    rng = np.random.default_rng(SEED)
    parts = []
    for cls in ("GALAXY", "QSO", "STAR"):
        sub = cat[cat["class"] == cls]
        n = min(PER_CLASS, len(sub))
        idx = rng.choice(len(sub), size=n, replace=False)
        parts.append(sub.iloc[idx])
        if n < PER_CLASS:
            print(f"warning: only {n} {cls} available (< {PER_CLASS})")
    sample = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED)

    cols = ["class", "redshift", "g", "r", "z_mag", "bp", "rp",
            "g_err", "r_err", "z_err"]
    sample[cols].to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(sample)} rows -> {OUT_CSV}")
    print("median sigma_mag (g,r,z):",
          sample[["g_err", "r_err", "z_err"]].median().round(4).to_dict())


if __name__ == "__main__":
    main()
