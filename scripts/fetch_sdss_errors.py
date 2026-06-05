#!/usr/bin/env python3
"""fetch_sdss_errors — pull per-object photometric uncertainties for SDSS DR17.

The Kaggle SDSS catalogue (`data/star_classification.csv`) carries magnitudes
but no magnitude errors. This script joins each `obj_ID` to SDSS DR17
`PhotoObjAll` via the SkyServer SQL web service and retrieves the per-band
PSF magnitude errors, writing them to `data/sdss_errors.csv`.

Phase 0 of the real-photometric-noise extension. Re-run is idempotent: already
fetched obj_IDs are skipped if an output file is present.

Usage:
    python scripts/fetch_sdss_errors.py
"""

import sys
import time
import urllib.parse
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_CSV = DATA_DIR / "star_classification.csv"
OUT_CSV = DATA_DIR / "sdss_errors.csv"

SKYSERVER = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
ERR_COLS = ["psfMagErr_u", "psfMagErr_g", "psfMagErr_r", "psfMagErr_i", "psfMagErr_z"]
BATCH = 250           # obj_IDs per SQL query (IN-clause; URL length limited)
MAX_RETRY = 6
PAUSE = 1.0           # polite delay between requests (s); service throttles bursts


def run_query(sql: str) -> pd.DataFrame:
    """Execute one SkyServer SQL query, return rows as a DataFrame."""
    url = SKYSERVER + "?" + urllib.parse.urlencode({"cmd": sql, "format": "csv"})
    last_err = None
    for attempt in range(MAX_RETRY):
        try:
            raw = urllib.request.urlopen(url, timeout=60).read().decode()
            # SkyServer prefixes a "#Table1" comment line before the header.
            lines = [ln for ln in raw.splitlines() if not ln.startswith("#")]
            return pd.read_csv(StringIO("\n".join(lines)))
        except Exception as e:  # network / service hiccups -> backoff
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"query failed after {MAX_RETRY} retries: {last_err}")


def main() -> None:
    df = pd.read_csv(SRC_CSV, usecols=["obj_ID"])
    obj_ids = df["obj_ID"].drop_duplicates().astype("int64").tolist()
    print(f"{len(obj_ids)} unique obj_IDs to fetch")

    done: set[int] = set()
    parts: list[pd.DataFrame] = []
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        parts.append(prev)
        done = set(prev["objID"].astype("int64"))
        print(f"resuming: {len(done)} already in {OUT_CSV.name}")

    todo = [o for o in obj_ids if o not in done]
    n_batches = (len(todo) + BATCH - 1) // BATCH
    cols = ", ".join(ERR_COLS)

    for bi in range(0, len(todo), BATCH):
        chunk = todo[bi:bi + BATCH]
        id_list = ",".join(str(o) for o in chunk)
        sql = f"SELECT objID, {cols} FROM PhotoObjAll WHERE objID IN ({id_list})"
        res = run_query(sql)
        parts.append(res)
        got = sum(len(p) for p in parts)
        print(f"batch {bi // BATCH + 1}/{n_batches}  +{len(res)} rows  total={got}",
              flush=True)
        # checkpoint every 10 batches so a crash loses little
        if (bi // BATCH) % 10 == 9:
            pd.concat(parts, ignore_index=True).drop_duplicates("objID").to_csv(
                OUT_CSV, index=False)
        time.sleep(PAUSE)

    out = pd.concat(parts, ignore_index=True).drop_duplicates("objID")
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(out)} rows -> {OUT_CSV}")
    miss = len(obj_ids) - len(out)
    if miss:
        print(f"warning: {miss} obj_IDs returned no PhotoObjAll match", file=sys.stderr)


if __name__ == "__main__":
    main()
