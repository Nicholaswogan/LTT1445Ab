from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# pscomppars columns (NASA Exoplanet Archive)
DEFAULT_COLUMNS = [
    "pl_name",
    "hostname",
    "pl_rade",
    "pl_masse",
    "pl_eqt",
    "pl_trandur",
    "pl_orbsmax",
    "st_rad",
    "st_teff",
    "st_met",
    "st_logg",
    "sy_kmag",
]


INTERNAL_TO_NEA = {
    "TRAPPIST-1c": "TRAPPIST-1 c",
    "TRAPPIST-1b": "TRAPPIST-1 b",
    "LHS1140c": "LHS 1140 c",
    "LTT1445Ab": "LTT 1445 A b",
    "GJ357b": "GJ 357 b",
    "L9859c": "L 98-59 c",
    "TOI270b": "TOI-270 b",
    "GJ3929b": "GJ 3929 b",
    "GJ1132b": "GJ 1132 b",
    "LHS1478b": "LHS 1478 b",
    "TOI1468b": "TOI-1468 b",
    "GJ486b": "GJ 486 b",
    "HD260655b": "HD 260655 b",
    "GJ3473b": "GJ 3473 b",
}

def fetch_pscomppars(
    planet_names: list[str],
    *,
    columns: list[str] | None = None,
    table: str = "pscomppars",
) -> pd.DataFrame:
    cols = columns or DEFAULT_COLUMNS
    if not planet_names:
        return pd.DataFrame(columns=cols)

    def _sql_quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    names_sql = ",".join(_sql_quote(name) for name in planet_names)
    query = f"select {','.join(cols)} from {table} where pl_name in ({names_sql})"
    url = f"{TAP_SYNC_URL}?query={quote_plus(query)}&format=csv"
    return pd.read_csv(url)


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return float(value)


def main() -> int:
    internal_names = list(INTERNAL_TO_NEA.keys())
    nea_names = [INTERNAL_TO_NEA[n] for n in internal_names]

    df = fetch_pscomppars(nea_names)
    rows_by_name = {row["pl_name"]: row for _, row in df.iterrows()}

    missing = [n for n in nea_names if n not in rows_by_name]
    if missing:
        raise SystemExit(f"Missing planets from NEA query: {missing}")

    cache: dict[str, dict] = {}
    for internal in internal_names:
        nea_name = INTERNAL_TO_NEA[internal]
        row = rows_by_name[nea_name]

        # NEA: pl_trandur is in hours; convert to seconds.
        trandur_h = _to_float(row.get("pl_trandur"))
        trandur_s = None if trandur_h is None else trandur_h * 3600.0

        cache[internal] = {
            "nea_name": nea_name,
            "hostname": row.get("hostname"),
            "planet": {
                "radius_re": _to_float(row.get("pl_rade")),
                "mass_me": _to_float(row.get("pl_masse")),
                "teq_k": _to_float(row.get("pl_eqt")),
                "transit_duration_s": trandur_s,
                "a_au": _to_float(row.get("pl_orbsmax")),
            },
            "star": {
                "radius_rsun": _to_float(row.get("st_rad")),
                "teff_k": _to_float(row.get("st_teff")),
                "metal": _to_float(row.get("st_met")),
                "logg": _to_float(row.get("st_logg")),
                "kmag": _to_float(row.get("sy_kmag")),
            },
        }

    outpath = Path("data/nea_planet_cache.json")
    outpath.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {outpath} with {len(cache)} planets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
