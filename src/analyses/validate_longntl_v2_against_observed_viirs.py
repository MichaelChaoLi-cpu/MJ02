#!/usr/bin/env python3
"""Validate LongNTL V2 against the accepted observed-VIIRS overlap panel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LONGNTL = ROOT / "data/processed/longntl_v2_historical_boundary_climate_preprocessed.parquet"
VIIRS = ROOT / "data/processed/viirs_historical_boundary_climate_preprocessed.parquet"
OUTPUT = ROOT / "data/exp/longntl-v2-overlap-validation"

KEYS = ["Grid Row", "Grid Column", "Year"]
NEW_RAW = "Annual NPP-VIIRS-like Radiance"
NEW_ASINH = "Asinh Annual NPP-VIIRS-like Radiance"
OLD_RAW = "Annual Mean Radiance"
OLD_ASINH = "Asinh Annual Mean Radiance"
SUPPORT = "Historical-Boundary Common Support 5 km"


def summarize(group: pd.DataFrame, scope: str, year: int | str, light_sample: str) -> dict:
    selected = group.copy()
    if light_sample == "positive in either product":
        selected = selected.loc[selected[NEW_RAW].gt(0) | selected[OLD_RAW].gt(0)]
    return {
        "scope": scope,
        "year": year,
        "light_sample": light_sample,
        "cell_years": len(selected),
        "longntl_positive_share": selected[NEW_RAW].gt(0).mean(),
        "observed_viirs_positive_share": selected[OLD_RAW].gt(0).mean(),
        "longntl_mean_asinh": selected[NEW_ASINH].mean(),
        "observed_viirs_mean_asinh": selected[OLD_ASINH].mean(),
        "pearson_asinh": selected[NEW_ASINH].corr(selected[OLD_ASINH]),
        "spearman_asinh": selected[NEW_ASINH].rank().corr(selected[OLD_ASINH].rank()),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    longntl_columns = KEYS + [NEW_RAW, NEW_ASINH, SUPPORT]
    viirs_columns = KEYS + [OLD_RAW, OLD_ASINH, SUPPORT]
    longntl = pd.read_parquet(LONGNTL, columns=longntl_columns)
    longntl = longntl.loc[longntl["Year"].between(2013, 2021)]
    viirs = pd.read_parquet(VIIRS, columns=viirs_columns)
    merged = longntl.merge(
        viirs,
        on=KEYS,
        how="outer",
        suffixes=("_longntl", "_viirs"),
        indicator=True,
        validate="one_to_one",
    )
    key_audit = pd.DataFrame(
        [
            {
                "longntl_rows": len(longntl),
                "observed_viirs_rows": len(viirs),
                "matched_rows": int(merged["_merge"].eq("both").sum()),
                "longntl_only_rows": int(merged["_merge"].eq("left_only").sum()),
                "observed_viirs_only_rows": int(merged["_merge"].eq("right_only").sum()),
            }
        ]
    )
    key_audit.to_csv(OUTPUT / "overlap_key_audit.csv", index=False)
    if not merged["_merge"].eq("both").all():
        raise RuntimeError("LongNTL and observed VIIRS do not have complete overlap keys")
    merged = merged.drop(columns="_merge")
    support_long = f"{SUPPORT}_longntl"
    support_old = f"{SUPPORT}_viirs"
    if not merged[support_long].eq(merged[support_old]).all():
        raise RuntimeError("Five-kilometre support differs between products")

    rows = []
    for scope, frame in (
        ("full research crop", merged),
        ("frozen 5 km support", merged.loc[merged[support_long].eq(1)]),
    ):
        for light_sample in ("all cells", "positive in either product"):
            rows.append(summarize(frame, scope, "pooled 2013-2021", light_sample))
            for year, year_frame in frame.groupby("Year", observed=True):
                rows.append(summarize(year_frame, scope, int(year), light_sample))
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(OUTPUT / "overlap_distribution_and_correlation.csv", index=False)

    readme = f"""# LongNTL V2 and observed-VIIRS overlap validation

- The comparison uses exact common grid-row, grid-column, and year keys for 2013-2021.
- LongNTL uses the authors' annual-composite and background-screening construction; EOG VNL V2.1
  uses a different official annual processing chain. Equality of levels or zero masks is not expected.
- Correlations are reported for all cells and for cells positive in either product, both for the full
  crop and the frozen 5 km boundary support.
- This audit evaluates measurement compatibility only. It does not estimate historical-side effects.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(key_audit.to_string(index=False))
    print(
        diagnostics.loc[
            diagnostics["year"].eq("pooled 2013-2021"),
            ["scope", "light_sample", "cell_years", "pearson_asinh", "spearman_asinh"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
