#!/usr/bin/env python3
"""LongNTL Measurement Validation.

Plan: Audit fixed-grid balance, observed-VIIRS overlap, and the 2012-2013 source transition.
Framework: AnaSOP Sections 5.5, 6.10, and 7 measurement gate before coefficient interpretation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from longntl_boundary_estimation import ABS_DISTANCE, CELL, OUTCOME, ROOT
from longntl_table_style import write_one_sheet


PANEL = ROOT / "data/processed/longntl_v2_historical_boundary_climate_preprocessed.parquet"
OVERLAP = ROOT / "data/exp/longntl-v2-overlap-validation/overlap_distribution_and_correlation.csv"
KEY_AUDIT = ROOT / "data/exp/longntl-v2-overlap-validation/overlap_key_audit.csv"
OUTPUT = ROOT / "data/exp/legacy-results/tables/Table_longntl_measurement_validation.xlsx"
EXP_OUTPUT = ROOT / "data/exp/longntl-v2-boundary-experiment/longntl_measurement_validation.csv"


def coverage_rows(panel: pd.DataFrame) -> list[dict[str, object]]:
    definitions = [
        ("Full research crop", panel, "2000-2024", "mixed"),
        ("Frozen 5 km support", panel.loc[panel["Historical-Boundary Common Support 5 km"].eq(1)], "2000-2024", "mixed"),
        ("Frozen 5 km support", panel.loc[panel["Historical-Boundary Common Support 5 km"].eq(1) & panel["Year"].le(2012)], "2000-2012", "reconstructed"),
        ("Frozen 5 km support", panel.loc[panel["Historical-Boundary Common Support 5 km"].eq(1) & panel["Year"].ge(2013)], "2013-2024", "observed annual composite"),
    ]
    rows = []
    for scope, sample, period, stage in definitions:
        years = sample["Year"].nunique()
        cells = sample[CELL].nunique()
        expected = years * cells
        rows.append({
            "Diagnostic family": "Fixed-grid coverage",
            "Scope": scope,
            "Period": period,
            "Source stage": stage,
            "Observations": len(sample),
            "Grid cells": cells,
            "Positive share": sample["Any Positive Annual NPP-VIIRS-like Radiance"].mean(),
            "Pearson": np.nan,
            "Spearman": np.nan,
            "Transition rank": np.nan,
            "Pass rule": f"Exactly {expected:,} cell-years with no duplicate cell-year keys",
            "Status": "Pass" if len(sample) == expected and not sample.duplicated([CELL, "Year"]).any() else "Fail",
            "Interpretation": "Balanced fixed-grid panel; source zeros retained",
        })
    return rows


def overlap_rows() -> list[dict[str, object]]:
    overlap = pd.read_csv(OVERLAP)
    keys = pd.read_csv(KEY_AUDIT).iloc[0]
    selections = [
        ("full research crop", "all cells"),
        ("full research crop", "positive in either product"),
        ("frozen 5 km support", "all cells"),
        ("frozen 5 km support", "positive in either product"),
    ]
    rows = []
    for scope, light_sample in selections:
        row = overlap.loc[
            overlap["scope"].eq(scope)
            & overlap["year"].astype(str).eq("pooled 2013-2021")
            & overlap["light_sample"].eq(light_sample)
        ].iloc[0]
        rows.append({
            "Diagnostic family": "Observed-VIIRS overlap",
            "Scope": f"{scope}; {light_sample}",
            "Period": "2013-2021",
            "Source stage": "LongNTL observed composite versus EOG VIIRS",
            "Observations": int(row["cell_years"]),
            "Grid cells": int(row["cell_years"] // 9),
            "Positive share": row["longntl_positive_share"],
            "Pearson": row["pearson_asinh"],
            "Spearman": row["spearman_asinh"],
            "Transition rank": np.nan,
            "Pass rule": "Exact overlap keys; report association without selecting on coefficient sign",
            "Status": "Pass" if keys["longntl_only_rows"] == 0 and keys["observed_viirs_only_rows"] == 0 else "Fail",
            "Interpretation": "Independent sensors are strongly associated in lit support; all-cell rank association is lower because zeros dominate",
        })
    return rows


def transition_rows(panel: pd.DataFrame) -> list[dict[str, object]]:
    values = panel.pivot(index=CELL, columns="Year", values=OUTCOME).sort_index(axis=1)
    positive = panel.pivot(index=CELL, columns="Year", values="Any Positive Annual NPP-VIIRS-like Radiance").sort_index(axis=1)
    records = []
    for year in range(2001, 2025):
        pair = values[[year - 1, year]].dropna()
        records.append({
            "from_year": year - 1,
            "to_year": year,
            "pearson": pair.corr(method="pearson").iloc[0, 1],
            "spearman": pair.corr(method="spearman").iloc[0, 1],
            "mean_change": values[year].mean() - values[year - 1].mean(),
            "positive_change": positive[year].mean() - positive[year - 1].mean(),
        })
    adjacent = pd.DataFrame(records)
    adjacent["absolute_mean_change_rank"] = adjacent["mean_change"].abs().rank(method="min", ascending=False).astype(int)
    rows = []
    for _, row in adjacent.loc[adjacent["to_year"].between(2010, 2015)].iterrows():
        is_transition = int(row["to_year"]) == 2013
        rows.append({
            "Diagnostic family": "Adjacent-year source transition" if is_transition else "Adjacent-year reference",
            "Scope": "Full research crop",
            "Period": f"{int(row['from_year'])}-{int(row['to_year'])}",
            "Source stage": "reconstructed to observed composite" if is_transition else "same-stage or neighboring reference",
            "Observations": len(values),
            "Grid cells": len(values),
            "Positive share": positive[int(row["to_year"])].mean(),
            "Pearson": row["pearson"],
            "Spearman": row["spearman"],
            "Transition rank": f"{int(row['absolute_mean_change_rank'])} of {len(adjacent)} by absolute mean change",
            "Pass rule": "Report 2012-2013 transition and compare with all adjacent-year changes",
            "Status": "Pass" if not is_transition or int(row["absolute_mean_change_rank"]) > 1 else "Review",
            "Interpretation": f"Mean asinh change {row['mean_change']:.4f}; positive-share change {row['positive_change']:.4f}",
        })
    return rows


def main() -> None:
    panel = pd.read_parquet(PANEL)
    if panel.duplicated([CELL, "Year"]).any():
        raise RuntimeError("Duplicate LongNTL cell-year keys")
    table = pd.DataFrame(coverage_rows(panel) + overlap_rows() + transition_rows(panel))
    if table["Status"].eq("Fail").any():
        raise RuntimeError("At least one LongNTL measurement validation failed")
    EXP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(EXP_OUTPUT, index=False)
    write_one_sheet(
        table,
        output=OUTPUT,
        sheet="Measurement Validation",
        title="LongNTL Measurement Validation",
        note="LongNTL Version 2 is balanced over 2000-2024. Correlations with observed EOG VIIRS are measurement diagnostics, not interchangeability claims. The 2012-2013 transition is compared with all adjacent-year changes before boundary estimates are interpreted.",
        widths=[24, 31, 14, 32, 14, 12, 15, 12, 12, 20, 41, 11, 54],
        numeric_columns={"Positive share", "Pearson", "Spearman"},
    )
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Rows: {len(table)}; sheets: 1")


if __name__ == "__main__":
    main()
