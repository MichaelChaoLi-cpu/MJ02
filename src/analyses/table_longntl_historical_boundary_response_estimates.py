#!/usr/bin/env python3
"""LongNTL Historical Boundary Response Estimates.

Plan: Report full-period, source-stage, robustness, bandwidth, and mirrored-strip estimates.
Framework: AnaSOP Sections 5.5, 6.10, and 7 long-run nighttime validation workflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from longntl_boundary_estimation import OUTPUT as EXP_DIR, ROOT, estimate_all
from longntl_table_style import write_one_sheet


OUTPUT = ROOT / "data/exp/legacy-results/tables/Table_longntl_historical_boundary_response_estimates.xlsx"


def main() -> None:
    source = EXP_DIR / "longntl_boundary_response_estimates.csv"
    results = pd.read_csv(source) if source.exists() else estimate_all()
    table = results[
        [
            "outcome", "period", "source_stage", "support_type", "distance_range_km",
            "specification", "estimate", "common_standardized_estimate", "common_standardized_ci_low",
            "common_standardized_ci_high", "sesoi_classification", "observations",
            "grid_cells", "cross_side_communes", "interpretation",
        ]
    ].rename(columns={
        "outcome": "Outcome",
        "period": "Period",
        "source_stage": "Source stage",
        "support_type": "Support type",
        "distance_range_km": "Distance range km",
        "specification": "Specification",
        "estimate": "Natural estimate",
        "common_standardized_estimate": "Standardized estimate",
        "common_standardized_ci_low": "Standardized 95% CI low",
        "common_standardized_ci_high": "Standardized 95% CI high",
        "sesoi_classification": "SESOI comparison",
        "observations": "Observations",
        "grid_cells": "Grid cells",
        "cross_side_communes": "Cross-side communes",
        "interpretation": "Interpretation",
    })
    numeric = [
        "Natural estimate",
        "Standardized estimate", "Standardized 95% CI low", "Standardized 95% CI high",
    ]
    if table[numeric].isna().any().any():
        raise RuntimeError("Missing LongNTL coefficient or confidence interval")
    if not (table["Standardized 95% CI low"] <= table["Standardized estimate"]).all():
        raise RuntimeError("Standardized estimate below its confidence interval")
    if not (table["Standardized estimate"] <= table["Standardized 95% CI high"]).all():
        raise RuntimeError("Standardized estimate above its confidence interval")
    table["Compatibility frontier SD"] = table[
        ["Standardized 95% CI low", "Standardized 95% CI high"]
    ].abs().max(axis=1)
    columns = list(table.columns)
    columns.insert(columns.index("SESOI comparison"), columns.pop(columns.index("Compatibility frontier SD")))
    table = table[columns]
    widths = [31, 13, 28, 27, 17, 42, 15, 18, 19, 19, 20, 29, 14, 12, 19, 48]
    write_one_sheet(
        table,
        output=OUTPUT,
        sheet="Boundary Responses",
        title="LongNTL Historical Boundary Response Estimates",
        note="The 5 km full-period and within-modern-commune models are the local estimands. Cumulative models absorb grid-cell and segment-by-year fixed effects; confirmation additionally absorbs commune-by-year. Mirrored strips absorb grid-cell and segment-by-1-km-bin-by-year fixed effects. All uncertainty is two-way clustered by grid cell and district-by-year.",
        widths=widths,
        numeric_columns=set(numeric + ["Compatibility frontier SD"]),
    )
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Rows: {len(table)}; sheets: 1")
    print(table.loc[table["Specification"].isin(["Full period primary 5 km", "Full period within-commune 5 km"]),
                    ["Specification", "Standardized estimate", "Standardized 95% CI low", "Standardized 95% CI high"]].to_string(index=False))


if __name__ == "__main__":
    main()
