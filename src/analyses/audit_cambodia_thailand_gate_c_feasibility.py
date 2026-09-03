#!/usr/bin/env python3
"""Outcome-blind feasibility audit for post-conflict hazard amplification (Gate C)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data/processed/cambodia_national_annual_satellite_climate_panel_preprocessed.parquet"
WEIGHTS = ROOT / "data/exp/experiment-design/cambodia-thailand-gate-b/gate_b_binary_support_weights.parquet"
FLOOD_MANIFEST = ROOT / "data/raw/flood/global_flood_database/gfd_cambodia_event_manifest.csv"
OUT = ROOT / "data/exp/feasibility-check/cambodia-thailand-gate-c"

OUTCOMES = {
    "LongNTL economic activity": "Asinh Annual NPP-VIIRS-like Radiance",
    "Annual land NPP": "Annual Land NPP Anomaly Z 2001-2020",
}
HAZARDS = {
    "Drought": "May October Dry Rainfall Intensity",
    "Extreme rainfall": "May October Extreme Wet Rainfall Intensity",
    "Heat": "May October Heat Intensity",
    "Compound hot-dry": "May October Compound Hot-Dry Intensity",
}


def period(year: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [year.between(2000, 2007), year.between(2008, 2011), year.between(2012, 2024)],
            ["Pre-conflict 2000-2007", "Escalation 2008-2011", "Post-conflict 2012-2024"],
            default="Outside scope",
        ),
        index=year.index,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    weights = pd.read_parquet(WEIGHTS).copy()
    weights["Continuous Conflict Dose"] = np.where(
        weights["Conflict Distance Ring"].eq("Over 60 km"),
        0.0,
        np.maximum(
            0.0,
            1 - weights["Candidate Conflict Year 2011 Nearest Event Distance km"] / 60,
        ),
    )
    weights["Conflict Exposed"] = weights["Continuous Conflict Dose"].gt(0)

    columns = [
        "National Grid Cell ID",
        "Year",
        "Climate Cell ID",
        *OUTCOMES.values(),
        *HAZARDS.values(),
    ]
    panel = pd.read_parquet(PANEL, columns=columns)
    frame = panel.merge(
        weights[
            [
                "National Grid Cell ID",
                "Conflict Distance Ring",
                "Border Analysis Sector",
                "Spatial Block ID",
                "Gate B Binary Support Overlap Weight",
                "Continuous Conflict Dose",
                "Conflict Exposed",
            ]
        ],
        on="National Grid Cell ID",
        validate="many_to_one",
    )
    frame["Period"] = period(frame["Year"])

    availability_rows: list[dict[str, object]] = []
    for family, variable in {**OUTCOMES, **HAZARDS}.items():
        valid = frame.loc[frame[variable].notna()]
        availability_rows.append(
            {
                "Family": family,
                "Variable": variable,
                "First Year": int(valid["Year"].min()),
                "Last Year": int(valid["Year"].max()),
                "Valid Years": int(valid["Year"].nunique()),
                "Valid Grid Cells": int(valid["National Grid Cell ID"].nunique()),
                "Valid Cell-Years": len(valid),
                "Missing Share": float(frame[variable].isna().mean()),
            }
        )
    availability = pd.DataFrame(availability_rows)
    availability.to_csv(OUT / "dataset_availability.csv", index=False)

    cell_support = weights.groupby(["Conflict Distance Ring"], observed=True).agg(
        Grid_Cells=("National Grid Cell ID", "nunique"),
        Spatial_Blocks=("Spatial Block ID", "nunique"),
        Mean_Conflict_Dose=("Continuous Conflict Dose", "mean"),
    ).reset_index()
    cell_support.to_csv(OUT / "exposure_support.csv", index=False)

    grid_climate = frame.loc[frame["Year"].eq(2000), [
        "National Grid Cell ID", "Climate Cell ID", "Spatial Block ID",
        "Conflict Exposed", "Continuous Conflict Dose"
    ]].drop_duplicates("National Grid Cell ID")
    climate_support = grid_climate.groupby("Climate Cell ID").agg(
        Grid_Cells=("National Grid Cell ID", "nunique"),
        Exposed_Grid_Cells=("Conflict Exposed", "sum"),
        Conflict_Dose_SD=("Continuous Conflict Dose", "std"),
        Spatial_Blocks=("Spatial Block ID", "nunique"),
    ).reset_index()
    climate_summary = pd.DataFrame(
        [
            {
                "Gate B Grid Cells": len(weights),
                "Exposed Grid Cells": int(weights["Conflict Exposed"].sum()),
                "Frontier Control Grid Cells": int((~weights["Conflict Exposed"]).sum()),
                "All Spatial Blocks": weights["Spatial Block ID"].nunique(),
                "Exposed Spatial Blocks": weights.loc[weights["Conflict Exposed"], "Spatial Block ID"].nunique(),
                "Climate Cells": climate_support["Climate Cell ID"].nunique(),
                "Climate Cells with Within-Cell Dose Variation": int(climate_support["Conflict_Dose_SD"].gt(0.01).sum()),
                "Climate Cells with Exposed and Control Cells": int(
                    (
                        climate_support["Exposed_Grid_Cells"].gt(0)
                        & climate_support["Exposed_Grid_Cells"].lt(climate_support["Grid_Cells"])
                    ).sum()
                ),
            }
        ]
    )
    climate_summary.to_csv(OUT / "spatial_identification_support.csv", index=False)

    # Count shocks at the climate-cell-year level and then ask how many exposed
    # spatial blocks experience at least two positive-intensity years both before
    # and after the conflict. This avoids treating replicated 1 km cells as
    # independent hazard realizations.
    shock_rows: list[dict[str, object]] = []
    block_year = frame.loc[frame["Conflict Exposed"]].groupby(
        ["Spatial Block ID", "Year", "Period"], as_index=False
    )[[*HAZARDS.values()]].mean()
    climate_year = frame.loc[frame["Conflict Exposed"]].drop_duplicates(
        ["Climate Cell ID", "Year"]
    )
    for family, variable in HAZARDS.items():
        pre_values = climate_year.loc[
            climate_year["Period"].eq("Pre-conflict 2000-2007"), variable
        ].dropna()
        pre_lower = float(pre_values.quantile(0.01))
        pre_upper = float(pre_values.quantile(0.99))
        for label in ("Pre-conflict 2000-2007", "Post-conflict 2012-2024"):
            cy = climate_year.loc[climate_year["Period"].eq(label) & climate_year[variable].notna()]
            by = block_year.loc[block_year["Period"].eq(label) & block_year[variable].notna()]
            positive_by_block = (
                by.assign(Positive=by[variable].gt(0))
                .groupby("Spatial Block ID")["Positive"]
                .sum()
            )
            shock_rows.append(
                {
                    "Hazard Family": family,
                    "Variable": variable,
                    "Period": label,
                    "Calendar Years": cy["Year"].nunique(),
                    "Exposed Climate Cells": cy["Climate Cell ID"].nunique(),
                    "Exposed Climate-Cell-Years": len(cy),
                    "Positive Shock Climate-Cell-Years": int(cy[variable].gt(0).sum()),
                    "Positive Shock Share": float(cy[variable].gt(0).mean()),
                    "Intensity Q05": float(cy[variable].quantile(0.05)),
                    "Intensity Median": float(cy[variable].median()),
                    "Intensity Q95": float(cy[variable].quantile(0.95)),
                    "Share within Pre-conflict Q01-Q99 Range": float(
                        cy[variable].between(pre_lower, pre_upper).mean()
                    ),
                    "Exposed Blocks with Two or More Positive Shock Years": int(
                        positive_by_block.ge(2).sum()
                    ),
                    "Exposed Blocks Evaluated": int(positive_by_block.size),
                }
            )
    shock_support = pd.DataFrame(shock_rows)
    shock_support.to_csv(OUT / "hazard_support.csv", index=False)

    flood = pd.read_csv(FLOOD_MANIFEST, parse_dates=["Event Start Date", "Event End Date"])
    flood["Period"] = period(flood["Event Start Date"].dt.year)
    flood_support = flood.groupby("Period", as_index=False).agg(
        Flood_Events=("Event ID", "nunique"),
        First_Event=("Event Start Date", "min"),
        Last_Event=("Event End Date", "max"),
    )
    flood_support.to_csv(OUT / "observed_flood_temporal_support.csv", index=False)

    questions = pd.DataFrame(
        [
            {
                "Question": "Did the 2011 conflict directly reduce local nighttime activity?",
                "Status": "partly-testable and already estimated",
                "Reason": "LongNTL spans 2000-2024 and CCNL independently covers 2010-2013, but treatment is concentrated in two conflict sectors.",
            },
            {
                "Question": "Did conflict exposure increase later drought sensitivity?",
                "Status": "testable subject to prospective power audit",
                "Reason": "Annual drought intensity and both main outcomes span the required pre/post periods with repeated exposed-block shocks.",
            },
            {
                "Question": "Did conflict exposure increase later extreme-rainfall sensitivity?",
                "Status": "partly-testable after common-support restriction",
                "Reason": "Annual extreme-rainfall intensity spans 2000-2024, but only about 69% of post-conflict exposed climate-cell-years lie within the pre-conflict Q01-Q99 intensity range; it must not be labelled observed flooding.",
            },
            {
                "Question": "Did conflict exposure increase later heat or compound-hot-dry sensitivity?",
                "Status": "weakly-testable with current pre-conflict support",
                "Reason": "Only 15 exposed blocks have at least two positive pre-conflict heat years and only 3 have two compound hot-dry years; about 24% of post-conflict heat observations fall within the pre-conflict Q01-Q99 range.",
            },
            {
                "Question": "Did conflict exposure increase the impact of satellite-observed inundation?",
                "Status": "weakly-testable with current data",
                "Reason": "The acquired GFD inventory begins in 2007 and supplies only one pre-conflict flood year, so it cannot by itself identify a pre/post slope change.",
            },
            {
                "Question": "Can the current data support a broad causal claim about conflict and climate resilience?",
                "Status": "not yet",
                "Reason": "The conflict treatment has only two sectors and the validated direct light decline is concentrated in Preah Vihear; grid-cell counts do not replace independent conflict assignments.",
            },
        ]
    )
    questions.to_csv(OUT / "question_feasibility.csv", index=False)

    pre_npp = availability.loc[availability["Family"].eq("Annual land NPP")].iloc[0]
    README = f"""# Gate C feasibility audit: 2011 border conflict and later hazard sensitivity

This is an outcome-blind data-support audit for the newly confirmed research
question. It does not estimate the conflict-by-post-by-hazard coefficient and
does not update AnaSOP.

## Bottom line

- Annual drought is **technically estimable now** on the frozen Gate B support.
  Extreme rainfall is partly estimable after a prospective common-intensity-
  support restriction. Heat and compound hot-dry pre/post slope changes have
  weak pre-conflict shock support and should not be primary with current data.
- LongNTL covers 2000-2024; annual NPP supplies {int(pre_npp['Valid Years'])}
  valid years ending in {int(pre_npp['Last Year'])} on this support.
- The sample contains {len(weights):,} cells, {int(weights['Conflict Exposed'].sum()):,}
  exposed cells, and {weights.loc[weights['Conflict Exposed'], 'Spatial Block ID'].nunique()}
  exposed 10 km spatial blocks.
- Satellite-observed flood amplification is **not yet a credible primary pre/post
  test**: the acquired GFD inventory has only one pre-conflict event year (2007).
- The binding limitation is causal replication, not row count. Treatment comes
  from two conflict sectors, and the validated direct signal is concentrated in
  one. Thousands of cells and repeated hazard years cannot create additional
  independent conflict assignments.

## Recommended disposition

Proceed first to a prospective power and design-rank audit for drought, then for
extreme rainfall under common intensity support. Keep heat, compound hot-dry,
and actual inundation secondary unless longer pre-conflict support is obtained.
Do not promote the design to a broad causal claim unless spatial placebo
inference, sector robustness, and pre-conflict hazard-sensitivity tests pass.
"""
    (OUT / "README.md").write_text(README, encoding="utf-8")
    print(climate_summary.to_string(index=False))
    print("\nHazard support")
    print(shock_support.to_string(index=False))
    print("\nObserved flood support")
    print(flood_support.to_string(index=False))
    print("\nQuestion feasibility")
    print(questions.to_string(index=False))


if __name__ == "__main__":
    main()
