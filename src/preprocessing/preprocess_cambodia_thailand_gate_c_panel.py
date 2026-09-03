#!/usr/bin/env python3
"""Build the frozen annual Gate C conflict-by-hazard analysis panel."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PANEL = ROOT / "data/processed/cambodia_national_annual_satellite_climate_panel_preprocessed.parquet"
WEIGHTS = ROOT / "data/exp/experiment-design/cambodia-thailand-gate-b/gate_b_binary_support_weights.parquet"
OUTPUT = ROOT / "data/processed/cambodia_thailand_gate_c_annual_panel_preprocessed.parquet"
AUDIT = ROOT / "data/exp/data-preprocessing/cambodia-thailand-gate-c"

NPP_NATURAL = "Annual Land NPP Anomaly kg C per m2"
NPP_STANDARDIZED = "Annual Land NPP Anomaly Z 2001-2020"
DROUGHT = "May October Dry Rainfall Intensity"
DRY_SPELL = "May October Maximum Consecutive Dry Days Anomaly Z"
EXTREME_RAIN = "May October Extreme Wet Rainfall Intensity"
MAX_FIVE_DAY = "May October Maximum Five-Day Precipitation mm Anomaly Z"
HEAT = "May October Heat Intensity"
COMPOUND = "May October Compound Hot-Dry Intensity"

SOURCE_COLUMNS = [
    "National Grid Cell ID",
    "Year",
    "Longitude",
    "Latitude",
    "Province Code",
    "Province Name",
    "District Code",
    "District Name",
    "Commune Code",
    "Commune Name",
    "Climate Cell ID",
    NPP_NATURAL,
    NPP_STANDARDIZED,
    "Grid 2001-2020 SD Annual Land NPP kg C per m2",
    "NPP Complete 2001-2020 Baseline",
    "Mean NPP QC Filled Growing-Season Days Percent",
    DROUGHT,
    DRY_SPELL,
    EXTREME_RAIN,
    MAX_FIVE_DAY,
    HEAT,
    COMPOUND,
]


def analysis_period(year: pd.Series) -> pd.Categorical:
    values = np.select(
        [
            year.between(2001, 2007),
            year.between(2008, 2010),
            year.eq(2011),
            year.between(2012, 2014),
            year.between(2015, 2019),
            year.between(2020, 2024),
        ],
        [
            "Pre-conflict 2001-2007",
            "Escalation 2008-2010",
            "Conflict year 2011",
            "Early post-conflict 2012-2014",
            "Medium post-conflict 2015-2019",
            "Late post-conflict 2020-2024",
        ],
        default="Outside scope",
    )
    return pd.Categorical(
        values,
        categories=[
            "Outside scope",
            "Pre-conflict 2001-2007",
            "Escalation 2008-2010",
            "Conflict year 2011",
            "Early post-conflict 2012-2014",
            "Medium post-conflict 2015-2019",
            "Late post-conflict 2020-2024",
        ],
        ordered=True,
    )


def main() -> None:
    panel = pd.read_parquet(SOURCE_PANEL, columns=SOURCE_COLUMNS)
    weights = pd.read_parquet(WEIGHTS).copy()
    weights["Continuous Conflict Dose"] = np.where(
        weights["Conflict Distance Ring"].eq("Over 60 km"),
        0.0,
        np.maximum(
            0.0,
            1 - weights["Candidate Conflict Year 2011 Nearest Event Distance km"] / 60,
        ),
    )
    weights["Conflict Exposed within 60 km"] = weights["Continuous Conflict Dose"].gt(0)
    frame = panel.merge(weights, on="National Grid Cell ID", validate="many_to_one")
    frame["Gate C Analysis Period"] = analysis_period(frame["Year"])
    frame["Post-conflict 2012-2024"] = frame["Year"].between(2012, 2024)
    frame["Gate C Main Pre-Post Period"] = frame["Year"].between(2001, 2007) | frame[
        "Year"
    ].between(2012, 2024)
    frame["Conflict Dose x Post-conflict"] = (
        frame["Continuous Conflict Dose"] * frame["Post-conflict 2012-2024"].astype(float)
    )
    frame["Conflict Dose x Drought"] = frame["Continuous Conflict Dose"] * frame[DROUGHT]
    frame["Conflict Dose x Post-conflict x Drought"] = (
        frame["Continuous Conflict Dose"]
        * frame["Post-conflict 2012-2024"].astype(float)
        * frame[DROUGHT]
    )
    frame["Conflict Dose x Extreme Rainfall"] = (
        frame["Continuous Conflict Dose"] * frame[EXTREME_RAIN]
    )
    frame["Conflict Dose x Post-conflict x Extreme Rainfall"] = (
        frame["Continuous Conflict Dose"]
        * frame["Post-conflict 2012-2024"].astype(float)
        * frame[EXTREME_RAIN]
    )

    # Freeze common-intensity support from unique pre-conflict climate-cell-years
    # among exposed cells only. No outcome enters these bounds.
    exposed_pre = frame.loc[
        frame["Conflict Exposed within 60 km"] & frame["Year"].between(2001, 2007),
        ["Climate Cell ID", "Year", DROUGHT, EXTREME_RAIN],
    ].drop_duplicates(["Climate Cell ID", "Year"])
    support_bounds: dict[str, dict[str, float]] = {}
    for family, variable in (("Drought", DROUGHT), ("Extreme Rainfall", EXTREME_RAIN)):
        lower = float(exposed_pre[variable].quantile(0.01))
        upper = float(exposed_pre[variable].quantile(0.99))
        support_bounds[family] = {"q01": lower, "q99": upper}
        frame[f"{family} within Pre-conflict Exposed Q01-Q99 Support"] = frame[variable].between(
            lower, upper
        )

    if frame.duplicated(["National Grid Cell ID", "Year"]).any():
        raise RuntimeError("Gate C panel is not unique by grid cell and year")
    if not frame["Continuous Conflict Dose"].between(0, 1).all():
        raise RuntimeError("Continuous conflict dose falls outside [0, 1]")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False)
    AUDIT.mkdir(parents=True, exist_ok=True)

    variables = []
    roles = {
        NPP_NATURAL: "primary ecological-production outcome in natural units",
        NPP_STANDARDIZED: "standardized ecological-production robustness outcome",
        DROUGHT: "primary hazard",
        DRY_SPELL: "alternative drought definition",
        EXTREME_RAIN: "secondary hazard",
        MAX_FIVE_DAY: "alternative extreme-rainfall definition",
        HEAT: "secondary unsupported hazard retained for reference",
        COMPOUND: "secondary unsupported hazard retained for reference",
        "Continuous Conflict Dose": "primary conflict exposure",
        "Post-conflict 2012-2024": "post-conflict period indicator",
        "Gate C Main Pre-Post Period": "primary temporal sample flag",
        "Gate B Binary Support Overlap Weight": "frozen outcome-independent analysis weight",
    }
    for column in frame.columns:
        variables.append(
            {
                "Readable Variable Name": column,
                "Role": roles.get(column, "identifier, fixed-effect field, support field, or derived interaction"),
                "Dtype": str(frame[column].dtype),
                "Missing Share": float(frame[column].isna().mean()),
                "Is Final Gate C Variable": "yes" if column in roles or "Conflict Dose x" in column else "no",
            }
        )
    pd.DataFrame(variables).to_csv(AUDIT / "variable_list.csv", index=False)

    decisions = {
        "status": "analysis-ready candidate Gate C panel; effect estimation unopened at preprocessing",
        "source_rows": len(panel),
        "output_rows": len(frame),
        "grid_cells": frame["National Grid Cell ID"].nunique(),
        "years": [int(frame["Year"].min()), int(frame["Year"].max())],
        "primary_hazard": DROUGHT,
        "secondary_hazard": EXTREME_RAIN,
        "outcomes": [NPP_NATURAL, NPP_STANDARDIZED],
        "main_time_rule": "2001-2007 pre-conflict versus 2012-2024 post-conflict; exclude 2008-2011 from the main slope-change estimand",
        "conflict_exposure": "continuous linear distance dose equal to one at a verified 2011 event and zero at 60 km or farther",
        "weights": "frozen Gate B outcome-independent overlap weights",
        "common_intensity_support": support_bounds,
        "missing_rule": "retain source missingness; do not impute outcomes or hazards",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (AUDIT / "decisions.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (AUDIT / "README.md").write_text(
        """# Cambodia–Thailand Gate C annual analysis panel

The panel merges the frozen outcome-independent Gate B support with annual
NPP outcomes and climate hazards. The main comparison uses 2001–2007 as
pre-conflict and 2012–2024 as post-conflict; 2008–2011 is retained but excluded
from the primary slope-change estimand. Drought is primary. Extreme rainfall is
secondary and has a prospectively frozen pre-conflict exposed Q01–Q99 support
flag. LongNTL is excluded by human decision. No Gate C target coefficient was
inspected while creating this panel.
""",
        encoding="utf-8",
    )
    print(f"Saved {OUTPUT.relative_to(ROOT)}")
    print(f"Rows: {len(frame):,}; grid cells: {frame['National Grid Cell ID'].nunique():,}")
    print(json.dumps(support_bounds, indent=2))


if __name__ == "__main__":
    main()
