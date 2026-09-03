#!/usr/bin/env python3
"""Outcome-blind 2 km power gate for the temperature bandwidth sensitivity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from feasibility_boundary_temperature_shocks import (
    ANNUAL_COLUMNS,
    ANNUAL_INPUT,
    ANNUAL_SCENARIOS,
    ANNUAL_SHOCKS,
    HF_COLUMNS,
    HF_INPUT,
    HF_SCENARIOS,
    HF_SHOCKS,
    annual_support,
    hf_support,
    power_rows,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data/exp/feasibility-check/temperature-extension"
BANDWIDTH = 2


def main() -> None:
    annual = pd.read_parquet(ROOT / ANNUAL_INPUT, columns=ANNUAL_COLUMNS)
    annual = annual.loc[
        annual["Year"].between(2001, 2021)
        & annual["Absolute Distance to Historical Repression Boundary km"].le(BANDWIDTH)
        & annual["NPP Complete 2001-2020 Baseline"].eq(1)
    ].copy()
    annual["Village Code"] = annual["Village Code"].astype("string").str.zfill(8)
    annual["Commune Code"] = annual["Commune Code"].astype("string").str.zfill(6)
    annual["segment_year"] = annual["Historical Boundary Segment"].astype(str) + "|" + annual["Year"].astype(str)
    annual["commune_year"] = annual["Linked Climate Commune Code"].astype(str) + "|" + annual["Year"].astype(str)
    annual = annual.dropna(
        subset=list(ANNUAL_SHOCKS.values())
        + ["May October Hot Day Intensity", "Temperature Grid May October Dry Rainfall Intensity"]
    ).sort_values(["Village Code", "Year"])
    cross = (
        annual.drop_duplicates("Village Code")
        .groupby("Linked Climate Commune Code", observed=True)["Higher-Repression Southwest Zone"]
        .nunique()
    )
    annual_confirmation = annual.loc[
        annual["Linked Climate Commune Code"].isin(set(cross[cross.eq(2)].index))
    ].copy()

    support = [
        annual_support(annual, "main 2 km"),
        annual_support(annual_confirmation, "within-climate-commune 2 km"),
    ]
    rows = power_rows(
        annual,
        "annual",
        "main 2 km",
        ANNUAL_SHOCKS,
        ["Village Code", "segment_year"],
        ANNUAL_SCENARIOS,
        "Year",
        "May October Hot Day Intensity",
        "Temperature Grid May October Dry Rainfall Intensity",
    )
    rows += power_rows(
        annual_confirmation,
        "annual",
        "within-climate-commune 2 km",
        ANNUAL_SHOCKS,
        ["Village Code", "segment_year", "commune_year"],
        ANNUAL_SCENARIOS,
        "Year",
        "May October Hot Day Intensity",
        "Temperature Grid May October Dry Rainfall Intensity",
    )

    hf = pd.read_parquet(ROOT / HF_INPUT, columns=HF_COLUMNS)
    hf["Composite Date"] = pd.to_datetime(hf["Composite Date"])
    hf["evi_available"] = (
        hf["EVI Valid Pixel Count"].ge(hf["Required Valid Pixel Count"])
        & hf["EVI Slot Valid Years 2001-2020"].ge(10)
        & hf["EVI Slot SD 2001-2020"].gt(0)
    )
    hf = hf.loc[
        hf["Year"].between(2001, 2021)
        & hf["Absolute Distance to Historical Repression Boundary km"].le(BANDWIDTH)
        & hf["Cross-Side CHIRPS Cell"].eq(1)
        & hf["evi_available"]
    ].copy()
    hf["Village Code"] = hf["Village Code"].astype("string").str.zfill(8)
    hf = hf.dropna(subset=list(HF_SHOCKS.values()) + ["Hot Day Intensity", "Dry Rainfall Intensity"])
    hf["cell_date"] = hf["CHIRPS Cell ID"].astype(str) + "|" + hf["Composite Date"].astype(str)
    hf["cell_commune_date"] = (
        hf["CHIRPS Cell ID"].astype(str)
        + "|"
        + hf["Linked Climate Commune Code"].astype(str)
        + "|"
        + hf["Composite Date"].astype(str)
    )
    cross_pairs = (
        hf.drop_duplicates("Village Code")
        .groupby(["CHIRPS Cell ID", "Linked Climate Commune Code"], observed=True)[
            "Higher-Repression Southwest Zone"
        ]
        .nunique()
    )
    pairs = pd.MultiIndex.from_frame(hf[["CHIRPS Cell ID", "Linked Climate Commune Code"]])
    hf_confirmation = hf.loc[pairs.isin(set(cross_pairs[cross_pairs.eq(2)].index))].copy()
    support += [
        hf_support(hf, "main cross-side climate-cell 2 km"),
        hf_support(hf_confirmation, "within-climate-commune 2 km"),
    ]
    rows += power_rows(
        hf.sort_values(["Village Code", "Composite Date"]),
        "16-day EVI",
        "main cross-side climate-cell 2 km",
        HF_SHOCKS,
        ["Village Code", "cell_date"],
        HF_SCENARIOS,
        "Composite Date",
        "Hot Day Intensity",
        "Dry Rainfall Intensity",
    )
    rows += power_rows(
        hf_confirmation.sort_values(["Village Code", "Composite Date"]),
        "16-day EVI",
        "within-climate-commune 2 km",
        HF_SHOCKS,
        ["Village Code", "cell_commune_date"],
        HF_SCENARIOS,
        "Composite Date",
        "Hot Day Intensity",
        "Dry Rainfall Intensity",
    )

    power = pd.DataFrame(rows)
    strong = power.loc[power["dependence_scenario"].eq("strong clustered dependence")].copy()
    strong["bandwidth_km"] = BANDWIDTH
    support_frame = pd.DataFrame(support)
    support_frame["bandwidth_km"] = BANDWIDTH
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    support_frame.to_csv(OUTPUT_DIR / "temperature_2km_support.csv", index=False)
    strong.to_csv(OUTPUT_DIR / "temperature_2km_blinded_power.csv", index=False)
    print(support_frame.to_string(index=False))
    print("\nStrong-dependence MDE")
    print(
        strong[["design", "specification", "shock", "mde_80_standardized_outcome_per_one_sd_shock", "power_gate_pass"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
