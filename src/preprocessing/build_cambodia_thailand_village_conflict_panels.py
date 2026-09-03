#!/usr/bin/env python3
"""Build village-level candidate conflict exposure and analysis panels.

The treatment variables in this script are deliberately labelled ``Candidate``.
UCDP event coordinates and public humanitarian reports identify conflict sectors
and affected administrative areas, but they do not enumerate every evacuated
village of origin.  Consequently, these outputs are suitable for support audits,
design diagnostics, and provisional estimation only; they are not the frozen
causal treatment required by AnaSOP Section 5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[2]
EVENTS = ROOT / "data/processed/ucdp_cambodia_thailand_state_conflict_candidates_preprocessed.parquet"
POINTS = ROOT / "data/processed/cses_village_points_preprocessed.parquet"
ANNUAL = ROOT / "data/processed/cses_village_buffer_annual_satellite_preprocessed.parquet"
STATIC = ROOT / "data/processed/cses_village_buffer_static_geography_preprocessed.parquet"
CSES = ROOT / "data/processed/cses_village_year_satellite_preprocessed.parquet"
CSES_PUBLIC_CROSSWALK = ROOT / "data/processed/cses_to_cambodia_public_village_point_crosswalk_preprocessed.parquet"

EXPOSURE_OUT = ROOT / "data/processed/cambodia_thailand_village_conflict_exposure_candidate_preprocessed.parquet"
NPP_OUT = ROOT / "data/processed/cambodia_thailand_village_npp_conflict_panel_candidate_preprocessed.parquet"
CSES_OUT = ROOT / "data/processed/cambodia_thailand_cses_conflict_panel_candidate_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/cambodia-thailand-village-conflict"

DISTANCE_RADII_KM = (10, 20, 40, 60)
NPP_START_YEAR = 2001
NPP_END_YEAR = 2024

SECTOR_RULES = {
    "Preah Vihear": {
        "province": "Preah Vihear",
        "district": "Choam Khsant",
        "first_date": pd.Timestamp("2008-10-15"),
        "event_rule": "longitude >= 104.0",
    },
    "Ta Moan-Ta Krabey": {
        "province": "Otdar Meanchey",
        "district": "Banteay Ampil",
        "first_date": pd.Timestamp("2011-04-22"),
        "event_rule": "longitude < 104.0",
    },
}


def assign_sector(events: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(events["longitude"].ge(104.0), "Preah Vihear", "Ta Moan-Ta Krabey"),
        index=events.index,
        dtype="string",
    )


def build_exposure(points: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    matched = points.loc[points["Public Village Point Matched"].eq(1)].copy()
    if matched[["Point Longitude", "Point Latitude"]].isna().any().any():
        raise RuntimeError("Mapped village points contain missing coordinates")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    village_e, village_n = transformer.transform(
        matched["Point Longitude"].to_numpy(float), matched["Point Latitude"].to_numpy(float)
    )
    event_e, event_n = transformer.transform(
        events["longitude"].to_numpy(float), events["latitude"].to_numpy(float)
    )
    distance = np.sqrt(
        (village_e[:, None] - event_e[None, :]) ** 2
        + (village_n[:, None] - event_n[None, :]) ** 2
    ) / 1000.0

    output = matched.drop(columns=["geometry"], errors="ignore").copy()
    if CSES_PUBLIC_CROSSWALK.exists():
        nested = pd.read_parquet(
            CSES_PUBLIC_CROSSWALK,
            columns=["Village Code", "National Village Point ID", "National Public Village Point Matched"],
        )
        output = output.merge(nested, on="Village Code", how="left", validate="one_to_one")
    nearest = distance.argmin(axis=1)
    output["Candidate Nearest UCDP Event Distance km"] = distance.min(axis=1).astype("float32")
    output["Candidate Nearest UCDP Event ID"] = events.iloc[nearest]["id"].to_numpy(dtype="int64")
    output["Candidate Nearest UCDP Event Date"] = events.iloc[nearest]["date_start"].to_numpy()
    output["Candidate Nearest UCDP Event Place"] = events.iloc[nearest]["where_coordinates"].astype("string").to_numpy()
    output["Candidate Nearest UCDP Event Spatial Precision"] = events.iloc[nearest]["where_prec"].to_numpy(dtype="int16")

    for sector in SECTOR_RULES:
        mask = events["Candidate Conflict Sector"].eq(sector).to_numpy()
        sector_distance = distance[:, mask]
        output[f"Candidate {sector} Nearest Event Distance km"] = sector_distance.min(axis=1).astype("float32")
        for radius in DISTANCE_RADII_KM:
            inside = sector_distance <= radius
            output[f"Candidate {sector} Event Count Within {radius} km"] = inside.sum(axis=1).astype("int16")
            output[f"Candidate {sector} Exposure Within {radius} km"] = inside.any(axis=1).astype("int8")

    output["Candidate Affected Province"] = 0
    output["Candidate Affected District"] = 0
    output["Candidate Conflict Sector"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output["Candidate First Conflict Date"] = pd.NaT
    output["Candidate Treatment Evidence Grade"] = "Outside documented affected administrative areas"
    for sector, rule in SECTOR_RULES.items():
        province = output["Province Name"].eq(rule["province"])
        district = province & output["District Name"].eq(rule["district"])
        output.loc[province, "Candidate Affected Province"] = 1
        output.loc[district, "Candidate Affected District"] = 1
        output.loc[district, "Candidate Conflict Sector"] = sector
        output.loc[district, "Candidate First Conflict Date"] = rule["first_date"]
        output.loc[province, "Candidate Treatment Evidence Grade"] = (
            "D: affected province documented; too coarse for primary treatment"
        )
        output.loc[district, "Candidate Treatment Evidence Grade"] = (
            "C: event-containing affected district; village origins not enumerated"
        )

    output["Candidate First Conflict Year"] = output["Candidate First Conflict Date"].dt.year.astype("Int16")
    output["Candidate District Treatment Is Frozen"] = np.int8(0)
    output["Candidate Treatment Suitable for Causal Claim"] = np.int8(0)
    if output["Village Code"].duplicated().any():
        raise RuntimeError("Candidate exposure is not unique by Village Code")
    return output.sort_values("Village Code").reset_index(drop=True)


def add_timing(frame: pd.DataFrame, year_column: str) -> pd.DataFrame:
    output = frame.copy()
    year = pd.to_numeric(output[year_column], errors="coerce")
    first_year = pd.to_numeric(output["Candidate First Conflict Year"], errors="coerce")
    treated = output["Candidate Affected District"].eq(1)
    output["Candidate Post Conflict Period"] = pd.Series(0, index=output.index, dtype="Int8")
    output.loc[treated & year.gt(first_year), "Candidate Post Conflict Period"] = 1
    output.loc[treated & year.eq(first_year), "Candidate Post Conflict Period"] = pd.NA
    output.loc[~treated, "Candidate Post Conflict Period"] = 0
    output["Candidate Conflict Transition Year"] = (
        treated & year.eq(first_year)
    ).astype("int8")
    output["Candidate Event Time Year"] = (year - first_year).where(treated).astype("Int16")
    return output


def build_npp_panel(exposure: pd.DataFrame) -> pd.DataFrame:
    annual = pd.read_parquet(ANNUAL)
    annual = annual.loc[annual["Year"].between(NPP_START_YEAR, NPP_END_YEAR)].copy()
    static = pd.read_parquet(STATIC)
    panel = annual.merge(static, on=["Village Code", "Buffer Radius km"], how="left", validate="many_to_one")
    panel = panel.merge(exposure, on="Village Code", how="left", validate="many_to_one")
    panel = add_timing(panel, "Year")
    key = ["Village Code", "Buffer Radius km", "Year"]
    if panel.duplicated(key).any():
        raise RuntimeError("NPP panel contains duplicate village-buffer-year keys")
    return panel.sort_values(key).reset_index(drop=True)


def build_cses_panel(exposure: pd.DataFrame) -> pd.DataFrame:
    cses = pd.read_parquet(CSES)
    panel = cses.merge(exposure, on="Village Code", how="left", validate="many_to_one", suffixes=("", " Conflict"))
    panel = add_timing(panel, "Survey Year")
    panel["Candidate Exact Survey Timing Resolved"] = (
        panel["Single Survey Month Within Village"].eq(1)
        & panel["Survey Month Minimum"].notna()
    ).astype("int8")
    key = ["Village Code", "Buffer Radius km", "Survey Year"]
    if panel.duplicated(key).any():
        raise RuntimeError("CSES panel contains duplicate village-buffer-survey-year keys")
    return panel.sort_values(key).reset_index(drop=True)


def write_audits(exposure: pd.DataFrame, npp: pd.DataFrame, cses: pd.DataFrame, events: pd.DataFrame) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    source_ledger = pd.DataFrame([
        {
            "Source ID": "UCDP-GED-26.1",
            "Source Type": "georeferenced conflict events",
            "Source URL": "https://ucdp.uu.se/downloads/",
            "What It Supports": "event coordinates, dates, fatalities, and spatial precision",
            "What It Does Not Support": "complete enumeration of evacuated village origins",
        },
        {
            "Source ID": "IFRC-2011-02-14",
            "Source Type": "humanitarian information bulletin",
            "Source URL": "https://www.ifrc.org/docs/appeals/rpts11/KHTHidp14021101.pdf",
            "What It Supports": "February displacement from communities near the border in Preah Vihear",
            "What It Does Not Support": "origin-village list; named places are mainly evacuation destinations",
        },
        {
            "Source ID": "IFRC-2011-04-27",
            "Source Type": "humanitarian information bulletin",
            "Source URL": "https://www.ifrc.org/docs/appeals/rpts11/KHTHidp27041102.pdf",
            "What It Supports": "April conflict near Ta Moan/Ta Krabey and displacement in Oddar Meanchey",
            "What It Does Not Support": "complete origin-village list; IDP sub-areas include destinations",
        },
        {
            "Source ID": "UN-S-2011-264",
            "Source Type": "official letter to the United Nations",
            "Source URL": "https://digitallibrary.un.org/record/702132/files/S_2011_264-EN.pdf",
            "What It Supports": "22 April attack area reported at Kork Morn village in Oddar Meanchey",
            "What It Does Not Support": "unambiguous crosswalk from historical spelling to current village code",
        },
    ])
    source_ledger.to_csv(AUDIT_DIR / "conflict_treatment_source_ledger.csv", index=False)

    event_inventory = events[[
        "id", "date_start", "date_end", "Candidate Conflict Sector", "where_coordinates",
        "where_description", "adm_1", "adm_2", "latitude", "longitude", "where_prec",
        "date_prec", "best", "low", "high", "source_article",
    ]].sort_values(["date_start", "id"])
    event_inventory.to_csv(AUDIT_DIR / "ucdp_candidate_event_inventory.csv", index=False)

    exposure_summary = (
        exposure.groupby(
            ["Candidate Conflict Sector", "Candidate Treatment Evidence Grade"], dropna=False
        )
        .agg(
            Mapped_Villages=("Village Code", "nunique"),
            Provinces=("Province Name", "nunique"),
            Districts=("District Name", "nunique"),
        )
        .reset_index()
    )
    exposure_summary.to_csv(AUDIT_DIR / "candidate_treatment_village_counts.csv", index=False)

    cses_5 = cses.loc[cses["Buffer Radius km"].eq(5)].copy()
    cses_support = (
        cses_5.groupby(
            ["Candidate Conflict Sector", "Candidate Affected District", "Survey Year"], dropna=False
        )
        .agg(
            Villages=("Village Code", "nunique"),
            Village_Years=("Village Code", "size"),
            Households=("Sample Households Database", "sum"),
            People=("Sample People Database", "sum"),
            Mapped_Villages=("Public Village Point Matched", "sum"),
        )
        .reset_index()
    )
    cses_support.to_csv(AUDIT_DIR / "cses_support_by_candidate_sector_and_wave.csv", index=False)

    npp_support = (
        npp.loc[npp["Buffer Radius km"].eq(5)]
        .groupby(["Candidate Conflict Sector", "Candidate Affected District"], dropna=False)
        .agg(
            Villages=("Village Code", "nunique"),
            First_Year=("Year", "min"),
            Last_Year=("Year", "max"),
            Village_Years=("Year", "size"),
            NPP_Nonmissing=("Buffer Mean Annual Land NPP Anomaly Z 2001-2020", "count"),
        )
        .reset_index()
    )
    npp_support.to_csv(AUDIT_DIR / "npp_support_by_candidate_sector.csv", index=False)

    metadata = {
        "status": "candidate treatment and analysis panels; treatment not frozen",
        "candidate_event_rows": int(len(events)),
        "mapped_villages": int(len(exposure)),
        "candidate_affected_district_villages": int(exposure["Candidate Affected District"].sum()),
        "npp_panel_rows": int(len(npp)),
        "npp_panel_years": [int(npp["Year"].min()), int(npp["Year"].max())],
        "cses_panel_rows": int(len(cses)),
        "primary_buffer_km": 5,
        "sensitivity_buffers_km": [2, 10],
        "key_warning": "CSES support in candidate affected districts must include pre-conflict waves before causal before-after models are estimated.",
        "outputs": {
            "exposure": str(EXPOSURE_OUT.relative_to(ROOT)),
            "npp_panel": str(NPP_OUT.relative_to(ROOT)),
            "cses_panel": str(CSES_OUT.relative_to(ROOT)),
        },
    }
    (AUDIT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    decisions = {
        "treatment_status": "candidate only; not frozen for causal analysis",
        "sector_rules": {
            sector: {
                "province": rule["province"],
                "district": rule["district"],
                "first_conflict_date": str(rule["first_date"].date()),
                "event_coordinate_rule": rule["event_rule"],
            }
            for sector, rule in SECTOR_RULES.items()
        },
        "distance_radii_km": list(DISTANCE_RADII_KM),
        "npp_years": [NPP_START_YEAR, NPP_END_YEAR],
        "transition_year_rule": "missing candidate post indicator in the sector first-conflict calendar year",
        "causal_use_rule": "requires an origin-village or prospectively approved administrative treatment before outcome interpretation",
    }
    (AUDIT_DIR / "decisions.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    readme = f"""# Cambodia-Thailand village conflict preprocessing

- UCDP candidate events: {len(events):,}
- Mapped CSES villages: {len(exposure):,}
- Candidate affected-district villages with mapped points: {int(exposure['Candidate Affected District'].sum()):,}
- Annual NPP panel rows, 2001-2024: {len(npp):,}
- CSES candidate panel rows: {len(cses):,}

The affected-district variables are deliberately provisional. Public humanitarian
bulletins document the affected sectors and administrative areas but do not provide
a complete list of evacuated origin villages. Distance-to-event fields are retained
for design diagnostics and cannot silently replace the treatment definition.
"""
    (AUDIT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("\nCandidate treatment summary")
    print(exposure_summary.to_string(index=False))
    print("\nAffected-district CSES support")
    print(cses_support.loc[cses_support["Candidate Affected District"].eq(1)].to_string(index=False))


def main() -> None:
    for path in [EVENTS, POINTS, ANNUAL, STATIC, CSES]:
        if not path.exists():
            raise FileNotFoundError(path)
    events = pd.read_parquet(EVENTS).sort_values(["date_start", "id"]).reset_index(drop=True)
    events = events.loc[events["year"].between(2008, 2011)].copy()
    events["Candidate Conflict Sector"] = assign_sector(events)
    points = pd.read_parquet(POINTS)
    exposure = build_exposure(points, events)
    npp = build_npp_panel(exposure)
    cses = build_cses_panel(exposure)
    EXPOSURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    exposure.to_parquet(EXPOSURE_OUT, index=False)
    npp.to_parquet(NPP_OUT, index=False)
    cses.to_parquet(CSES_OUT, index=False)
    write_audits(exposure, npp, cses, events)


if __name__ == "__main__":
    main()
