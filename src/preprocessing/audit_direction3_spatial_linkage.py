#!/usr/bin/env python3
"""Audit exact village-code linkage between CSES waves and the public mine baseline."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import geopandas as gpd
import pandas as pd

from inventory_direction3_sources import DataSource, discover_sources


CROSSWALK_SPECS = {
    "2007": "CSES 2007/Village data/Village2007/dbo_psulisting.dta",
    "2009": "CSES 2009/Stata CSES09/areainfo.dta",
    # Use the complete 360-row listing. The separately supplied "EDIT Cname"
    # file contains only 224 PSUs and would truncate the survey linkage.
    "2011-12": "CSES 2011-12/11psulisting.dta",
    "2013": "CSES 2013.zip::PSUListing.dta",
    "2014": "CSES 2014/CSES2014/psu listing.dta",
    "2016": "CSES2016/CSES2016/psulisting.dta",
    "2017": "CSES2017/CSES2017/2017hz_psulisting.dta",
    "2019": "CSES2019/CSES2019_Village.dta",
    "2021": "Data of CSES2021/AreaInformation.dta",
}


def read_source(source: DataSource) -> pd.DataFrame:
    input_obj: Path | io.BytesIO
    input_obj = io.BytesIO(source.read_bytes()) if source.archive_members else source.root_file
    return pd.read_stata(input_obj, convert_categoricals=False)


def find_source(root: Path, suffix: str, sources: list[DataSource]) -> DataSource:
    matches = [source for source in sources if source.display_name(root).endswith(suffix)]
    if len(matches) != 1:
        names = [source.display_name(root) for source in matches]
        raise ValueError(f"Expected one source ending in {suffix!r}; found {names}")
    return matches[0]


def get_column(frame: pd.DataFrame, *aliases: str) -> pd.Series:
    lookup = {str(column).lower(): column for column in frame.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return frame[lookup[alias.lower()]]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def code_part(series: pd.Series, width: int = 2) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return numeric.astype("string").str.zfill(width)


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def standardize_crosswalk(frame: pd.DataFrame, wave: str) -> pd.DataFrame:
    province = code_part(get_column(frame, "province_code", "province"))
    district = code_part(get_column(frame, "district_code", "district"))
    commune = code_part(get_column(frame, "commune_code", "commune"))
    village = code_part(get_column(frame, "village_code", "village"))
    output = pd.DataFrame({
        "survey_year": wave,
        "psu": clean_text(get_column(frame, "psu", "psu11")),
        "province_code": province,
        "district_code": district,
        "commune_code": commune,
        "village_code": village,
        "province_name": clean_text(get_column(frame, "province_name", "pname")),
        "district_name": clean_text(get_column(frame, "district_name", "dname")),
        "commune_name": clean_text(get_column(frame, "commune_name", "cname")),
        "village_name": clean_text(get_column(frame, "village_name", "vname")),
        "urban_rural": get_column(frame, "urbanrural", "urban"),
        "survey_month": get_column(frame, "surveymonth", "survey_month"),
    })
    output["commune_code_full"] = province + district + commune
    output["village_code_full"] = province + district + commune + village
    return output.drop_duplicates().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = root / "data" / "exp" / "feasibility-check"
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = discover_sources(root)

    mine_path = root / "data" / "raw" / "mine" / "cambodia_mine_erw_baseline_2009_2014.zip"
    mine = gpd.read_file(f"zip://{mine_path}")
    mine["village_code_full"] = mine["VilCode"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)
    mine["commune_code_full"] = mine["village_code_full"].str[:6]
    mine_villages = set(mine["village_code_full"].dropna())

    crosswalks: list[pd.DataFrame] = []
    audit_rows = [{
        "survey_year": "2004",
        "crosswalk_rows": 0,
        "unique_psus": 0,
        "unique_villages": 0,
        "exact_mine_matches": 0,
        "match_share_pct": 0.0,
        "status": "no administrative village-code crosswalk located",
    }]

    for wave, suffix in CROSSWALK_SPECS.items():
        source = find_source(root, suffix, sources)
        crosswalk = standardize_crosswalk(read_source(source), wave)
        crosswalk["mine_baseline_match"] = crosswalk["village_code_full"].isin(mine_villages)
        crosswalk["source_dataset"] = source.display_name(root)
        crosswalks.append(crosswalk)

        valid = crosswalk["village_code_full"].notna()
        unique_villages = crosswalk.loc[valid, "village_code_full"].nunique()
        matched = crosswalk.loc[valid & crosswalk["mine_baseline_match"], "village_code_full"].nunique()
        audit_rows.append({
            "survey_year": wave,
            "crosswalk_rows": len(crosswalk),
            "unique_psus": crosswalk["psu"].nunique(dropna=True),
            "unique_villages": unique_villages,
            "exact_mine_matches": matched,
            "match_share_pct": round(100 * matched / unique_villages, 2) if unique_villages else 0.0,
            "status": "exact administrative-code linkage available",
        })

    combined = pd.concat(crosswalks, ignore_index=True)
    combined.to_csv(output_dir / "cses_village_crosswalk_candidate.csv", index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "direction3_spatial_linkage_by_wave.csv", index=False)

    mine_summary = pd.DataFrame([
        {"metric": "contamination_points", "value": len(mine)},
        {"metric": "unique_villages", "value": mine["village_code_full"].nunique()},
        {"metric": "unique_communes", "value": mine["commune_code_full"].nunique()},
        {"metric": "unique_provinces", "value": mine["Province"].nunique()},
        {"metric": "survey_date_min", "value": mine["SurveyDate"].min()},
        {"metric": "survey_date_max", "value": mine["SurveyDate"].max()},
        {"metric": "crs", "value": str(mine.crs)},
    ])
    mine_summary.to_csv(output_dir / "mine_baseline_summary.csv", index=False)

    print(audit.to_string(index=False))
    print("\nMine baseline")
    print(mine_summary.to_string(index=False))


if __name__ == "__main__":
    main()
