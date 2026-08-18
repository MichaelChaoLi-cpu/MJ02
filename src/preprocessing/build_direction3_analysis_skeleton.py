#!/usr/bin/env python3
"""Build Direction 3 PSU-year and household-year analysis spines.

These are linkage and coverage tables, not final estimation datasets. Detailed
outcomes remain in their row-grain CSES modules until concept-specific
aggregation and monetary harmonization have been approved.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


WAVE_YEAR = {
    "2004": 2004,
    "2007": 2007,
    "2009": 2009,
    "2011-12": 2011,
    "2013": 2013,
    "2014": 2014,
    "2016": 2016,
    "2017": 2017,
    "2019": 2019,
    "2021": 2021,
}


def title_from_module(module: str) -> str:
    return re.sub(r"\s+", " ", module.replace("_", " ")).title()


def module_from_path(path: Path) -> str:
    return path.stem.removeprefix("cses_").removesuffix("_preprocessed")


def build_psu_spine(root: Path, module_paths: list[Path]) -> pd.DataFrame:
    processed = root / "data" / "processed"
    crosswalk = pd.read_parquet(processed / "cses_village_crosswalk_preprocessed.parquet")
    crosswalk["Survey Year"] = crosswalk["Survey Wave"].map(WAVE_YEAR).astype("Int64")
    crosswalk["Province Code"] = crosswalk["Commune Code"].astype("string").str[:2]
    crosswalk["District Code"] = crosswalk["Commune Code"].astype("string").str[:4]
    crosswalk["Main Linked Sample"] = True
    crosswalk["Geography Link Matched"] = crosswalk["Village Code"].notna()
    crosswalk["Mine Baseline Recorded"] = crosswalk["Mine Baseline Recorded"].fillna(False).astype(bool)

    mine = pd.read_parquet(processed / "mine_exposure_village_preprocessed.parquet")
    mine_link_columns = [
        column
        for column in mine.columns
        if column
        not in {
            "Commune Code",
            "Province Name",
            "District Name",
            "Commune Name",
            "Village Name",
        }
    ]
    crosswalk = crosswalk.merge(
        mine[mine_link_columns],
        on="Village Code",
        how="left",
        validate="many_to_one",
    )
    count_columns = [
        column
        for column in mine_link_columns
        if column.endswith("Record Count") or column in {"Mine Operator Count", "Corrected Mine Survey Date Count"}
    ]
    for column in count_columns:
        crosswalk[column] = crosswalk[column].fillna(0)
    crosswalk["Mine Exposure Link Matched"] = crosswalk["Mine Baseline Record Count"].gt(0)
    if not crosswalk["Mine Exposure Link Matched"].equals(crosswalk["Mine Baseline Recorded"]):
        raise ValueError("Mine exposure merge disagrees with the audited baseline match flag")
    crosswalk["Mine Baseline Comparison Status"] = crosswalk["Mine Baseline Recorded"].map(
        {
            True: "recorded public baseline point",
            False: "no recorded point in public baseline",
        }
    )

    climate = pd.read_parquet(processed / "climate_commune_year_preprocessed.parquet")
    crosswalk = crosswalk.merge(
        climate,
        left_on=["Commune Code", "Survey Year"],
        right_on=["Commune Code", "Year"],
        how="left",
        validate="many_to_one",
    )
    crosswalk["Climate Link Matched"] = crosswalk["Annual Rainfall mm"].notna()
    crosswalk = crosswalk.drop(columns=["Year"])

    for path in module_paths:
        module = module_from_path(path)
        title = title_from_module(module)
        keys = pd.read_parquet(path, columns=["Survey Wave", "PSU", "Household ID"])
        keys = keys[keys["Survey Wave"].ne("2004") & keys["PSU"].notna()]
        coverage = keys.groupby(["Survey Wave", "PSU"], dropna=False).agg(
            **{
                f"{title} Row Count": ("PSU", "size"),
                f"{title} Household Count": ("Household ID", "nunique"),
            }
        ).reset_index()
        crosswalk = crosswalk.merge(
            coverage,
            on=["Survey Wave", "PSU"],
            how="left",
            validate="one_to_one",
        )
        for column in [f"{title} Row Count", f"{title} Household Count"]:
            crosswalk[column] = crosswalk[column].fillna(0).astype("int64")

    if crosswalk.duplicated(["Survey Wave", "PSU"]).any():
        raise ValueError("PSU-year spine has duplicate keys")
    return crosswalk.sort_values(["Survey Year", "PSU"]).reset_index(drop=True)


def build_household_spine(psu_spine: pd.DataFrame, module_paths: list[Path]) -> pd.DataFrame:
    members_path = next(path for path in module_paths if module_from_path(path) == "household_members")
    members = pd.read_parquet(
        members_path,
        columns=["Survey Year", "Survey Wave", "Main Linked Sample", "Household ID", "PSU"],
    )
    households = members.dropna(subset=["Household ID"]).drop_duplicates(
        ["Survey Wave", "Household ID"], keep="first"
    )

    psu_columns_to_drop = [
        column
        for column in psu_spine.columns
        if column.endswith(" Row Count") or column.endswith(" Household Count")
    ]
    exposure = psu_spine.drop(columns=psu_columns_to_drop + ["Main Linked Sample"])
    households = households.merge(
        exposure,
        on=["Survey Year", "Survey Wave", "PSU"],
        how="left",
        validate="many_to_one",
    )
    households["Geography Link Matched"] = households["Village Code"].notna()
    households["Climate Link Matched"] = households["Climate Link Matched"].fillna(False)
    households["Mine Baseline Recorded"] = households["Mine Baseline Recorded"].astype("boolean")

    for path in module_paths:
        module = module_from_path(path)
        title = title_from_module(module)
        keys = pd.read_parquet(path, columns=["Survey Wave", "Household ID"])
        keys = keys.dropna(subset=["Household ID"])
        counts = keys.groupby(["Survey Wave", "Household ID"], dropna=False).size().rename(
            f"{title} Row Count"
        ).reset_index()
        households = households.merge(
            counts,
            on=["Survey Wave", "Household ID"],
            how="left",
            validate="one_to_one",
        )
        count_column = f"{title} Row Count"
        households[count_column] = households[count_column].fillna(0).astype("int64")
        households[f"Has {title} Record"] = households[count_column].gt(0)

    if households.duplicated(["Survey Wave", "Household ID"]).any():
        raise ValueError("Household-year spine has duplicate keys")
    return households.sort_values(["Survey Year", "PSU", "Household ID"]).reset_index(drop=True)


def validation_rows(name: str, frame: pd.DataFrame, keys: list[str]) -> list[dict[str, object]]:
    return [
        {"dataset": name, "metric": "rows", "value": len(frame)},
        {"dataset": name, "metric": "columns", "value": len(frame.columns)},
        {"dataset": name, "metric": "duplicate_key_rows", "value": int(frame.duplicated(keys).sum())},
        {
            "dataset": name,
            "metric": "main_sample_rows",
            "value": int(frame["Main Linked Sample"].fillna(False).sum()),
        },
        {
            "dataset": name,
            "metric": "geography_linked_rows",
            "value": int(frame["Geography Link Matched"].fillna(False).sum()),
        },
        {
            "dataset": name,
            "metric": "climate_linked_rows",
            "value": int(frame["Climate Link Matched"].fillna(False).sum()),
        },
        {
            "dataset": name,
            "metric": "mine_baseline_recorded_rows",
            "value": int(frame["Mine Baseline Recorded"].fillna(False).sum()),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    processed = root / "data" / "processed"
    module_paths = sorted(
        path
        for path in processed.glob("cses_*_preprocessed.parquet")
        if path.name != "cses_village_crosswalk_preprocessed.parquet"
    )
    if not module_paths:
        raise FileNotFoundError("No processed CSES modules found")

    psu = build_psu_spine(root, module_paths)
    psu_path = processed / "direction3_psu_year_analysis_skeleton_preprocessed.parquet"
    psu.to_parquet(psu_path, index=False)

    households = build_household_spine(psu, module_paths)
    household_path = processed / "direction3_household_year_spine_preprocessed.parquet"
    households.to_parquet(household_path, index=False)

    rows = validation_rows("direction3_psu_year_analysis_skeleton", psu, ["Survey Wave", "PSU"])
    rows.extend(
        validation_rows(
            "direction3_household_year_spine",
            households,
            ["Survey Wave", "Household ID"],
        )
    )
    validation = pd.DataFrame(rows)
    validation_path = root / "data" / "exp" / "data-preprocessing" / "direction3_processed_validation.csv"
    validation.to_csv(validation_path, index=False)

    release_rows: list[dict[str, object]] = []
    for path in sorted(processed.glob("*_preprocessed.parquet")):
        parquet = pq.ParquetFile(path)
        release_rows.append(
            {
                "dataset": path.stem.removesuffix("_preprocessed"),
                "output_path": str(path.relative_to(root)),
                "rows": parquet.metadata.num_rows,
                "columns": len(parquet.schema_arrow.names),
                "size_bytes": path.stat().st_size,
            }
        )
    release_manifest = pd.DataFrame(release_rows)
    release_manifest.to_csv(
        root / "data" / "exp" / "data-preprocessing" / "direction3_processed_release_manifest.csv",
        index=False,
    )

    print(f"psu_spine: rows={len(psu):,}, columns={len(psu.columns)}, output={psu_path.relative_to(root)}")
    print(
        f"household_spine: rows={len(households):,}, columns={len(households.columns)}, "
        f"output={household_path.relative_to(root)}"
    )
    print(validation.to_string(index=False))
    print(f"release_files={len(release_manifest)}")


if __name__ == "__main__":
    main()
