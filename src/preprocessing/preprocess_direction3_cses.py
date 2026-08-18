#!/usr/bin/env python3
"""Harmonize CSES source modules needed for Direction 3.

The script preserves the original row grain of each survey module, standardizes
linkage identifiers, attaches the audited PSU-village geography, and creates
ASCII English-readable column names. It deliberately does not impute, winsorize,
deflate, aggregate, or otherwise alter substantive survey values.
"""

from __future__ import annotations

import argparse
import io
import re
import unicodedata
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from inventory_direction3_sources import DataSource, discover_sources, normalize_wave, read_metadata


WAVE_ORDER = {
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

FIXED_COLUMNS = [
    "Survey Year",
    "Survey Wave",
    "Main Linked Sample",
    "Household ID",
    "Person ID",
    "PSU",
    "Province Code",
    "District Code",
    "Commune Code",
    "Village Code",
    "Province Name",
    "District Name",
    "Commune Name",
    "Village Name",
    "Urban Rural",
    "Survey Month",
    "Geography Link Matched",
    "Source Dataset",
    "Source Submodule",
    "Module Row ID",
]

RAW_IDENTIFIER_NAMES = {
    "psu": "PSU",
    "psu11": "PSU",
    "vid": "PSU",
    "vid11": "PSU",
    "hhid": "Household ID",
    "persid": "Person ID",
}

SPECIAL_NAMES = {
    "pkid": "Source Primary Key",
    "plid": "Parcel Record ID",
    "pid": "Person Line ID",
    "province": "Source Province Code",
    "provincecode": "Source Province Code",
    "district": "Source District Code",
    "districtcode": "Source District Code",
    "commune": "Source Commune Code",
    "communecode": "Source Commune Code",
    "village": "Source Village Code",
    "villagecode": "Source Village Code",
    "urban": "Source Urban Rural",
    "urbanrural": "Source Urban Rural",
    "strata": "Source Strata",
    "stratum": "Source Strata",
    "zone": "Source Zone",
    "operatorcode": "Source Operator Code",
    "entryuser": "Source Entry User",
    "changedate": "Source Change Date",
}

MODULE_TITLES = {
    "household_core": "Household Core",
    "household_members": "Household Members",
    "education": "Education",
    "agriculture_land": "Agriculture Land",
    "agriculture_crop_production": "Agriculture Crop Production",
    "agriculture_crop_costs": "Agriculture Crop Costs",
    "agriculture_crop_sales": "Agriculture Crop Sales",
    "agriculture_crop_inventory": "Agriculture Crop Inventory",
    "food_consumption": "Food Consumption",
    "food_security": "Food Security",
    "vulnerability": "Vulnerability",
    "nonfood_consumption": "Nonfood Consumption",
    "durable_goods": "Durable Goods",
    "liabilities": "Liabilities",
    "nonagriculture_1": "Nonagriculture Activity List",
    "nonagriculture_2": "Nonagriculture Costs",
    "nonagriculture_3": "Nonagriculture Income",
    "migration": "Migration",
    "livestock_ownership": "Livestock Ownership",
    "livestock_expenses": "Livestock Expenses",
    "employment_current": "Current Employment",
    "other_income": "Other Income",
    "housing": "Housing",
    "village_infrastructure": "Village Infrastructure",
    "village_climate_history": "Village Climate History",
    "household_weights": "Household Weights",
    "person_weights": "Person Weights",
}


def source_leaf(source: DataSource) -> str:
    target = source.archive_members[-1] if source.archive_members else source.root_file.name
    return PurePosixPath(target).name


def token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def is_village_source(source_name: str, leaf_key: str) -> bool:
    low = source_name.lower().replace("\\", "/")
    return any(
        marker in low
        for marker in (
            "village data",
            "data_village",
            "/v cses",
            "/vcses",
            "village_data",
            "village2007",
        )
    ) or bool(re.search(r"(?:^|/)20\d\dvl_", low)) or leaf_key.startswith("2011vl")


def modules_for_source(source: DataSource, root: Path) -> list[str]:
    name = source.display_name(root)
    low = name.lower().replace("\\", "/")
    leaf = source_leaf(source)
    key = token(leaf.removesuffix(".dta"))
    if "/code/" in low or "allvar" in key:
        return []
    village = is_village_source(name, key)
    modules: list[str] = []

    if key == "households" or key.endswith("dbohouseholds") or key.endswith("headinghouseholds"):
        modules.append("household_core")
    if "hhmembers" in key or key.endswith("s01ahhmember") or key == "members":
        modules.append("household_members")
    if not village and (
        "personeducation" in key
        or key.endswith("hhs02education")
        or key in {"02education", "education"}
    ):
        modules.append("education")
    if "landownership" in key or key.endswith("landown"):
        modules.append("agriculture_land")
    if any(marker in key for marker in ("productioncrops", "productcrop", "cropsproduction")):
        modules.append("agriculture_crop_production")
    if any(marker in key for marker in ("costcultivation", "costcrops")):
        modules.append("agriculture_crop_costs")
    if "salescrops" in key or "cropsales" in key:
        modules.append("agriculture_crop_sales")
    if "inventorycrops" in key or "cropinventory" in key:
        modules.append("agriculture_crop_inventory")
    if "foodconsumption" in key:
        modules.append("food_consumption")
    if "otherfoodsecurity" in key:
        modules.append("food_security")
    if "vulnerability" in key:
        modules.append("vulnerability")
    if "recallnonfood" in key or "nonfoodexpenses" in key:
        modules.append("nonfood_consumption")
    if ("durablegoods" in key or key.endswith("durables")) and "code" not in low:
        modules.append("durable_goods")
    if "liabilities" in key:
        modules.append("liabilities")
    if "nonagriculture1" in key or "nonagrilist" in key:
        modules.append("nonagriculture_1")
    if "nonagriculture2" in key or "nonagricost" in key:
        modules.append("nonagriculture_2")
    if "nonagriculture3" in key or "nonagriincome" in key:
        modules.append("nonagriculture_3")
    if "migration" in key:
        modules.append("migration")
    if "livestock1" in key or key.endswith("s05e1livestock"):
        modules.append("livestock_ownership")
    if "livestock2" in key or "livestockexpenses" in key:
        modules.append("livestock_expenses")
    if not village and any(marker in key for marker in ("personecocurrent", "ecocurrent", "labor7days")):
        modules.append("employment_current")
    if "incomeothersource" in key or "incomeother" in key:
        modules.append("other_income")
    if key.endswith("hhhousing") or key in {"housing", "04hhhousing"}:
        modules.append("housing")
    if village and ("ecoinfrastructure" in key or "economicinfrastructure" in key):
        modules.append("village_infrastructure")
    if village and ("rainfalldisaster" in key or "rainfall_disaster" in leaf.lower()):
        modules.append("village_climate_history")
    if any(marker in key for marker in ("weighthouseholds", "weighthousehold", "sizehouseholds")):
        modules.append("household_weights")
    if any(marker in key for marker in ("weightpersons", "weightindividual")):
        modules.append("person_weights")
    return sorted(set(modules))


def read_frame(source: DataSource) -> pd.DataFrame:
    input_obj: Path | io.BytesIO = (
        io.BytesIO(source.read_bytes()) if source.archive_members else source.root_file
    )
    return pd.read_stata(input_obj, convert_categoricals=False)


def canonical_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def ascii_words(value: str, limit: int = 92) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", normalized)
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:limit].rstrip()


def clean_variable_label(label: str, original: str) -> str:
    cleaned = ascii_words(label)
    code = ascii_words(original).replace(" ", "")
    if code:
        cleaned = re.sub(rf"^{re.escape(code)}\s*", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def fallback_name(original: str) -> str:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", original)
    return ascii_words(spaced).title() or "Source Variable"


def readable_name(original: str, label: str) -> str:
    key = canonical_name(original)
    if key in SPECIAL_NAMES:
        return SPECIAL_NAMES[key]
    clean_label = clean_variable_label(label, original)
    if re.fullmatch(r"q[0-9a-z]+", key):
        code = key.upper()
        return f"{code} {clean_label}".strip()
    if clean_label:
        return clean_label
    return fallback_name(original)


def clean_identifier(series: pd.Series, width: int | None = None) -> pd.Series:
    values = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    values = values.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    if width is not None:
        values = values.str.zfill(width)
    return values


def get_series(frame: pd.DataFrame, *aliases: str) -> pd.Series:
    lookup = {canonical_name(str(column)): column for column in frame.columns}
    for alias in aliases:
        key = canonical_name(alias)
        if key in lookup:
            return frame[lookup[key]]
    return pd.Series(pd.NA, index=frame.index, dtype="string")


def normalize_mixed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    string_columns = {
        "Survey Wave",
        "Household ID",
        "Person ID",
        "PSU",
        "Province Code",
        "District Code",
        "Commune Code",
        "Village Code",
        "Province Name",
        "District Name",
        "Commune Name",
        "Village Name",
        "Urban Rural",
        "Survey Month",
        "Source Dataset",
        "Source Submodule",
        "Module Row ID",
    }
    for column in frame.columns:
        series = frame[column]
        if column in string_columns:
            frame[column] = series.astype("string")
            continue
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_numeric_dtype(series):
            continue
        if pd.api.types.is_bool_dtype(series):
            continue
        values = series.astype("string").str.strip().replace("", pd.NA)
        present = values.notna()
        if present.any():
            numeric = pd.to_numeric(values, errors="coerce")
            if numeric[present].notna().mean() >= 0.995:
                frame[column] = numeric
                continue
        frame[column] = values
    return frame


def make_unique_names(mapping: dict[str, str], reserved: set[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    used = set(reserved)
    for original, proposed in mapping.items():
        candidate = proposed
        if candidate in used:
            suffix = ascii_words(original).upper().replace(" ", "") or "VARIABLE"
            candidate = f"{candidate} {suffix}"
        counter = 2
        base = candidate
        while candidate in used:
            candidate = f"{base} {counter}"
            counter += 1
        output[original] = candidate
        used.add(candidate)
    return output


def prepare_crosswalk(root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "cses_village_crosswalk_preprocessed.parquet"
    frame = pd.read_parquet(path)
    frame["Province Code"] = frame["Commune Code"].astype("string").str[:2]
    frame["District Code"] = frame["Commune Code"].astype("string").str[:4]
    columns = [
        "Survey Wave",
        "PSU",
        "Province Code",
        "District Code",
        "Commune Code",
        "Village Code",
        "Province Name",
        "District Name",
        "Commune Name",
        "Village Name",
        "Urban Rural",
        "Survey Month",
    ]
    frame = frame[columns].copy()
    frame["Survey Wave"] = frame["Survey Wave"].astype("string")
    frame["PSU"] = clean_identifier(frame["PSU"], 5)
    conflicts = frame.groupby(["Survey Wave", "PSU"], dropna=False)["Village Code"].nunique(dropna=True)
    if (conflicts > 1).any():
        raise ValueError("CSES crosswalk contains PSU-wave keys linked to multiple village codes")
    return frame.drop_duplicates(["Survey Wave", "PSU"], keep="first")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    processed = root / "data" / "processed"
    exp = root / "data" / "exp" / "data-preprocessing"
    processed.mkdir(parents=True, exist_ok=True)
    exp.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(root)
    selected: dict[str, list[DataSource]] = defaultdict(list)
    for source in sources:
        wave = normalize_wave(source.display_name(root))
        if wave not in WAVE_ORDER:
            continue
        for module in modules_for_source(source, root):
            selected[module].append(source)

    # Prefer the direct 2019 village extract elsewhere and avoid exact duplicated source labels.
    for module in selected:
        unique = {source.display_name(root): source for source in selected[module]}
        selected[module] = sorted(
            unique.values(),
            key=lambda source: (WAVE_ORDER[normalize_wave(source.display_name(root))], source.display_name(root)),
        )

    crosswalk = prepare_crosswalk(root)
    geography_columns = [column for column in crosswalk.columns if column not in {"Survey Wave", "PSU"}]
    manifest_rows: list[dict[str, object]] = []
    dictionary_rows: list[dict[str, object]] = []

    for module in MODULE_TITLES:
        module_sources = selected.get(module, [])
        if not module_sources:
            print(f"{module}: no source located")
            continue

        label_candidates: dict[str, tuple[int, str, str]] = {}
        variable_waves: dict[str, set[str]] = defaultdict(set)
        original_examples: dict[str, str] = {}
        for source in module_sources:
            source_name = source.display_name(root)
            wave = normalize_wave(source_name)
            columns, labels, _ = read_metadata(source)
            for original in columns:
                key = canonical_name(original)
                if key in RAW_IDENTIFIER_NAMES:
                    continue
                variable_waves[key].add(wave)
                original_examples.setdefault(key, original)
                label = str(labels.get(original, "") or "")
                if clean_variable_label(label, original):
                    candidate = (WAVE_ORDER[wave], label, original)
                    if key not in label_candidates or candidate[0] >= label_candidates[key][0]:
                        label_candidates[key] = candidate

        raw_mapping: dict[str, str] = {}
        for key, original in original_examples.items():
            label = label_candidates.get(key, (0, "", original))[1]
            raw_mapping[key] = readable_name(original, label)
        raw_mapping = make_unique_names(raw_mapping, set(FIXED_COLUMNS))

        frames: list[pd.DataFrame] = []
        for source_index, source in enumerate(module_sources, start=1):
            source_name = source.display_name(root)
            wave = normalize_wave(source_name)
            raw = read_frame(source)
            hhid_width = 6 if wave == "2004" else 7
            person_width = 8 if wave == "2004" else 9
            psu_width = 4 if wave == "2004" else 5
            household_id = clean_identifier(get_series(raw, "hhid"), hhid_width)
            person_id = clean_identifier(get_series(raw, "persid"), person_width)
            # Village modules call the PSU ``vid`` or ``vid11``; these are
            # five-digit survey-cluster identifiers, not eight-digit village codes.
            psu = clean_identifier(get_series(raw, "psu", "psu11", "vid", "vid11"), psu_width)
            derived_psu = household_id.str[:psu_width]
            psu = psu.fillna(derived_psu)

            fixed = pd.DataFrame(
                {
                    "Survey Year": WAVE_ORDER[wave],
                    "Survey Wave": wave,
                    "Main Linked Sample": wave != "2004",
                    "Household ID": household_id,
                    "Person ID": person_id,
                    "PSU": psu,
                    "Source Dataset": source_name,
                    "Source Submodule": source_leaf(source),
                    "Module Row ID": [
                        f"{wave}-{source_index:02d}-{row_index + 1:07d}"
                        for row_index in range(len(raw))
                    ],
                }
            )
            substantive: dict[str, pd.Series] = {}
            for original in raw.columns:
                key = canonical_name(str(original))
                if key in RAW_IDENTIFIER_NAMES:
                    continue
                substantive[raw_mapping[key]] = raw[original].reset_index(drop=True)
            standardized = pd.concat([fixed, pd.DataFrame(substantive)], axis=1)

            if wave == "2004":
                for column in geography_columns:
                    standardized[column] = pd.NA
                standardized["Geography Link Matched"] = False
            else:
                standardized = standardized.merge(
                    crosswalk,
                    on=["Survey Wave", "PSU"],
                    how="left",
                    validate="many_to_one",
                )
                standardized["Geography Link Matched"] = standardized["Village Code"].notna()

            for column in FIXED_COLUMNS:
                if column not in standardized:
                    standardized[column] = pd.NA
            remaining = [column for column in standardized.columns if column not in FIXED_COLUMNS]
            standardized = standardized[FIXED_COLUMNS + remaining]
            frames.append(standardized)
            manifest_rows.append(
                {
                    "module": module,
                    "module_title": MODULE_TITLES[module],
                    "survey_wave": wave,
                    "source_dataset": source_name,
                    "source_rows": len(raw),
                    "source_columns": len(raw.columns),
                    "geography_match_rows": int(standardized["Geography Link Matched"].sum()),
                }
            )

        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined = normalize_mixed_columns(combined)
        if len(combined.columns) != len(set(combined.columns)):
            raise ValueError(f"{module}: duplicate readable names")
        if any(not str(column).isascii() for column in combined.columns):
            raise ValueError(f"{module}: non-ASCII readable name")
        destination = processed / f"cses_{module}_preprocessed.parquet"
        combined.to_parquet(destination, index=False)

        for key, readable in raw_mapping.items():
            original = original_examples[key]
            label = label_candidates.get(key, (0, "", original))[1]
            dictionary_rows.append(
                {
                    "module": module,
                    "module_title": MODULE_TITLES[module],
                    "original_name": original,
                    "canonical_original_name": key,
                    "readable_name": readable,
                    "latest_english_label": ascii_words(label),
                    "survey_waves": ";".join(sorted(variable_waves[key], key=WAVE_ORDER.get)),
                    "transformation": "identifier standardization or rename only; substantive values preserved",
                    "output_path": str(destination.relative_to(root)),
                }
            )
        print(
            f"{module}: sources={len(module_sources)}, rows={len(combined):,}, "
            f"columns={len(combined.columns)}, output={destination.relative_to(root)}"
        )

    manifest = pd.DataFrame(manifest_rows).sort_values(["module", "survey_wave", "source_dataset"])
    manifest.to_csv(exp / "direction3_cses_source_manifest.csv", index=False)
    dictionary = pd.DataFrame(dictionary_rows).sort_values(["module", "readable_name"])
    dictionary.to_csv(exp / "direction3_cses_variable_dictionary.csv", index=False)
    print(f"manifest_rows={len(manifest):,}")
    print(f"dictionary_rows={len(dictionary):,}")


if __name__ == "__main__":
    main()
