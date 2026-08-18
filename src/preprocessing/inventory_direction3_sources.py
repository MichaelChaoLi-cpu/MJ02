#!/usr/bin/env python3
"""Build a reproducible source and variable inventory for Direction 3.

The script reads Stata metadata from expanded files and ZIP archives without
modifying raw data. It classifies candidate variables for the mine exposure,
climate resilience, outcome, mechanism, and linkage layers.
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd


MAX_ARCHIVE_DEPTH = 3
WAVE_PATTERN = re.compile(r"20(?:04|07|09|11|12|13|14|16|17|19|21)")

MODULE_KEYWORDS = {
    "linkage": (
        "hhid", "household id", "persid", "person id", "psu", "province",
        "district", "commune", "village", "urban", "rural", "strata",
        "stratum", "weight", "sample unit", "identification",
    ),
    "agriculture": (
        "agric", "cultivat", "crop", "harvest", "farm", "farmland", "land",
        "irrig", "fertili", "livestock", "fish", "forest", "pesticide",
    ),
    "welfare": (
        "consum", "expend", "food security", "food insecurity", "income",
        "poverty", "poor", "durable", "asset", "vulnerab", "debt",
    ),
    "education": (
        "educ", "school", "student", "enrol", "attend", "literacy", "grade",
    ),
    "climate_disaster": (
        "rain", "flood", "drought", "storm", "disaster", "weather", "climate",
    ),
    "mechanism": (
        "migrat", "employment", "employed", "labor", "labour", "nonagri",
        "non-agri", "liabil", "credit", "loan", "road", "market",
        "infrastructure", "price", "transport", "access",
    ),
}

SOURCE_HINTS = {
    "linkage": ("psulist", "area", "heading", "demograp", "household", "member"),
    "agriculture": ("agri", "land", "crop", "livestock", "fish", "forest", "cultivation"),
    "welfare": ("consum", "expend", "food", "income", "durable", "vulnerab"),
    "education": ("educ",),
    "climate_disaster": ("rain", "disaster"),
    "mechanism": ("migrat", "employment", "labor", "nonagri", "liabil", "price", "infrastructure"),
}


@dataclass(frozen=True)
class DataSource:
    root_file: Path
    archive_members: tuple[str, ...] = ()

    @property
    def suffix(self) -> str:
        target = self.archive_members[-1] if self.archive_members else self.root_file.name
        return PurePosixPath(target).suffix.lower()

    def display_name(self, root: Path) -> str:
        label = str(self.root_file.relative_to(root))
        if self.archive_members:
            label += "::" + "::".join(self.archive_members)
        return label

    def read_bytes(self) -> bytes:
        payload: bytes | None = None
        for member in self.archive_members:
            archive_input = self.root_file if payload is None else io.BytesIO(payload)
            with zipfile.ZipFile(archive_input) as archive:
                payload = archive.read(member)
        if payload is None:
            raise ValueError(f"No archived payload for {self.root_file}")
        return payload


def is_noise(name: str) -> bool:
    return any(part == "__MACOSX" or part.startswith("._") for part in PurePosixPath(name).parts)


def discover_archive(
    archive_path: Path,
    members: tuple[str, ...] = (),
    archive_bytes: bytes | None = None,
    depth: int = 0,
) -> list[DataSource]:
    if depth > MAX_ARCHIVE_DEPTH:
        return []
    archive_input = archive_path if archive_bytes is None else io.BytesIO(archive_bytes)
    found: list[DataSource] = []
    try:
        with zipfile.ZipFile(archive_input) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                if info.is_dir() or is_noise(info.filename):
                    continue
                suffix = PurePosixPath(info.filename).suffix.lower()
                chain = members + (info.filename,)
                if suffix == ".dta":
                    found.append(DataSource(archive_path, chain))
                elif suffix == ".zip":
                    try:
                        nested = archive.read(info)
                    except Exception:
                        continue
                    found.extend(discover_archive(archive_path, chain, nested, depth + 1))
    except (OSError, zipfile.BadZipFile):
        return []
    return found


def discover_sources(root: Path) -> list[DataSource]:
    raw = root / "data" / "raw" / "CSE"
    sources = [DataSource(path) for path in raw.rglob("*.dta")]
    for archive in raw.rglob("*.zip"):
        if archive.with_suffix("").is_dir():
            continue
        sources.extend(discover_archive(archive))
    return sorted(sources, key=lambda source: source.display_name(root))


def normalize_wave(source_name: str) -> str:
    matches = WAVE_PATTERN.findall(source_name)
    if not matches:
        return "unknown"
    raw = matches[0]
    return {
        "2004": "2004", "2007": "2007", "2009": "2009", "2011": "2011-12",
        "2012": "2011-12", "2013": "2013", "2014": "2014", "2016": "2016",
        "2017": "2017", "2019": "2019", "2021": "2021",
    }[raw]


def read_metadata(source: DataSource) -> tuple[list[str], dict[str, str], int | None]:
    input_obj: Path | io.BytesIO
    input_obj = io.BytesIO(source.read_bytes()) if source.archive_members else source.root_file
    reader = pd.io.stata.StataReader(input_obj, convert_categoricals=False)
    labels = reader.variable_labels()
    try:
        columns = list(reader.read(0).columns)
    except Exception:
        columns = list(labels)
    nobs = getattr(reader, "nobs", None)
    close = getattr(reader, "close", None)
    if close is not None:
        close()
    return columns, labels, int(nobs) if nobs is not None else None


def match_modules(variable: str, label: str, source_name: str) -> tuple[list[str], str]:
    variable_text = variable.lower().replace("_", " ")
    label_text = (label or "").lower()
    combined = f"{variable_text} {label_text}"
    modules = [
        module for module, keywords in MODULE_KEYWORDS.items()
        if any(keyword in combined for keyword in keywords)
    ]
    reason = "variable_or_label"
    if not modules:
        source_lower = source_name.lower()
        modules = [
            module for module, hints in SOURCE_HINTS.items()
            if any(hint in source_lower for hint in hints)
        ]
        reason = "source_name"
    return sorted(set(modules)), reason


def load_existing_inventory(root: Path) -> dict[tuple[str, str], dict[str, str]]:
    path = root / "data" / "exp" / "data-preprocessing" / "variable_list.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype=str).fillna("")
    return {
        (row["source_dataset"], row["original_name"]): row.to_dict()
        for _, row in frame.iterrows()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = root / "data" / "exp" / "data-preprocessing"
    output.mkdir(parents=True, exist_ok=True)
    existing = load_existing_inventory(root)

    variable_rows: list[dict[str, object]] = []
    dataset_rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for source in discover_sources(root):
        source_name = source.display_name(root)
        try:
            variables, labels, nobs = read_metadata(source)
        except Exception as exc:
            errors.append({"source_dataset": source_name, "error": str(exc)})
            continue

        module_counts: Counter[str] = Counter()
        candidate_count = 0
        for variable in variables:
            label = labels.get(variable, "")
            modules, reason = match_modules(variable, label, source_name)
            if not modules:
                continue
            candidate_count += 1
            module_counts.update(modules)
            details = existing.get((source_name, variable), {})
            variable_rows.append({
                "survey_year": normalize_wave(source_name),
                "source_dataset": source_name,
                "original_name": variable,
                "variable_label": label,
                "candidate_modules": ";".join(modules),
                "match_reason": reason,
                "dtype": details.get("dtype", ""),
                "null_pct_sample": details.get("null_pct", ""),
                "sample_values": details.get("sample_values", ""),
            })
        dataset_rows.append({
            "survey_year": normalize_wave(source_name),
            "source_dataset": source_name,
            "row_count": nobs if nobs is not None else "",
            "variable_count": len(variables),
            "candidate_variable_count": candidate_count,
            "candidate_modules": ";".join(sorted(module_counts)),
            **{f"{module}_matches": module_counts.get(module, 0) for module in MODULE_KEYWORDS},
        })

    variables = pd.DataFrame(variable_rows).sort_values(
        ["survey_year", "candidate_modules", "source_dataset", "original_name"]
    )
    datasets = pd.DataFrame(dataset_rows).sort_values(
        ["survey_year", "candidate_variable_count", "source_dataset"],
        ascending=[True, False, True],
    )
    variables.to_csv(output / "direction3_candidate_variables.csv", index=False)
    datasets.to_csv(output / "direction3_dataset_registry.csv", index=False)
    pd.DataFrame(errors, columns=["source_dataset", "error"]).to_csv(
        output / "direction3_inventory_errors.csv", index=False
    )

    by_wave = defaultdict(Counter)
    for row in variable_rows:
        for module in str(row["candidate_modules"]).split(";"):
            if module:
                by_wave[str(row["survey_year"])][module] += 1

    print(f"datasets_scanned={len(dataset_rows)}")
    print(f"candidate_variables={len(variable_rows)}")
    print(f"errors={len(errors)}")
    for wave in sorted(by_wave):
        counts = ", ".join(f"{key}={value}" for key, value in sorted(by_wave[wave].items()))
        print(f"{wave}: {counts}")


if __name__ == "__main__":
    main()
