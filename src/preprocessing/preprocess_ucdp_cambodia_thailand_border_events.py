#!/usr/bin/env python3
"""Extract and audit Cambodia-Thailand border-conflict candidates from UCDP GED."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_ZIP = ROOT / "data/raw/UCDP/ged261-csv.zip"
REGIONAL_OUTPUT = ROOT / "data/processed/ucdp_cambodia_region_events_2000_2024_preprocessed.parquet"
CANDIDATE_OUTPUT = ROOT / "data/processed/ucdp_cambodia_thailand_state_conflict_candidates_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/ucdp-cambodia-thailand-events"

START_DATE = pd.Timestamp("2000-01-01")
END_DATE = pd.Timestamp("2024-12-31")
MIN_LAT, MAX_LAT = 9.5, 15.6
MIN_LON, MAX_LON = 101.0, 108.7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ged_zip(path: Path) -> tuple[pd.DataFrame, str]:
    with zipfile.ZipFile(path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise RuntimeError(f"Expected exactly one CSV in UCDP archive, found {csv_members}")
        member = csv_members[0]
        with archive.open(member) as handle:
            return pd.read_csv(handle, low_memory=False), member


def actor_text(frame: pd.DataFrame) -> pd.Series:
    fields = [name for name in ["side_a", "side_b", "dyad_name", "conflict_name"] if name in frame.columns]
    if not fields:
        raise RuntimeError("UCDP actor/dyad fields not found")
    return frame[fields].fillna("").astype(str).agg(" | ".join, axis=1)


def main() -> None:
    if not RAW_ZIP.exists():
        raise FileNotFoundError(RAW_ZIP)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    REGIONAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    data, member = read_ged_zip(RAW_ZIP)
    data.columns = [re.sub(r"\s+", "_", str(name).strip().lower()) for name in data.columns]
    required = {"id", "latitude", "longitude", "date_start", "date_end"}
    if not required.issubset(data.columns):
        raise RuntimeError(f"Missing required UCDP fields: {sorted(required - set(data.columns))}")

    data["date_start"] = pd.to_datetime(data["date_start"], errors="coerce")
    data["date_end"] = pd.to_datetime(data["date_end"], errors="coerce")
    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")

    regional = data.loc[
        data["date_end"].between(START_DATE, END_DATE)
        & data["latitude"].between(MIN_LAT, MAX_LAT)
        & data["longitude"].between(MIN_LON, MAX_LON)
    ].copy()
    actors = actor_text(regional).str.lower()
    cambodia = actors.str.contains(r"\bcambodia(?:n)?\b", regex=True)
    thailand = actors.str.contains(r"\bthai(?:land)?\b", regex=True)
    regional["cambodia_thailand_actor_pair_candidate"] = cambodia & thailand
    regional["candidate_review_reason"] = regional["cambodia_thailand_actor_pair_candidate"].map(
        {True: "actor/dyad text contains both Cambodia and Thailand", False: "regional context only"}
    )

    candidates = regional.loc[regional["cambodia_thailand_actor_pair_candidate"]].copy()
    regional.to_parquet(REGIONAL_OUTPUT, index=False)
    candidates.to_parquet(CANDIDATE_OUTPUT, index=False)

    review_columns = [
        name
        for name in [
            "id", "year", "date_start", "date_end", "date_prec", "where_prec",
            "country", "conflict_name", "dyad_name", "side_a", "side_b",
            "where_coordinates", "where_description", "adm_1", "adm_2",
            "latitude", "longitude", "best", "low", "high", "source_article",
        ]
        if name in candidates.columns
    ]
    candidates[review_columns].to_csv(AUDIT_DIR / "candidate_events_for_manual_review.csv", index=False)

    by_year = (
        candidates.groupby(candidates["date_end"].dt.year, dropna=False)
        .agg(Events=("id", "size"), Sites=("where_coordinates", "nunique"), Best_Deaths=("best", "sum"))
        .reset_index(names="Year")
    ) if len(candidates) else pd.DataFrame(columns=["Year", "Events", "Sites", "Best_Deaths"])
    by_year.to_csv(AUDIT_DIR / "candidate_event_counts_by_year.csv", index=False)

    metadata = {
        "source": "UCDP Georeferenced Event Dataset Global version 26.1",
        "source_url": "https://ucdp.uu.se/downloads/",
        "license": "CC BY 4.0 according to the UCDP download center",
        "raw_archive": str(RAW_ZIP.relative_to(ROOT)),
        "raw_archive_sha256": sha256(RAW_ZIP),
        "archive_member": member,
        "raw_rows": int(len(data)),
        "regional_rows_2000_2024": int(len(regional)),
        "actor_pair_candidate_rows": int(len(candidates)),
        "regional_bounding_box": {"min_lat": MIN_LAT, "max_lat": MAX_LAT, "min_lon": MIN_LON, "max_lon": MAX_LON},
        "date_window": [str(START_DATE.date()), str(END_DATE.date())],
        "candidate_rule": "regional event whose actor/dyad/conflict text contains both Cambodia and Thailand",
        "status": "candidate extract only; requires manual reconciliation with official chronology before treatment is frozen",
        "no_imputation": True,
    }
    (AUDIT_DIR / "ucdp_event_extract_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
