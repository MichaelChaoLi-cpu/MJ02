#!/usr/bin/env python3
"""Extract the Cambodia-Thailand interstate dyad from UCDP Dyadic 26.1."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/UCDP/ucdp-dyadic-261-csv.zip"
OUTPUT = ROOT / "data/processed/ucdp_cambodia_thailand_dyad_years_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/ucdp-cambodia-thailand-dyad"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not RAW.exists():
        raise FileNotFoundError(RAW)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise RuntimeError(f"Expected one CSV, found {members}")
        with archive.open(members[0]) as handle:
            data = pd.read_csv(handle)

    actor_a = data["side_a"].fillna("").str.lower()
    actor_b = data["side_b"].fillna("").str.lower()
    pair = (
        actor_a.str.contains("cambodia") & actor_b.str.contains("thailand")
    ) | (
        actor_a.str.contains("thailand") & actor_b.str.contains("cambodia")
    )
    dyad = data.loc[pair].sort_values(["conflict_id", "dyad_id", "year"]).copy()
    if dyad.empty:
        raise RuntimeError("Cambodia-Thailand government dyad not found")
    dyad.to_parquet(OUTPUT, index=False)
    dyad.to_csv(AUDIT_DIR / "cambodia_thailand_dyad_years.csv", index=False)

    study_period = dyad.loc[dyad["year"].between(2000, 2024)]
    metadata = {
        "source": "UCDP Dyadic Dataset version 26.1",
        "source_url": "https://ucdp.uu.se/downloads/",
        "raw_archive": str(RAW.relative_to(ROOT)),
        "raw_archive_sha256": sha256(RAW),
        "archive_member": members[0],
        "actor_pair_rule": "one side contains Cambodia and the other contains Thailand",
        "conflict_ids": sorted(dyad["conflict_id"].astype(int).unique().tolist()),
        "dyad_ids": sorted(dyad["dyad_id"].astype(int).unique().tolist()),
        "all_coded_years": sorted(dyad["year"].astype(int).unique().tolist()),
        "coded_years_2000_2024": sorted(study_period["year"].astype(int).unique().tolist()),
        "interpretation": "Dyad-year coding establishes UCDP conflict-year eligibility, not a complete chronology of all lower-intensity border incidents.",
    }
    (AUDIT_DIR / "dyad_extract_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
