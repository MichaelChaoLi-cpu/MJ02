#!/usr/bin/env python3
"""Preprocess public same-boundary poverty and human-capital outcomes.

The public replication source is an R spatial object. R is used only to extract the
documented fields and coordinates; pandas validates, renames, and writes the Parquet.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/exp/data-preprocessing/historical-boundary-source/main_0310.rds"
OUTPUT = ROOT / "data/processed/grasse_historical_boundary_level_outcomes_preprocessed.parquet"
EXP_DIR = ROOT / "data/exp/data-preprocessing/grasse-boundary-level-outcomes"


RENAME = {
    "vill_code": "Village Code",
    "vill_name": "Village Name",
    "comm": "Published Historical Commune",
    "treat": "Higher-Repression Southwest Zone",
    "dist_border": "Signed Distance to Historical Repression Boundary km",
    "dist.segment": "Historical Boundary Segment",
    "pov": "Village Poverty Rate Percent",
    "yrseduc": "Mean Years of Schooling",
    "t_lit15": "Adult Literacy Rate Percent",
    "ihs_light": "Published IHS Nighttime Luminosity",
}


def extract_with_r(destination: Path) -> None:
    expression = (
        f'x<-readRDS("{SOURCE}"); '
        'd<-as.data.frame(x)[,c("vill_code","vill_name","comm","treat","dist_border",'
        '"dist.segment","pov","yrseduc","t_lit15","ihs_light")]; '
        'xy<-sf::st_coordinates(sf::st_transform(x,4326)); d$Longitude<-xy[,1]; d$Latitude<-xy[,2]; '
        f'write.csv(d,"{destination}",row.names=FALSE,na="")'
    )
    subprocess.run(["Rscript", "-e", expression], check=True)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    with tempfile.TemporaryDirectory(prefix="mj02-grasse-level-") as temp_dir:
        extracted = Path(temp_dir) / "level_outcomes.csv"
        extract_with_r(extracted)
        data = pd.read_csv(extracted)
    missing = sorted(set(RENAME).difference(data.columns))
    if missing:
        raise ValueError(f"Missing expected public fields: {missing}")
    data = data.rename(columns=RENAME)
    output_columns = list(RENAME.values()) + ["Longitude", "Latitude"]
    data = data[output_columns].copy()
    data["Village Code"] = data["Village Code"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)
    numeric = [
        "Higher-Repression Southwest Zone",
        "Signed Distance to Historical Repression Boundary km",
        "Historical Boundary Segment",
        "Village Poverty Rate Percent",
        "Mean Years of Schooling",
        "Adult Literacy Rate Percent",
        "Published IHS Nighttime Luminosity",
        "Longitude",
        "Latitude",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data["Village Code"].duplicated().any():
        raise ValueError("Public village code is not unique")
    if not data["Higher-Repression Southwest Zone"].dropna().isin([0, 1]).all():
        raise ValueError("Historical-side assignment is not binary")
    if not data["Village Poverty Rate Percent"].dropna().between(0, 100).all():
        raise ValueError("Poverty rate is outside [0,100]")
    if not data["Adult Literacy Rate Percent"].dropna().between(0, 100).all():
        raise ValueError("Literacy rate is outside [0,100]")
    if not data["Mean Years of Schooling"].dropna().between(0, 20).all():
        raise ValueError("Mean years of schooling is outside [0,20]")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT, index=False)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame([
        {"check": "rows", "value": len(data), "status": "pass" if len(data) == 1359 else "review"},
        {"check": "unique village codes", "value": data["Village Code"].nunique(), "status": "pass"},
        {"check": "poverty nonmissing", "value": data["Village Poverty Rate Percent"].notna().sum(), "status": "pass"},
        {"check": "schooling nonmissing", "value": data["Mean Years of Schooling"].notna().sum(), "status": "pass"},
        {"check": "literacy nonmissing", "value": data["Adult Literacy Rate Percent"].notna().sum(), "status": "pass"},
        {"check": "longitude range", "value": f"{data['Longitude'].min():.4f} to {data['Longitude'].max():.4f}", "status": "pass" if data["Longitude"].between(102, 108).all() else "fail"},
        {"check": "latitude range", "value": f"{data['Latitude'].min():.4f} to {data['Latitude'].max():.4f}", "status": "pass" if data["Latitude"].between(10, 15).all() else "fail"},
    ])
    audit.to_csv(EXP_DIR / "preprocessing_audit.csv", index=False)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Rows: {len(data)}; columns: {len(data.columns)}")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
