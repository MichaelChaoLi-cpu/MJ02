#!/usr/bin/env python3
"""Acquire and prepare Cambodia CPI series for Direction 3 monetary deflation.

The authoritative source is the IMF Consumer Price Index SDMX dataflow, which
reports Cambodia series sourced from national authorities.  Only the all-items,
food and non-alcoholic beverages, and education components are requested.  A
compact source snapshot is retained under data/exp so ordinary preprocessing can
be reproduced offline; pass --refresh to update it from the public API.
"""

from __future__ import annotations

import argparse
import io
import ssl
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import certifi


API_BASE = (
    "https://api.imf.org/external/sdmx/3.0/data/dataflow/"
    "IMF.STA/CPI/5.0.0/*.*.*.*.*"
)
COMPONENTS = {
    "_T": "All Items",
    "CP01": "Food and Non-alcoholic Beverages",
    "CP10": "Education",
}
START_PERIOD = "2007-01"
END_PERIOD = "2021-12"


def query_url(component: str) -> str:
    constraints = {
        "c[COUNTRY]": "KHM",
        "c[INDEX_TYPE]": "CPI",
        "c[COICOP_1999]": component,
        "c[TYPE_OF_TRANSFORMATION]": "IX",
        "c[FREQUENCY]": "M",
        "c[TIME_PERIOD]": f"ge:{START_PERIOD}+le:{END_PERIOD}",
        "attributes": "none",
        "detail": "dataonly",
        "includeHistory": "false",
    }
    return f"{API_BASE}?{urlencode(constraints)}"


def download_component(component: str) -> pd.DataFrame:
    request = Request(
        query_url(component),
        headers={"Accept": "text/csv;version=2.0.0", "User-Agent": "MJ02-research/1.0"},
    )
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=120, context=ssl_context) as response:
        payload = response.read().decode("utf-8-sig")
    frame = pd.read_csv(io.StringIO(payload))
    required = [
        "COUNTRY",
        "INDEX_TYPE",
        "COICOP_1999",
        "TYPE_OF_TRANSFORMATION",
        "FREQUENCY",
        "TIME_PERIOD",
        "OBS_VALUE",
    ]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"IMF CPI response lacks columns: {sorted(missing)}")
    return frame[required].copy()


def validate_source(frame: pd.DataFrame) -> None:
    if frame.duplicated(["COICOP_1999", "TIME_PERIOD"]).any():
        raise ValueError("Duplicate CPI component-month keys")
    expected = pd.period_range(START_PERIOD, END_PERIOD, freq="M").astype(str)
    for component in COMPONENTS:
        subset = frame.loc[frame["COICOP_1999"].eq(component)].copy()
        observed = subset["TIME_PERIOD"].str.replace("-M", "-", regex=False)
        if len(subset) != len(expected) or set(observed) != set(expected):
            raise ValueError(f"Incomplete monthly CPI coverage for {component}")
        values = pd.to_numeric(subset["OBS_VALUE"], errors="coerce")
        if values.isna().any() or values.le(0).any():
            raise ValueError(f"Invalid CPI values for {component}")


def prepare(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = source.copy()
    monthly["Year"] = monthly["TIME_PERIOD"].str.slice(0, 4).astype(int)
    monthly["Month"] = monthly["TIME_PERIOD"].str.extract(r"M(\d{2})")[0].astype(int)
    monthly["CPI Component Code"] = monthly["COICOP_1999"]
    monthly["CPI Component"] = monthly["CPI Component Code"].map(COMPONENTS)
    monthly["CPI Index"] = pd.to_numeric(monthly["OBS_VALUE"], errors="raise")
    base = monthly.loc[monthly["Year"].eq(2021)].groupby("CPI Component Code")[
        "CPI Index"
    ].mean()
    monthly["CPI 2021 Annual Mean"] = monthly["CPI Component Code"].map(base)
    monthly["Deflator to 2021"] = monthly["CPI 2021 Annual Mean"] / monthly["CPI Index"]
    monthly["CPI Rebased 2021 Mean Equals 100"] = (
        100 * monthly["CPI Index"] / monthly["CPI 2021 Annual Mean"]
    )
    monthly["Original CPI Reference Period"] = "October-December 2006 = 100"
    monthly["Source"] = "IMF Consumer Price Index (CPI), IMF.STA:CPI(5.0.0)"
    monthly["Source Country"] = "Cambodia"
    monthly = monthly[
        [
            "Year",
            "Month",
            "CPI Component Code",
            "CPI Component",
            "CPI Index",
            "CPI 2021 Annual Mean",
            "Deflator to 2021",
            "CPI Rebased 2021 Mean Equals 100",
            "Original CPI Reference Period",
            "Source",
            "Source Country",
        ]
    ].sort_values(["CPI Component Code", "Year", "Month"])

    annual = monthly.groupby(
        ["Year", "CPI Component Code", "CPI Component"], as_index=False
    ).agg(
        **{
            "Annual Mean CPI Index": ("CPI Index", "mean"),
            "Observed CPI Months": ("CPI Index", "count"),
            "CPI 2021 Annual Mean": ("CPI 2021 Annual Mean", "first"),
        }
    )
    annual["Annual Deflator to 2021"] = (
        annual["CPI 2021 Annual Mean"] / annual["Annual Mean CPI Index"]
    )
    annual["Annual CPI Rebased 2021 Mean Equals 100"] = (
        100 * annual["Annual Mean CPI Index"] / annual["CPI 2021 Annual Mean"]
    )
    return monthly.reset_index(drop=True), annual.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    exp = root / "data" / "exp" / "data-preprocessing"
    source_dir = exp / "cpi_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = source_dir / "imf_cambodia_cpi_monthly_2007_2021.csv"

    if args.refresh or not snapshot_path.exists():
        source = pd.concat(
            [download_component(component) for component in COMPONENTS],
            ignore_index=True,
        )
        validate_source(source)
        source.sort_values(["COICOP_1999", "TIME_PERIOD"]).to_csv(snapshot_path, index=False)
    else:
        source = pd.read_csv(snapshot_path)
        validate_source(source)

    monthly, annual = prepare(source)
    processed = root / "data" / "processed"
    monthly_path = processed / "cambodia_cpi_monthly_preprocessed.parquet"
    annual_path = processed / "cambodia_cpi_annual_preprocessed.parquet"
    monthly.to_parquet(monthly_path, index=False)
    annual.to_parquet(annual_path, index=False)

    validation = monthly.groupby("CPI Component", as_index=False).agg(
        **{
            "Monthly Rows": ("CPI Index", "size"),
            "First Year": ("Year", "min"),
            "Last Year": ("Year", "max"),
            "Missing CPI Values": ("CPI Index", lambda values: int(values.isna().sum())),
            "Minimum CPI Index": ("CPI Index", "min"),
            "Maximum CPI Index": ("CPI Index", "max"),
            "CPI 2021 Annual Mean": ("CPI 2021 Annual Mean", "first"),
        }
    )
    validation.to_csv(exp / "direction3_cpi_validation.csv", index=False)
    manifest = pd.DataFrame(
        [
            {
                "component_code": component,
                "component_name": name,
                "source_dataset": "IMF Consumer Price Index (CPI), IMF.STA:CPI(5.0.0)",
                "source_url": query_url(component),
                "source_snapshot": str(snapshot_path.relative_to(root)),
                "period": f"{START_PERIOD} to {END_PERIOD}",
                "frequency": "monthly",
                "original_reference_period": "October-December 2006 = 100",
                "target_price_basis": "2021 annual mean = 100",
            }
            for component, name in COMPONENTS.items()
        ]
    )
    manifest.to_csv(exp / "direction3_cpi_source_manifest.csv", index=False)
    print(f"source_rows={len(source):,}")
    print(f"monthly_rows={len(monthly):,}, output={monthly_path.relative_to(root)}")
    print(f"annual_rows={len(annual):,}, output={annual_path.relative_to(root)}")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
