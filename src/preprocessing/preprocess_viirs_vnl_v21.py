#!/usr/bin/env python3
"""Build a treatment-blinded VIIRS VNL V2.1 pixel-year panel.

This script reads only the cropped VIIRS raster. It does not load the historical
boundary, treatment assignment, rainfall, or any existing outcome estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


YEARS = tuple(range(2013, 2022))
SOURCE_METRICS = ("average_masked", "median_masked", "cf_cvg", "cvg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/raw/independent_validation/viirs_vnl_v21/"
            "viirs_vnl_v21_kampong_speu_boundary_2013_2021.tif"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/viirs_vnl_v21_pixel_year_preprocessed.parquet"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/exp/data-preprocessing/viirs-vnl-v21"),
    )
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def build_panel(dataset: rasterio.io.DatasetReader) -> pd.DataFrame:
    expected_bands = len(YEARS) * len(SOURCE_METRICS)
    if dataset.count != expected_bands:
        raise ValueError(f"Expected {expected_bands} bands, found {dataset.count}")
    if dataset.crs is None or not dataset.crs.is_geographic:
        raise ValueError(f"Expected a geographic CRS, found {dataset.crs}")

    rows, columns = np.indices((dataset.height, dataset.width))
    longitudes, latitudes = rasterio.transform.xy(
        dataset.transform, rows, columns, offset="center"
    )
    longitude = np.asarray(longitudes, dtype="float64").reshape(-1)
    latitude = np.asarray(latitudes, dtype="float64").reshape(-1)
    row_flat = rows.reshape(-1).astype("int16")
    column_flat = columns.reshape(-1).astype("int16")
    cell_id = np.asarray(
        [f"viirs_r{row:03d}_c{column:03d}" for row, column in zip(row_flat, column_flat, strict=True)],
        dtype=object,
    )

    frames: list[pd.DataFrame] = []
    for year_index, year in enumerate(YEARS):
        first_band = year_index * len(SOURCE_METRICS) + 1
        arrays = {
            metric: np.asarray(dataset.read(first_band + offset, masked=True).filled(np.nan), dtype="float32").reshape(-1)
            for offset, metric in enumerate(SOURCE_METRICS)
        }
        cloud_free_share = np.divide(
            arrays["cf_cvg"],
            arrays["cvg"],
            out=np.full(arrays["cf_cvg"].shape, np.nan, dtype="float32"),
            where=arrays["cvg"] > 0,
        )
        frame = pd.DataFrame(
            {
                "Grid Cell ID": cell_id,
                "Grid Row": row_flat,
                "Grid Column": column_flat,
                "Longitude": longitude,
                "Latitude": latitude,
                "Year": np.full(cell_id.size, year, dtype="int16"),
                "Annual Mean Radiance": arrays["average_masked"],
                "Annual Median Radiance": arrays["median_masked"],
                "Cloud-Free Observations": arrays["cf_cvg"],
                "Total Observations": arrays["cvg"],
                "Cloud-Free Observation Share": cloud_free_share,
                "Asinh Annual Mean Radiance": np.arcsinh(arrays["average_masked"]).astype("float32"),
                "Asinh Annual Median Radiance": np.arcsinh(arrays["median_masked"]).astype("float32"),
                "Any Nonzero Annual Mean Radiance": (arrays["average_masked"] != 0).astype("int8"),
                "Any Positive Annual Mean Radiance": (arrays["average_masked"] > 0).astype("int8"),
                "At Least 30 Cloud-Free Observations": (arrays["cf_cvg"] >= 30).astype("int8"),
                "At Least 40 Cloud-Free Observations": (arrays["cf_cvg"] >= 40).astype("int8"),
            }
        )
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel["Grid Cell ID"] = panel["Grid Cell ID"].astype("string")
    return panel


def make_audit(panel: pd.DataFrame, output_dir: Path) -> None:
    audit = (
        panel.groupby("Year", observed=True)
        .agg(
            **{
                "Pixel Rows": ("Grid Cell ID", "size"),
                "Unique Grid Cells": ("Grid Cell ID", "nunique"),
                "Mean Radiance Missing Share": ("Annual Mean Radiance", lambda values: values.isna().mean()),
                "Mean Radiance Zero Share": ("Annual Mean Radiance", lambda values: values.eq(0).mean()),
                "Mean Radiance Positive Share": ("Annual Mean Radiance", lambda values: values.gt(0).mean()),
                "Mean Radiance Negative Share": ("Annual Mean Radiance", lambda values: values.lt(0).mean()),
                "Median Cloud-Free Observations": ("Cloud-Free Observations", "median"),
                "Minimum Cloud-Free Observations": ("Cloud-Free Observations", "min"),
                "Share with at Least 30 Cloud-Free Observations": ("At Least 30 Cloud-Free Observations", "mean"),
                "Share with at Least 40 Cloud-Free Observations": ("At Least 40 Cloud-Free Observations", "mean"),
            }
        )
        .reset_index()
    )
    audit.to_csv(output_dir / "viirs_pixel_year_validation.csv", index=False, float_format="%.8g")


def write_metadata(panel: pd.DataFrame, dataset: rasterio.io.DatasetReader, output_dir: Path) -> None:
    variable_rows = []
    final_variables = {
        "Asinh Annual Mean Radiance": "yes",
        "Asinh Annual Median Radiance": "yes",
        "Any Nonzero Annual Mean Radiance": "yes",
    }
    roles = {
        "Asinh Annual Mean Radiance": "primary outcome",
        "Asinh Annual Median Radiance": "robustness outcome",
        "Any Nonzero Annual Mean Radiance": "robustness outcome",
        "Any Positive Annual Mean Radiance": "reference diagnostic",
        "Cloud-Free Observations": "quality measure",
        "Total Observations": "quality measure",
        "Cloud-Free Observation Share": "quality measure",
        "At Least 30 Cloud-Free Observations": "candidate quality screen",
        "At Least 40 Cloud-Free Observations": "candidate quality screen",
    }
    for column in panel.columns:
        variable_rows.append(
            {
                "readable_name": column,
                "dtype": str(panel[column].dtype),
                "null_percentage": float(100 * panel[column].isna().mean()),
                "is_final_variable": final_variables.get(column, "no"),
                "role": roles.get(column, "linkage key"),
                "preprocessing": "deterministic construction; no imputation; no winsorization",
            }
        )
    pd.DataFrame(variable_rows).to_csv(output_dir / "variable_list.csv", index=False)

    decisions = {
        "dataset": "EOG VIIRS Annual VNL V2.1",
        "period": f"{YEARS[0]}-{YEARS[-1]}",
        "rows": len(panel),
        "unique_grid_cells": int(panel["Grid Cell ID"].nunique()),
        "variables": variable_rows,
        "rules": {
            "primary_outcome": "Asinh Annual Mean Radiance over all pixels, including zeros",
            "robustness_outcomes": [
                "Asinh Annual Median Radiance",
                "Any Nonzero Annual Mean Radiance",
            ],
            "coverage_measures": [
                "Cloud-Free Observations",
                "Total Observations",
                "Cloud-Free Observation Share",
            ],
            "candidate_coverage_thresholds": [30, 40],
            "coverage_threshold_status": "not final until human confirmation",
            "missing_data": "preserve missing; no imputation",
            "outliers": "no clipping or winsorization",
            "negative_radiance": "retain; asinh is defined for negative values",
            "year_2012": "excluded because the Earth Engine V2.1 product uses a non-comparable processing path",
            "treatment_or_outcomes_inspected": False,
        },
        "raster": {
            "width": dataset.width,
            "height": dataset.height,
            "bands": dataset.count,
            "crs": str(dataset.crs),
            "bounds": list(dataset.bounds),
        },
    }
    (output_dir / "decisions.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    readme = f"""# VIIRS VNL V2.1 preprocessing

- Unit: 500 m-class raster grid cell by calendar year.
- Period: {YEARS[0]}-{YEARS[-1]}.
- Rows: {len(panel):,}; unique cells: {panel['Grid Cell ID'].nunique():,}.
- Primary outcome: `Asinh Annual Mean Radiance`, retaining zero and negative values.
- Robustness outcomes: `Asinh Annual Median Radiance` and `Any Nonzero Annual Mean Radiance`.
- Missing values are preserved; no winsorization or imputation is applied.
- Treatment assignment and the historical boundary are not loaded in this script.
- The 30- and 40-observation flags are candidate quality screens, not final exclusions.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_path = resolve(root, args.input)
    output_path = resolve(root, args.output)
    audit_output = resolve(root, args.audit_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_output.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as dataset:
        panel = build_panel(dataset)
        panel.to_parquet(output_path, index=False)
        make_audit(panel, audit_output)
        write_metadata(panel, dataset, audit_output)

    print(f"Wrote {len(panel):,} rows and {len(panel.columns)} columns to {output_path}")
    print(f"Wrote preprocessing audit to {audit_output}")


if __name__ == "__main__":
    main()
