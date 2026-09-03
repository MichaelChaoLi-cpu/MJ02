#!/usr/bin/env python3
"""Build a balanced LongNTL Version 2 grid-cell by year panel."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy


YEAR_PATTERN = re.compile(r"longntl_v2_(\d{4})_kampong_speu_boundary\.tif$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/independent_validation/longntl_v2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/longntl_v2_pixel_year_preprocessed.parquet"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/exp/data-preprocessing/longntl-v2"),
    )
    parser.add_argument(
        "--reference-grid",
        type=Path,
        default=Path(
            "data/raw/independent_validation/viirs_vnl_v21/"
            "viirs_vnl_v21_kampong_speu_boundary_2013_2021.tif"
        ),
    )
    return parser.parse_args()


def resolved(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_panel(source: Path, reference_grid: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    paths = sorted(source.glob("longntl_v2_*_kampong_speu_boundary.tif"))
    if not paths:
        raise FileNotFoundError(f"No annual LongNTL crops found under {source}")

    with rasterio.open(reference_grid) as reference_dataset:
        target_width = reference_dataset.width
        target_height = reference_dataset.height
        target_transform = reference_dataset.transform
        target_crs = reference_dataset.crs
    target_signature = (
        target_width,
        target_height,
        tuple(target_transform),
        "WGS84",
    )
    target_rows, target_columns = np.indices((target_height, target_width))
    longitudes, latitudes = xy(
        target_transform, target_rows, target_columns, offset="center"
    )
    frames = []
    for path in paths:
        match = YEAR_PATTERN.match(path.name)
        if not match:
            continue
        year = int(match.group(1))
        with rasterio.open(path) as dataset:
            source_values = dataset.read(1, masked=True)
            source_array = np.asarray(source_values.filled(np.nan), dtype="float32")
            target_array = np.full((target_height, target_width), np.nan, dtype="float32")
            # All sources and the reference are WGS84 north-up rasters with the
            # same nominal resolution. Map each reference-cell centre to its
            # containing native source cell. This is nearest-neighbour alignment
            # without invoking a PROJ-dependent coordinate transformation.
            inverse = ~dataset.transform
            source_columns_float, source_rows_float = inverse * (
                np.asarray(longitudes),
                np.asarray(latitudes),
            )
            source_columns = np.floor(source_columns_float).astype(int)
            source_rows = np.floor(source_rows_float).astype(int)
            valid = (
                (source_rows >= 0)
                & (source_rows < dataset.height)
                & (source_columns >= 0)
                & (source_columns < dataset.width)
            )
            target_flat = target_array.ravel()
            target_flat[valid] = source_array[source_rows[valid], source_columns[valid]]
            radiance = target_array.ravel()
            frame = pd.DataFrame(
                {
                    "LongNTL Grid Cell ID": [
                        f"longntl_r{row:03d}_c{column:03d}"
                        for row, column in zip(
                            target_rows.ravel(), target_columns.ravel(), strict=True
                        )
                    ],
                    "Grid Row": target_rows.ravel().astype("int16"),
                    "Grid Column": target_columns.ravel().astype("int16"),
                    "Longitude": np.asarray(longitudes).ravel(),
                    "Latitude": np.asarray(latitudes).ravel(),
                    "Year": year,
                    "Annual NPP-VIIRS-like Radiance": radiance,
                }
            )
            frames.append(frame)

    panel = pd.concat(frames, ignore_index=True)
    panel["Asinh Annual NPP-VIIRS-like Radiance"] = np.arcsinh(
        panel["Annual NPP-VIIRS-like Radiance"]
    )
    panel["Any Nonzero Annual NPP-VIIRS-like Radiance"] = (
        panel["Annual NPP-VIIRS-like Radiance"].ne(0)
        & panel["Annual NPP-VIIRS-like Radiance"].notna()
    ).astype("Int8")
    panel["Any Positive Annual NPP-VIIRS-like Radiance"] = (
        panel["Annual NPP-VIIRS-like Radiance"].gt(0)
    ).astype("Int8")
    panel["LongNTL Source Stage"] = np.where(
        panel["Year"].le(2012), "reconstructed", "observed annual composite"
    )
    panel["Reconstructed LongNTL Indicator"] = panel["Year"].le(2012).astype("Int8")
    panel = panel.sort_values(["LongNTL Grid Cell ID", "Year"]).reset_index(drop=True)

    width, height, transform, crs = target_signature
    metadata = {
        "years": [int(panel["Year"].min()), int(panel["Year"].max())],
        "year_count": int(panel["Year"].nunique()),
        "grid_cells": int(panel["LongNTL Grid Cell ID"].nunique()),
        "rows": int(len(panel)),
        "width": int(width),
        "height": int(height),
        "transform": list(transform),
        "crs": crs,
        "reference_grid": str(reference_grid),
        "grid_alignment_rule": (
            "Nearest-neighbour alignment to the existing EOG VNL V2.1 analysis grid; "
            "native resolutions are the same and origin offsets are sub-pixel"
        ),
        "unit": "nW cm-2 sr-1",
        "reconstruction_period": [2000, 2012],
        "observed_composite_period": [2013, 2024],
        "primary_outcome": "Asinh Annual NPP-VIIRS-like Radiance",
        "missing_rule": "Preserve source missing values; no imputation",
        "zero_rule": "Retain source zeros as zeros",
        "outlier_rule": "No winsorization or clipping",
    }
    return panel, metadata


def validate(panel: pd.DataFrame, metadata: dict[str, object]) -> None:
    keys = ["LongNTL Grid Cell ID", "Year"]
    if panel.duplicated(keys).any():
        raise RuntimeError("Duplicate grid-cell-year keys")
    years_per_cell = panel.groupby("LongNTL Grid Cell ID")["Year"].nunique()
    if not years_per_cell.eq(metadata["year_count"]).all():
        raise RuntimeError("LongNTL panel is not balanced")
    if panel["Year"].nunique() != 25 or panel["Year"].min() != 2000 or panel["Year"].max() != 2024:
        raise RuntimeError("Expected complete 2000-2024 coverage")
    if panel["Annual NPP-VIIRS-like Radiance"].lt(0).any():
        raise RuntimeError("Unexpected negative LongNTL radiance")


def audit(panel: pd.DataFrame, output: Path, metadata: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    outcome = "Annual NPP-VIIRS-like Radiance"
    asinh = "Asinh Annual NPP-VIIRS-like Radiance"
    coverage = (
        panel.groupby(["Year", "LongNTL Source Stage"], observed=True)
        .agg(
            grid_cells=("LongNTL Grid Cell ID", "size"),
            observed_radiance=(outcome, "count"),
            zero_radiance=(outcome, lambda x: int(x.eq(0).sum())),
            positive_radiance=(outcome, lambda x: int(x.gt(0).sum())),
            mean_radiance=(outcome, "mean"),
            median_radiance=(outcome, "median"),
            p95_radiance=(outcome, lambda x: float(x.quantile(0.95))),
            maximum_radiance=(outcome, "max"),
            mean_asinh_radiance=(asinh, "mean"),
        )
        .reset_index()
    )
    coverage["observed_share"] = coverage["observed_radiance"] / coverage["grid_cells"]
    coverage["positive_share"] = coverage["positive_radiance"] / coverage["grid_cells"]
    coverage["annual_change_mean_asinh"] = coverage["mean_asinh_radiance"].diff()
    coverage["annual_change_positive_share"] = coverage["positive_share"].diff()
    coverage.to_csv(output / "coverage_and_distribution_by_year.csv", index=False)

    wide = panel.pivot(index="LongNTL Grid Cell ID", columns="Year", values=asinh)
    correlations = []
    for year in range(2001, 2025):
        left = wide[year - 1]
        right = wide[year]
        observed = left.notna() & right.notna()
        pearson = left.loc[observed].corr(right.loc[observed])
        rank_pearson = left.loc[observed].rank().corr(right.loc[observed].rank())
        correlations.append(
            {
                "from_year": year - 1,
                "to_year": year,
                "cross_sensor_transition": year == 2013,
                "observed_grid_cells": int(observed.sum()),
                "pearson_correlation": pearson,
                "spearman_rank_correlation": rank_pearson,
            }
        )
    correlation_table = pd.DataFrame(correlations)
    correlation_table.to_csv(output / "adjacent_year_cell_consistency.csv", index=False)

    transition = coverage.loc[coverage["Year"].between(2010, 2015)].merge(
        correlation_table,
        left_on="Year",
        right_on="to_year",
        how="left",
        validate="one_to_one",
    )
    transition.to_csv(output / "source_transition_diagnostic_2010_2015.csv", index=False)
    (output / "panel_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    readme = f"""# LongNTL Version 2 preprocessing

- Balanced panel: {metadata['grid_cells']:,} grid cells by {metadata['year_count']} years ({metadata['rows']:,} rows).
- Period: 2000-2024.
- Reconstructed source stage: 2000-2012.
- Observed annual-composite source stage: 2013-2024.
- Primary outcome: `Asinh Annual NPP-VIIRS-like Radiance`.
- Source zeros are retained. Missing values are not imputed. Values are not winsorized.
- The 2012/2013 source transition is audited before any historical-side coefficient is estimated.
- The accepted EOG VNL V2.1 panel remains a separate overlap-period measurement benchmark.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source = resolved(root, args.input)
    output = resolved(root, args.output)
    audit_output = resolved(root, args.audit_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reference_grid = resolved(root, args.reference_grid)
    panel, metadata = load_panel(source, reference_grid)
    validate(panel, metadata)
    panel.to_parquet(output, index=False)
    audit(panel, audit_output, metadata)
    print(
        f"Wrote {len(panel):,} rows, {panel['LongNTL Grid Cell ID'].nunique():,} cells, "
        f"{panel['Year'].nunique()} years to {output}"
    )


if __name__ == "__main__":
    main()
