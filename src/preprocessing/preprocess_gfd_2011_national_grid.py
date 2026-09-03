#!/usr/bin/env python3
"""Aggregate the two 2011 Global Flood Database events to Cambodia's 1 km grid."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject


GRID = Path("data/processed/cambodia_national_1km_grid_preprocessed.parquet")
OUTPUT = Path("data/processed/cambodia_national_2011_gfd_flood_exposure_preprocessed.parquet")
AUDIT_DIR = Path("data/exp/data-preprocessing/gfd-2011-national-grid")
EVENTS = {
    3850: "DFO_3850_From_20110805_to_20120109",
    3853: "DFO_3853_From_20110810_to_20111115",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def project_average(
    values: np.ndarray,
    source: rasterio.io.DatasetReader,
    shape: tuple[int, int],
    transform,
    resampling: Resampling = Resampling.average,
) -> np.ndarray:
    destination = np.full(shape, np.nan, dtype="float32")
    reproject(
        source=values.astype("float32", copy=False),
        destination=destination,
        src_transform=source.transform,
        src_crs=source.crs,
        src_nodata=np.nan,
        dst_transform=transform,
        dst_crs="EPSG:32648",
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return destination


def process_event(
    root: Path,
    event_id: int,
    event_name: str,
    grid: pd.DataFrame,
    shape: tuple[int, int],
    transform,
    raster_rows: np.ndarray,
    raster_columns: np.ndarray,
) -> pd.DataFrame:
    archive = root / "data/raw/flood/global_flood_database" / f"{event_name}.zip"
    virtual_path = f"/vsizip/{archive}/{event_name}.tif"
    with rasterio.open(virtual_path) as source:
        expected = ("flooded", "duration", "clear_views", "clear_perc", "jrc_perm_water")
        if source.descriptions != expected:
            raise RuntimeError(f"Unexpected GFD bands for event {event_id}: {source.descriptions}")
        arrays = source.read([1, 2, 3, 5], masked=True)
        flooded, duration, clear_views, permanent = np.ma.filled(arrays, np.nan).astype(
            "float32", copy=False
        )
        covered = np.isfinite(flooded)
        clear = covered & np.isfinite(clear_views) & (clear_views > 0)
        nonpermanent_flood = (
            covered
            & (flooded >= 0.5)
            & np.isfinite(permanent)
            & (permanent < 0.5)
        )
        duration_area_days = np.where(
            nonpermanent_flood & np.isfinite(duration), np.maximum(duration, 0), 0
        ).astype("float32")
        duration_for_max = np.where(nonpermanent_flood, duration_area_days, 0).astype("float32")

        coverage_grid = project_average(covered.astype("float32"), source, shape, transform)
        clear_grid = project_average(clear.astype("float32"), source, shape, transform)
        flood_grid = project_average(
            nonpermanent_flood.astype("float32"), source, shape, transform
        )
        duration_area_grid = project_average(duration_area_days, source, shape, transform)
        maximum_duration_grid = project_average(
            duration_for_max, source, shape, transform, Resampling.max
        )

    coverage = coverage_grid[raster_rows, raster_columns]
    clear_share = clear_grid[raster_rows, raster_columns]
    flood_share = flood_grid[raster_rows, raster_columns]
    duration_area = duration_area_grid[raster_rows, raster_columns]
    maximum_duration = maximum_duration_grid[raster_rows, raster_columns]
    # Reprojection of the binary masks produces zero outside the event's valid
    # raster footprint.  Preserve that zero in the explicit coverage field, but
    # do not let it masquerade as an observed dry cell in the analytical fields.
    outside_event_footprint = ~np.isfinite(coverage) | (coverage <= 0)
    clear_share = np.where(outside_event_footprint, np.nan, clear_share)
    flood_share = np.where(outside_event_footprint, np.nan, flood_share)
    duration_area = np.where(outside_event_footprint, np.nan, duration_area)
    maximum_duration = np.where(outside_event_footprint, np.nan, maximum_duration)
    mean_duration = np.divide(
        duration_area,
        flood_share,
        out=np.full_like(duration_area, np.nan),
        where=np.isfinite(flood_share) & (flood_share > 0),
    )
    stem = f"Event {event_id}"
    return pd.DataFrame(
        {
            "National Grid Cell ID": grid["National Grid Cell ID"].astype("string"),
            f"{stem} Raster Coverage Share": coverage,
            f"{stem} Clear Observation Share": clear_share,
            f"{stem} Flooded Share Excluding Permanent Water": flood_share,
            f"{stem} Mean Flood Duration Days among Flooded Area": mean_duration,
            f"{stem} Maximum Flood Duration Days": maximum_duration,
        }
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    grid = pd.read_parquet(root / GRID)
    min_column = int(grid["Grid Column"].min())
    max_column = int(grid["Grid Column"].max())
    min_row = int(grid["Grid Row"].min())
    max_row = int(grid["Grid Row"].max())
    shape = (max_row - min_row + 1, max_column - min_column + 1)
    transform = from_origin(min_column * 1_000, (max_row + 1) * 1_000, 1_000, 1_000)
    raster_rows = max_row - grid["Grid Row"].to_numpy(int)
    raster_columns = grid["Grid Column"].to_numpy(int) - min_column

    event_frames = []
    for event_id, event_name in EVENTS.items():
        print(f"Processing GFD event {event_id}", flush=True)
        event_frames.append(
            process_event(
                root,
                event_id,
                event_name,
                grid,
                shape,
                transform,
                raster_rows,
                raster_columns,
            )
        )
    output = event_frames[0]
    for event_frame in event_frames[1:]:
        output = output.merge(event_frame, on="National Grid Cell ID", validate="one_to_one")

    coverage_columns = [f"Event {event_id} Raster Coverage Share" for event_id in EVENTS]
    clear_columns = [f"Event {event_id} Clear Observation Share" for event_id in EVENTS]
    flood_columns = [
        f"Event {event_id} Flooded Share Excluding Permanent Water" for event_id in EVENTS
    ]
    max_duration_columns = [f"Event {event_id} Maximum Flood Duration Days" for event_id in EVENTS]
    output["2011 Mapped Flood Event Count"] = (
        output[coverage_columns].gt(0).sum(axis=1).astype("int8")
    )
    output["2011 Clear Observed Flood Event Count"] = (
        output[clear_columns].fillna(0).gt(0).sum(axis=1).astype("int8")
    )
    output["2011 Maximum Flooded Share Excluding Permanent Water"] = output[
        flood_columns
    ].max(axis=1, skipna=True)
    output["2011 Cumulative Flooded Share Excluding Permanent Water"] = output[
        flood_columns
    ].sum(axis=1, min_count=1)
    output["2011 Maximum Flood Duration Days"] = output[max_duration_columns].max(
        axis=1, skipna=True
    )
    output["2011 Minimum Clear Observation Share across Covered Events"] = output[
        clear_columns
    ].min(axis=1, skipna=True)
    # A dry classification requires at least one clear satellite observation;
    # mere inclusion in an event raster is not enough.
    observed = output["2011 Clear Observed Flood Event Count"].gt(0)
    any_flood = output["2011 Maximum Flooded Share Excluding Permanent Water"].gt(0).astype(
        "float32"
    )
    output["2011 Any Satellite Observed Flooding"] = any_flood.where(observed, np.nan)

    output_path = root / OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)

    audit_dir = root / AUDIT_DIR
    audit_dir.mkdir(parents=True, exist_ok=True)
    coverage_summary = []
    for event_id in EVENTS:
        coverage_summary.append(
            {
                "Event ID": event_id,
                "Grid Cells": len(output),
                "Raster Covered Cells": int(output[f"Event {event_id} Raster Coverage Share"].gt(0).sum()),
                "Clear Observed Cells": int(output[f"Event {event_id} Clear Observation Share"].gt(0).sum()),
                "Flooded Cells": int(
                    output[f"Event {event_id} Flooded Share Excluding Permanent Water"].gt(0).sum()
                ),
                "Mean Flooded Share among Flooded Cells": float(
                    output.loc[
                        output[f"Event {event_id} Flooded Share Excluding Permanent Water"].gt(0),
                        f"Event {event_id} Flooded Share Excluding Permanent Water",
                    ].mean()
                ),
            }
        )
    pd.DataFrame(coverage_summary).to_csv(audit_dir / "event_grid_coverage.csv", index=False)
    decisions = {
        "status": "candidate flood-confound experiment; AnaSOP update deferred",
        "source_events": EVENTS,
        "selected_variables": output.columns.tolist(),
        "readable_name_rule": "event IDs retained for source traceability; all analytical names are English and explicit about permanent-water exclusion",
        "spatial_aggregation": "area-average from approximately 250 m GFD pixels to the stable Cambodia EPSG:32648 1 km grid",
        "flood_definition": "flooded band >= 0.5 and JRC permanent-water band < 0.5",
        "missing_rule": "cells outside an event raster remain missing; missing cells are not coded as no flood",
        "combined_event_rule": "maximum share is primary to avoid double-counting overlapping events; cumulative share is diagnostic",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (audit_dir / "decisions.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metadata = {
        "rows": len(output),
        "grid_cells": output["National Grid Cell ID"].nunique(),
        "events": list(EVENTS),
        "output": str(OUTPUT),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (audit_dir / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved {OUTPUT}; rows={len(output):,}")
    print(pd.DataFrame(coverage_summary).to_string(index=False))


if __name__ == "__main__":
    main()
