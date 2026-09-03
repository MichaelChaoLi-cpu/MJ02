#!/usr/bin/env python3
"""Run the pre-specified fixed-cropland support and blinded power gate.

The script never exports historical-side response coefficients.  It first
freezes two early-period land-cover masks, checks spatial support, constructs
mask-specific EVI anomalies, and releases only coverage plus the standard error
and 80% minimum detectable effect for the immediate dry-response contrast.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as transform_coordinates
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analyses"))
from _high_frequency_recovery_models import fit_distributed_lag, linear_contrast  # noqa: E402


LAND_COVER_MANIFEST = ROOT / "data/exp/feasibility-check/fixed-cropland/source/source_manifest.csv"
VEGETATION_MANIFEST = ROOT / "data/exp/data-preprocessing/dynamic-vegetation-source/modis-13Q1/source_manifest.csv"
BASE_PANEL = ROOT / "data/processed/historical_boundary_16day_climate_vegetation_preprocessed.parquet"
OUTPUT_DIR = ROOT / "data/exp/feasibility-check/fixed-cropland"

EVI_ASSET = "250m_16_days_EVI"
RELIABILITY_ASSET = "250m_16_days_pixel_reliability"
OUTCOME = "High-Frequency Cropland EVI Anomaly Z"
BUFFER_METRES = 1000.0
MIN_MASKED_PIXELS = 4
MIN_VALID_SHARE = 0.50
MDE_FACTOR = 1.959964 + 0.841621
MASKS = {
    "strict_cropland": {12},
    "inclusive_agriculture": {12, 14},
}
THRESHOLDS = {
    "cross_side_cells": 6,
    "cross_side_villages": 120,
    "villages_each_side": 50,
    "composite_dates": 400,
    "observed_outcome_share": 0.55,
    "blinded_80pct_mde_sd": 0.20,
}


def pixel_centres(dataset: rasterio.io.DatasetReader) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices((dataset.height, dataset.width))
    x, y = rasterio.transform.xy(dataset.transform, rows.ravel(), cols.ravel(), offset="center")
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def land_cover_masks(reference: rasterio.io.DatasetReader) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    x, y = pixel_centres(reference)
    longitude, latitude = transform_coordinates(
        reference.crs, "EPSG:4326", x.tolist(), y.tolist()
    )
    manifest = pd.read_csv(LAND_COVER_MANIFEST).sort_values("year")
    if tuple(manifest["year"].astype(int)) != (2001, 2002, 2003):
        raise ValueError("Land-cover manifest must contain exactly 2001-2003")
    classifications = []
    for row in manifest.itertuples(index=False):
        path = ROOT / str(row.local_path)
        with rasterio.open(path) as dataset:
            values = np.fromiter(
                (sample[0] for sample in dataset.sample(zip(longitude, latitude))),
                dtype=np.uint8,
                count=len(x),
            )
        classifications.append(values)
    annual = np.vstack(classifications)
    masks = {
        name: np.isin(annual, list(classes)).sum(axis=0) >= 2
        for name, classes in MASKS.items()
    }
    pixels = pd.DataFrame(
        {
            "modis_pixel_index": np.arange(len(x)),
            "x": x,
            "y": y,
            "longitude": longitude,
            "latitude": latitude,
            "lc_type1_2001": annual[0],
            "lc_type1_2002": annual[1],
            "lc_type1_2003": annual[2],
            **{name: mask.astype(np.int8) for name, mask in masks.items()},
        }
    )
    return pixels, masks


def village_mapping(
    villages: pd.DataFrame,
    reference: rasterio.io.DatasetReader,
) -> list[np.ndarray]:
    x, y = pixel_centres(reference)
    tree = cKDTree(np.column_stack([x, y]))
    village_x, village_y = transform_coordinates(
        "EPSG:4326",
        reference.crs,
        villages["Longitude"].to_numpy(float).tolist(),
        villages["Latitude"].to_numpy(float).tolist(),
    )
    return [
        np.asarray(indices, dtype=int)
        for indices in tree.query_ball_point(
            np.column_stack([village_x, village_y]), r=BUFFER_METRES
        )
    ]


def masked_mapping(mapping: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
    return [indices[mask[indices]] for indices in mapping]


def read_flat(path: Path, profile: tuple[object, ...]) -> np.ndarray:
    with rasterio.open(path) as dataset:
        observed = (dataset.crs.to_wkt(), dataset.transform, dataset.width, dataset.height)
        if observed != profile:
            raise ValueError(f"Raster grid differs from frozen MOD13Q1 reference: {path}")
        return dataset.read(1).ravel()


def immediate_dry_weights() -> dict[str, float]:
    return {f"Southwest x Dry k={lag}": 1.0 / 3.0 for lag in (0, 1, 2)}


def add_anomalies(panel: pd.DataFrame) -> pd.DataFrame:
    reference = panel["Year"].between(2001, 2020)
    baseline = (
        panel.loc[reference]
        .groupby(["Village Code", "MODIS Composite Slot"], observed=True)["Mean Cropland EVI"]
        .agg(["mean", "std", "count"])
        .rename(
            columns={
                "mean": "Cropland EVI Slot Mean 2001-2020",
                "std": "Cropland EVI Slot SD 2001-2020",
                "count": "Cropland EVI Slot Valid Years 2001-2020",
            }
        )
        .reset_index()
    )
    panel = panel.merge(
        baseline,
        on=["Village Code", "MODIS Composite Slot"],
        how="left",
        validate="many_to_one",
    )
    panel[OUTCOME] = (
        panel["Mean Cropland EVI"] - panel["Cropland EVI Slot Mean 2001-2020"]
    ) / panel["Cropland EVI Slot SD 2001-2020"]
    invalid = (
        panel["Cropland EVI Slot Valid Years 2001-2020"].lt(10)
        | panel["Cropland EVI Slot SD 2001-2020"].le(0)
    )
    panel.loc[invalid, OUTCOME] = np.nan
    return panel


def support_rows(
    villages: pd.DataFrame,
    mapping_by_mask: dict[str, list[np.ndarray]],
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    rows = []
    cross_side_cells: dict[str, set[str]] = {}
    for name, mapping in mapping_by_mask.items():
        counts = np.asarray([len(indices) for indices in mapping])
        eligible = villages.loc[counts >= MIN_MASKED_PIXELS].copy()
        cell_sides = eligible.groupby("CHIRPS Cell ID", observed=True)[
            "Higher-Repression Southwest Zone"
        ].nunique()
        cells = set(cell_sides.loc[cell_sides.eq(2)].index.astype(str))
        cross_side_cells[name] = cells
        cross = eligible.loc[eligible["CHIRPS Cell ID"].astype(str).isin(cells)]
        sides = cross.groupby("Higher-Repression Southwest Zone", observed=True)[
            "Village Code"
        ].nunique()
        rows.append(
            {
                "mask": name,
                "eligible_villages": int(eligible["Village Code"].nunique()),
                "eligible_villages_west": int(
                    eligible.loc[eligible["Higher-Repression Southwest Zone"].eq(0), "Village Code"].nunique()
                ),
                "eligible_villages_southwest": int(
                    eligible.loc[eligible["Higher-Repression Southwest Zone"].eq(1), "Village Code"].nunique()
                ),
                "cross_side_cells": len(cells),
                "cross_side_villages": int(cross["Village Code"].nunique()),
                "cross_side_villages_west": int(sides.get(0, 0)),
                "cross_side_villages_southwest": int(sides.get(1, 0)),
                "median_masked_pixels": float(np.median(counts)),
                "p10_masked_pixels": float(np.quantile(counts, 0.10)),
                "p90_masked_pixels": float(np.quantile(counts, 0.90)),
            }
        )
    return pd.DataFrame(rows), cross_side_cells


def aggregate_outcomes(
    grouped_manifest: list[tuple[pd.Timestamp, pd.DataFrame]],
    villages: pd.DataFrame,
    mapping_by_mask: dict[str, list[np.ndarray]],
    profile: tuple[object, ...],
) -> dict[str, pd.DataFrame]:
    records: dict[str, list[dict[str, object]]] = {name: [] for name in MASKS}
    village_codes = villages["Village Code"].astype(str).to_numpy()
    for index, (date, date_rows) in enumerate(grouped_manifest, start=1):
        paths = {row.asset: ROOT / str(row.local_path) for row in date_rows.itertuples(index=False)}
        evi = read_flat(paths[EVI_ASSET], profile).astype(float)
        reliability = read_flat(paths[RELIABILITY_ASSET], profile)
        quality = np.isin(reliability, [0, 1]) & (evi >= -2000) & (evi <= 10000)
        for name, mapping in mapping_by_mask.items():
            for village_code, indices in zip(village_codes, mapping):
                candidate = len(indices)
                if candidate < MIN_MASKED_PIXELS:
                    continue
                required = max(MIN_MASKED_PIXELS, int(np.ceil(MIN_VALID_SHARE * candidate)))
                valid_indices = indices[quality[indices]]
                value = float(np.mean(evi[valid_indices]) * 0.0001) if len(valid_indices) >= required else np.nan
                records[name].append(
                    {
                        "Village Code": village_code,
                        "Composite Date": date,
                        "Year": date.year,
                        "MODIS Composite Slot": int((date.dayofyear - 1) // 16 + 1),
                        "Fixed Cropland Pixel Count": candidate,
                        "Valid Cropland EVI Pixel Count": len(valid_indices),
                        "Required Cropland EVI Pixel Count": required,
                        "Mean Cropland EVI": value,
                    }
                )
        if index % 50 == 0 or index == len(grouped_manifest):
            print(f"Aggregated masked EVI for {index}/{len(grouped_manifest)} composites")
    return {name: add_anomalies(pd.DataFrame(rows)) for name, rows in records.items()}


def gate_row(diagnostic: str, value: float, threshold: float, direction: str) -> dict[str, object]:
    passed = value >= threshold if direction == ">=" else value <= threshold
    return {
        "diagnostic": diagnostic,
        "value": value,
        "threshold": threshold,
        "direction": direction,
        "pass": bool(passed),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(BASE_PANEL)
    base["Village Code"] = base["Village Code"].astype(str)
    base["CHIRPS Cell ID"] = base["CHIRPS Cell ID"].astype(str)
    base["Composite Date"] = pd.to_datetime(base["Composite Date"])
    villages = base.drop_duplicates("Village Code").copy()

    vegetation = pd.read_csv(VEGETATION_MANIFEST)
    vegetation["composite_date"] = pd.to_datetime(vegetation["composite_date"])
    vegetation = vegetation.loc[vegetation["asset"].isin([EVI_ASSET, RELIABILITY_ASSET])].copy()
    vegetation["local_path"] = vegetation["local_path"].astype(str)
    grouped = list(vegetation.groupby("composite_date", observed=True, sort=True))
    if not grouped:
        raise ValueError("No MOD13Q1 EVI source clips found")
    first = ROOT / str(grouped[0][1].iloc[0]["local_path"])
    with rasterio.open(first) as reference:
        profile = (reference.crs.to_wkt(), reference.transform, reference.width, reference.height)
        pixels, masks = land_cover_masks(reference)
        mapping = village_mapping(villages, reference)
    pixels.to_parquet(OUTPUT_DIR / "fixed_mask_on_mod13q1_grid.parquet", index=False)
    mapping_by_mask = {name: masked_mapping(mapping, mask) for name, mask in masks.items()}
    support, cells_by_mask = support_rows(villages, mapping_by_mask)
    support.to_csv(OUTPUT_DIR / "spatial_support.csv", index=False)

    outcomes = aggregate_outcomes(grouped, villages, mapping_by_mask, profile)
    all_gate_rows = []
    power_rows = []
    coverage_rows = []
    for name, outcome_panel in outcomes.items():
        design_columns = [
            "Village Code", "Composite Date", "CHIRPS Cell ID",
            "Interval-Aligned Rainfall Anomaly Z",
            "Higher-Repression Southwest Zone",
            "Signed Distance to Historical Repression Boundary km",
            "Linked Climate Commune Code",
        ]
        analysis = outcome_panel.merge(
            base[design_columns],
            on=["Village Code", "Composite Date"],
            how="inner",
            validate="one_to_one",
        )
        analysis["Cross-Side CHIRPS Cell"] = analysis["CHIRPS Cell ID"].isin(
            cells_by_mask[name]
        ).astype(np.int8)
        sample = analysis.loc[analysis["Cross-Side CHIRPS Cell"].eq(1)].copy()
        support_row = support.loc[support["mask"].eq(name)].iloc[0]
        available_dates = int(sample.loc[sample[OUTCOME].notna(), "Composite Date"].nunique())
        observed_share = float(sample[OUTCOME].notna().mean())
        prefit_rows = [
            gate_row("Cross-side CHIRPS cells", support_row["cross_side_cells"], THRESHOLDS["cross_side_cells"], ">="),
            gate_row("Villages on cross-side cells", support_row["cross_side_villages"], THRESHOLDS["cross_side_villages"], ">="),
            gate_row("West villages on cross-side cells", support_row["cross_side_villages_west"], THRESHOLDS["villages_each_side"], ">="),
            gate_row("Southwest villages on cross-side cells", support_row["cross_side_villages_southwest"], THRESHOLDS["villages_each_side"], ">="),
            gate_row("Composite dates with observed outcome", available_dates, THRESHOLDS["composite_dates"], ">="),
            gate_row("Observed cropland EVI row share", observed_share, THRESHOLDS["observed_outcome_share"], ">="),
        ]
        coverage_rows.append(
            {
                "mask": name,
                "rows_on_cross_side_cells": len(sample),
                "observed_outcome_rows": int(sample[OUTCOME].notna().sum()),
                "observed_outcome_share": observed_share,
                "composite_dates_with_observed_outcome": available_dates,
                "median_fixed_cropland_pixels": float(sample["Fixed Cropland Pixel Count"].median()),
            }
        )
        if all(row["pass"] for row in prefit_rows):
            fit = fit_distributed_lag(analysis, OUTCOME, "Primary")
            contrast = linear_contrast(fit, immediate_dry_weights())
            standard_error = float(contrast["standard_error"])
            mde = MDE_FACTOR * standard_error
            power_rows.append(
                {
                    "mask": name,
                    "contrast": "Immediate average dry response, lags 0-2",
                    "standard_error_only": standard_error,
                    "blinded_80pct_mde_sd": mde,
                    "threshold_sd": THRESHOLDS["blinded_80pct_mde_sd"],
                    "pass": mde <= THRESHOLDS["blinded_80pct_mde_sd"],
                    "model_observations": fit.observations,
                    "model_composite_dates": fit.events,
                    "model_villages": fit.villages,
                    "model_groups": fit.groups,
                    "historical_side_estimate_released": False,
                }
            )
            prefit_rows.append(
                gate_row(
                    "Blinded 80% MDE for immediate dry response (SD)",
                    mde,
                    THRESHOLDS["blinded_80pct_mde_sd"],
                    "<=",
                )
            )
        else:
            power_rows.append(
                {
                    "mask": name,
                    "contrast": "Immediate average dry response, lags 0-2",
                    "standard_error_only": np.nan,
                    "blinded_80pct_mde_sd": np.nan,
                    "threshold_sd": THRESHOLDS["blinded_80pct_mde_sd"],
                    "pass": False,
                    "model_observations": np.nan,
                    "model_composite_dates": np.nan,
                    "model_villages": np.nan,
                    "model_groups": np.nan,
                    "historical_side_estimate_released": False,
                }
            )
        for row in prefit_rows:
            all_gate_rows.append({"mask": name, **row})

    gate = pd.DataFrame(all_gate_rows)
    power = pd.DataFrame(power_rows)
    coverage = pd.DataFrame(coverage_rows)
    gate.to_csv(OUTPUT_DIR / "blinded_activation_gate.csv", index=False)
    power.to_csv(OUTPUT_DIR / "blinded_power.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "outcome_coverage.csv", index=False)
    status = {
        name: bool(group["pass"].all())
        for name, group in gate.groupby("mask", observed=True)
    }
    readme = {
        "purpose": "Outcome-blind feasibility and power gate for a fixed early-period cropland EVI analysis",
        "primary_mask": "IGBP class 12 in at least two of 2001-2003",
        "prespecified_secondary_mask": "IGBP class 12 or 14 in at least two of 2001-2003",
        "mask_choice_is_outcome_adaptive": False,
        "minimum_masked_mod13q1_pixels_per_village": MIN_MASKED_PIXELS,
        "minimum_valid_pixel_share_per_date": MIN_VALID_SHARE,
        "historical_side_coefficients_saved": False,
        "gate_status": status,
        "next_step_if_primary_passes": "Materialize the fixed-cropland panel through data-preprocessing, then estimate the pre-specified response contrasts.",
        "next_step_if_primary_fails": "Do not estimate or report cropland historical-side response coefficients.",
    }
    (OUTPUT_DIR / "README_fixed_cropland_gate.json").write_text(
        json.dumps(readme, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nSpatial support")
    print(support.to_string(index=False))
    print("\nOutcome coverage")
    print(coverage.to_string(index=False))
    print("\nBlinded activation gate")
    print(gate.to_string(index=False))
    print("\nBlinded power")
    print(power.to_string(index=False))
    print(f"\nGate status: {status}")


if __name__ == "__main__":
    main()
