"""Construct satellite-observed flood exposure from Global Flood Database maps.

Permanent water is excluded from the flooded numerator. Flood area is divided
by the full mapped geography area, while raster coverage and clear-observation
quality are retained explicitly. The event database ends in 2018, so later
survey waves remain missing rather than being coded as no flood.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds


EVENT_FIELDS = [
    "Event ID",
    "Event Name",
    "Event Start Date",
    "Event End Date",
    "Main Cause",
    "Severity",
    "Validation Type",
]
SUM_FIELDS = [
    "Raster Covered Area km2",
    "Clear Observed Area km2",
    "Flooded Area Excluding Permanent Water km2",
    "Flood Duration Area Days km2",
    "Clear Percent Area Weighted",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args()


def load_geographies(root: Path) -> tuple[gpd.GeoDataFrame, dict[str, pd.DataFrame]]:
    boundaries = gpd.read_file(
        root / "data/raw/geography/odc_cambodia_communes_2014.gpkg"
    )
    projected = boundaries.to_crs(32648)
    communes = boundaries.to_crs(4326).copy()
    communes["Geography Code"] = communes["com_code"].astype(int).astype(str).str.zfill(6)
    communes["Geography Name"] = communes["com_name"]
    communes["District Code"] = communes["Geography Code"].str[:4]
    communes["Province Code"] = communes["Geography Code"].str[:2]
    communes["Geography Area km2"] = projected.geometry.area.to_numpy() / 1_000_000
    communes = communes.reset_index(drop=True)
    communes["Geography Index"] = np.arange(len(communes), dtype=np.int32)

    names = gpd.read_file(
        root / "data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"
    )
    district_names = (
        names.assign(code=names["ADM2_PCODE"].str.replace("KH", "", regex=False))
        .drop_duplicates("code")
        .set_index("code")["ADM2_EN"]
        .to_dict()
    )
    province_names = (
        names.assign(code=names["ADM1_PCODE"].str.replace("KH", "", regex=False))
        .drop_duplicates("code")
        .set_index("code")["ADM1_EN"]
        .to_dict()
    )
    lookup = {
        "commune": communes[
            [
                "Geography Index",
                "Geography Code",
                "Geography Name",
                "District Code",
                "Province Code",
                "Geography Area km2",
            ]
        ].copy(),
        "district": (
            communes.groupby("District Code", as_index=False)
            .agg(**{"Geography Area km2": ("Geography Area km2", "sum")})
            .rename(columns={"District Code": "Geography Code"})
            .assign(Geography_Name=lambda x: x["Geography Code"].map(district_names))
            .rename(columns={"Geography_Name": "Geography Name"})
        ),
        "province": (
            communes.groupby("Province Code", as_index=False)
            .agg(**{"Geography Area km2": ("Geography Area km2", "sum")})
            .rename(columns={"Province Code": "Geography Code"})
            .assign(Geography_Name=lambda x: x["Geography Code"].map(province_names))
            .rename(columns={"Geography_Name": "Geography Name"})
        ),
    }
    return communes, lookup


def weighted_bincount(
    geography_index: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    geography_count: int,
) -> np.ndarray:
    return np.bincount(
        geography_index[mask], weights=weights[mask], minlength=geography_count
    )


def process_event(
    archive_path: Path,
    manifest_row: pd.Series,
    communes: gpd.GeoDataFrame,
) -> pd.DataFrame:
    event_name = str(manifest_row["Event Name"])
    tif_name = f"{event_name}.tif"
    virtual_path = f"/vsizip/{archive_path}/{tif_name}"
    geography_count = len(communes)

    with rasterio.open(virtual_path) as source:
        expected = ("flooded", "duration", "clear_views", "clear_perc", "jrc_perm_water")
        if source.descriptions != expected:
            raise ValueError(
                f"Unexpected bands in {archive_path.name}: {source.descriptions}"
            )
        bounds = communes.total_bounds
        raw_window = from_bounds(*bounds, transform=source.transform)
        full_window = Window(0, 0, source.width, source.height)
        try:
            window = raw_window.intersection(full_window).round_offsets().round_lengths()
        except rasterio.errors.WindowError:
            window = None

        output = communes[
            [
                "Geography Index",
                "Geography Code",
                "Geography Name",
                "District Code",
                "Province Code",
                "Geography Area km2",
            ]
        ].copy()
        for column in SUM_FIELDS:
            output[column] = 0.0
        output["Maximum Flood Duration Days"] = 0.0

        if window is not None and window.width > 0 and window.height > 0:
            arrays = source.read([1, 2, 3, 4, 5], window=window, masked=True)
            values = np.ma.filled(arrays, np.nan).astype("float64", copy=False)
            flooded, duration, clear_views, clear_percent, permanent = values
            transform = source.window_transform(window)
            geography = rasterize(
                (
                    (geometry, int(index))
                    for geometry, index in zip(
                        communes.geometry, communes["Geography Index"]
                    )
                ),
                out_shape=(int(window.height), int(window.width)),
                transform=transform,
                fill=-1,
                dtype="int32",
                all_touched=False,
            )

            row_latitudes = transform.f + (np.arange(geography.shape[0]) + 0.5) * transform.e
            pixel_area_by_row = (
                111.32
                * abs(transform.a)
                * np.cos(np.deg2rad(row_latitudes))
                * 110.574
                * abs(transform.e)
            )
            pixel_area = np.broadcast_to(pixel_area_by_row[:, None], geography.shape).ravel()
            geography_flat = geography.ravel()
            inside = geography_flat >= 0
            finite = np.isfinite(flooded.ravel())
            covered = inside & finite
            clear_observed = covered & np.isfinite(clear_views.ravel()) & (clear_views.ravel() > 0)
            nonpermanent_flood = (
                covered
                & (flooded.ravel() >= 0.5)
                & np.isfinite(permanent.ravel())
                & (permanent.ravel() < 0.5)
            )
            duration_values = np.where(
                nonpermanent_flood & np.isfinite(duration.ravel()),
                np.maximum(duration.ravel(), 0),
                0,
            )
            clear_values = clear_percent.ravel().copy()
            finite_clear = covered & np.isfinite(clear_values)
            if finite_clear.any() and np.nanmax(clear_values[finite_clear]) <= 1.5:
                clear_values = 100 * clear_values

            output["Raster Covered Area km2"] = weighted_bincount(
                geography_flat, covered, pixel_area, geography_count
            )
            output["Clear Observed Area km2"] = weighted_bincount(
                geography_flat, clear_observed, pixel_area, geography_count
            )
            output["Flooded Area Excluding Permanent Water km2"] = weighted_bincount(
                geography_flat, nonpermanent_flood, pixel_area, geography_count
            )
            output["Flood Duration Area Days km2"] = weighted_bincount(
                geography_flat,
                nonpermanent_flood,
                pixel_area * duration_values,
                geography_count,
            )
            output["Clear Percent Area Weighted"] = weighted_bincount(
                geography_flat,
                finite_clear,
                pixel_area * np.where(np.isfinite(clear_values), clear_values, 0),
                geography_count,
            )
            maximum_duration = np.zeros(geography_count, dtype=float)
            np.maximum.at(
                maximum_duration,
                geography_flat[nonpermanent_flood],
                duration_values[nonpermanent_flood],
            )
            output["Maximum Flood Duration Days"] = maximum_duration

    for field in EVENT_FIELDS:
        output[field] = manifest_row[field]
    return output


def finalize_metrics(data: pd.DataFrame, resolution: str) -> pd.DataFrame:
    output = data.copy()
    output["Geography Resolution"] = resolution
    output["Event Raster Coverage Share"] = (
        output["Raster Covered Area km2"] / output["Geography Area km2"]
    ).clip(upper=1)
    output["Clear Observed Geography Share"] = (
        output["Clear Observed Area km2"] / output["Geography Area km2"]
    ).clip(upper=1)
    output["Flooded Geography Share"] = (
        output["Flooded Area Excluding Permanent Water km2"]
        / output["Geography Area km2"]
    ).clip(upper=1)
    output["Any Satellite Observed Flooding"] = output["Flooded Geography Share"].gt(0)
    output["Mean Flood Duration Days over Geography"] = (
        output["Flood Duration Area Days km2"] / output["Geography Area km2"]
    )
    output["Mean Flood Duration Days among Flooded Area"] = (
        output["Flood Duration Area Days km2"]
        / output["Flooded Area Excluding Permanent Water km2"].replace(0, np.nan)
    )
    output["Mean Clear Observation Percent"] = (
        output["Clear Percent Area Weighted"]
        / output["Raster Covered Area km2"].replace(0, np.nan)
    )
    keep = [
        "Geography Resolution",
        "Geography Code",
        "Geography Name",
        "Geography Area km2",
    ] + EVENT_FIELDS + [
        "Raster Covered Area km2",
        "Event Raster Coverage Share",
        "Clear Observed Geography Share",
        "Mean Clear Observation Percent",
        "Flooded Area Excluding Permanent Water km2",
        "Flooded Geography Share",
        "Any Satellite Observed Flooding",
        "Mean Flood Duration Days over Geography",
        "Mean Flood Duration Days among Flooded Area",
        "Maximum Flood Duration Days",
    ]
    return output[keep].sort_values(["Event Start Date", "Geography Code"])


def aggregate_resolution(
    commune_events: pd.DataFrame,
    lookup: pd.DataFrame,
    code_column: str,
    resolution: str,
) -> pd.DataFrame:
    grouped = (
        commune_events.groupby(EVENT_FIELDS + [code_column], as_index=False, dropna=False)
        .agg(
            **{
                **{field: (field, "sum") for field in SUM_FIELDS},
                "Maximum Flood Duration Days": ("Maximum Flood Duration Days", "max"),
            }
        )
        .rename(columns={code_column: "Geography Code"})
    )
    grouped = grouped.merge(
        lookup[["Geography Code", "Geography Name", "Geography Area km2"]],
        on="Geography Code",
        how="left",
        validate="many_to_one",
    )
    return finalize_metrics(grouped, resolution)


def summarize_window(events: pd.DataFrame) -> dict[str, object]:
    if events.empty:
        return {
            "Mapped Flood Event Count": 0,
            "Local Inundation Event Count": 0,
            "Maximum Flooded Geography Share": 0.0,
            "Cumulative Flooded Geography Share": 0.0,
            "Maximum Flood Duration Days": 0.0,
            "Any Satellite Observed Flooding": False,
            "Minimum Event Raster Coverage Share": 0.0,
            "Minimum Clear Observed Geography Share": 0.0,
        }
    return {
        "Mapped Flood Event Count": len(events),
        "Local Inundation Event Count": int(events["Any Satellite Observed Flooding"].sum()),
        "Maximum Flooded Geography Share": events["Flooded Geography Share"].max(),
        "Cumulative Flooded Geography Share": events["Flooded Geography Share"].sum(),
        "Maximum Flood Duration Days": events["Maximum Flood Duration Days"].max(),
        "Any Satellite Observed Flooding": bool(
            events["Any Satellite Observed Flooding"].any()
        ),
        "Minimum Event Raster Coverage Share": events["Event Raster Coverage Share"].min(),
        "Minimum Clear Observed Geography Share": events[
            "Clear Observed Geography Share"
        ].min(),
    }


def attach_to_psu(
    crosswalk: pd.DataFrame,
    event_panels: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    climate = pd.read_parquet(crosswalk.attrs["climate_path"])[
        ["Survey Year", "Survey Wave", "PSU", "Survey Month Numeric"]
    ]
    base = crosswalk[
        [
            "Survey Year",
            "Survey Wave",
            "PSU",
            "Climate Geography Resolution",
            "Climate Geography Code",
            "Climate Link Method",
        ]
    ].merge(
        climate,
        on=["Survey Year", "Survey Wave", "PSU"],
        how="left",
        validate="one_to_one",
    )
    panel_lookup = {
        resolution: {
            code: group.sort_values("Event Start Date")
            for code, group in panel.groupby("Geography Code")
        }
        for resolution, panel in event_panels.items()
    }
    rows: list[dict[str, object]] = []
    for record in base.to_dict("records"):
        resolution = str(record["Climate Geography Resolution"])
        code = str(record["Climate Geography Code"])
        events = panel_lookup[resolution][code]
        year = int(record["Survey Year"])
        year_start = pd.Timestamp(year=year, month=1, day=1)
        year_end = pd.Timestamp(year=year, month=12, day=31)
        year_covered = 2007 <= year <= 2018
        row = dict(record)
        row["Survey Year Satellite Flood Coverage"] = year_covered
        if year_covered:
            selected = events.loc[
                events["Event Start Date"].le(year_end)
                & events["Event End Date"].ge(year_start)
            ]
            row.update(
                {
                    f"Survey Year {key}": value
                    for key, value in summarize_window(selected).items()
                }
            )
        else:
            for key in summarize_window(events.iloc[0:0]):
                row[f"Survey Year {key}"] = pd.NA

        month = record["Survey Month Numeric"]
        survey_date = (
            pd.Timestamp(year=year, month=int(month), day=15)
            if pd.notna(month)
            else pd.NaT
        )
        window_start = survey_date - pd.Timedelta(days=365) if pd.notna(survey_date) else pd.NaT
        preceding_covered = bool(
            pd.notna(survey_date)
            and window_start >= pd.Timestamp("2007-01-01")
            and survey_date <= pd.Timestamp("2018-12-10")
        )
        row["Preceding 12 Month Satellite Flood Coverage"] = preceding_covered
        if preceding_covered:
            selected = events.loc[
                events["Event Start Date"].le(survey_date)
                & events["Event End Date"].ge(window_start)
            ]
            row.update(
                {
                    f"Preceding 12 Month {key}": value
                    for key, value in summarize_window(selected).items()
                }
            )
        else:
            for key in summarize_window(events.iloc[0:0]):
                row[f"Preceding 12 Month {key}"] = pd.NA
        rows.append(row)
    output = pd.DataFrame(rows)
    bool_columns = [column for column in output if column.endswith("Coverage")]
    bool_columns += [
        column for column in output if column.endswith("Any Satellite Observed Flooding")
    ]
    for column in bool_columns:
        output[column] = output[column].astype("boolean")
    return output.sort_values(["Survey Year", "PSU"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    processed = root / "data/processed"
    diagnostics = root / "data/exp/data-preprocessing"
    flood_raw = root / "data/raw/flood/global_flood_database"
    manifest = pd.read_csv(flood_raw / "gfd_cambodia_event_manifest.csv")
    manifest["Event Start Date"] = pd.to_datetime(manifest["Event Start Date"])
    manifest["Event End Date"] = pd.to_datetime(manifest["Event End Date"])
    communes, lookup = load_geographies(root)

    event_outputs = []
    for index, row in manifest.iterrows():
        archive = root / row["Local Path"]
        event = process_event(archive, row, communes)
        event_outputs.append(event)
        print(
            f"[{index + 1:02d}/{len(manifest):02d}] {row['Event Name']}: "
            f"flooded communes={int((event['Flooded Area Excluding Permanent Water km2'] > 0).sum()):,}",
            flush=True,
        )
    commune_raw = pd.concat(event_outputs, ignore_index=True)
    commune_panel = finalize_metrics(commune_raw, "commune")
    district_panel = aggregate_resolution(
        commune_raw, lookup["district"], "District Code", "district"
    )
    province_panel = aggregate_resolution(
        commune_raw, lookup["province"], "Province Code", "province"
    )
    event_panels = {
        "commune": commune_panel,
        "district": district_panel,
        "province": province_panel,
    }
    for resolution, panel in event_panels.items():
        path = processed / f"global_flood_database_{resolution}_event_preprocessed.parquet"
        panel.to_parquet(path, index=False)
        print(f"wrote {path.name}: {panel.shape}")

    crosswalk = pd.read_parquet(
        processed / "direction3_historical_geography_crosswalk_preprocessed.parquet"
    )
    crosswalk.attrs["climate_path"] = str(
        processed / "direction3_psu_climate_shocks_preprocessed.parquet"
    )
    psu = attach_to_psu(crosswalk, event_panels)
    psu_path = processed / "direction3_psu_satellite_flood_shocks_preprocessed.parquet"
    psu.to_parquet(psu_path, index=False)

    coverage = (
        psu.groupby("Survey Year", as_index=False)
        .agg(
            **{
                "PSU Wave Rows": ("PSU", "size"),
                "Survey Year Flood Coverage Rows": (
                    "Survey Year Satellite Flood Coverage",
                    "sum",
                ),
                "Preceding 12 Month Flood Coverage Rows": (
                    "Preceding 12 Month Satellite Flood Coverage",
                    "sum",
                ),
                "Survey Year Any Local Inundation Rows": (
                    "Survey Year Any Satellite Observed Flooding",
                    "sum",
                ),
                "Preceding 12 Month Any Local Inundation Rows": (
                    "Preceding 12 Month Any Satellite Observed Flooding",
                    "sum",
                ),
            }
        )
    )
    coverage.to_csv(
        diagnostics / "direction3_satellite_flood_coverage.csv", index=False
    )
    validation = pd.DataFrame(
        [
            {"Check": "Source flood events", "Value": len(manifest)},
            {"Check": "Commune-event rows", "Value": len(commune_panel)},
            {"Check": "District-event rows", "Value": len(district_panel)},
            {"Check": "Province-event rows", "Value": len(province_panel)},
            {"Check": "PSU-wave rows", "Value": len(psu)},
            {
                "Check": "Duplicate PSU-wave rows",
                "Value": int(psu.duplicated(["Survey Year", "Survey Wave", "PSU"]).sum()),
            },
            {
                "Check": "Maximum flooded geography share",
                "Value": commune_panel["Flooded Geography Share"].max(),
            },
            {
                "Check": "Maximum clear observation percent",
                "Value": commune_panel["Mean Clear Observation Percent"].max(),
            },
        ]
    )
    validation.to_csv(
        diagnostics / "direction3_satellite_flood_validation.csv", index=False
    )
    print(f"wrote {psu_path.name}: {psu.shape}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
