"""Construct long-baseline CHIRPS drought and extreme-wet shock measures.

Rainfall is extracted at commune, district, and province levels from the
1981-2021 Cambodia subset. SPI-3, SPI-6, and SPI-12 use gamma distributions fit
separately by geography and ending calendar month over the 1991-2020 climate
normal. Annual and May-October extremes use the same fixed baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from scipy.stats import gamma, norm


BASELINE_START = 1991
BASELINE_END = 2020
SPI_WINDOWS = (3, 6, 12)


def grid_cell_indices(
    geometry: object, latitudes: np.ndarray, longitudes: np.ndarray
) -> tuple[np.ndarray, str]:
    minx, miny, maxx, maxy = geometry.bounds
    lat_idx = np.flatnonzero((latitudes >= miny) & (latitudes <= maxy))
    lon_idx = np.flatnonzero((longitudes >= minx) & (longitudes <= maxx))
    if len(lat_idx) and len(lon_idx):
        lat_grid, lon_grid = np.meshgrid(lat_idx, lon_idx, indexing="ij")
        ys = latitudes[lat_grid.ravel()]
        xs = longitudes[lon_grid.ravel()]
        inside = shapely.contains_xy(geometry, xs, ys) | shapely.intersects_xy(
            geometry, xs, ys
        )
        if inside.any():
            flat = lat_grid.ravel()[inside] * len(longitudes) + lon_grid.ravel()[inside]
            return flat.astype(int), "Grid Centers within Polygon"
    point = geometry.representative_point()
    lat_nearest = int(np.abs(latitudes - point.y).argmin())
    lon_nearest = int(np.abs(longitudes - point.x).argmin())
    return (
        np.array([lat_nearest * len(longitudes) + lon_nearest]),
        "Nearest Grid Center to Representative Point",
    )


def extract_monthly(
    geographies: gpd.GeoDataFrame,
    code_column: str,
    name_column: str,
    times: pd.DatetimeIndex,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    precip_flat: np.ndarray,
) -> pd.DataFrame:
    frames = []
    for row in geographies.itertuples(index=False):
        code = str(getattr(row, code_column)).replace("KH", "")
        indices, method = grid_cell_indices(row.geometry, latitudes, longitudes)
        values = np.nanmean(precip_flat[:, indices], axis=1)
        frames.append(
            pd.DataFrame(
                {
                    "Climate Geography Code": code,
                    "Climate Geography Name": getattr(row, name_column),
                    "Date": times,
                    "Year": times.year,
                    "Month": times.month,
                    "Rainfall mm": values,
                    "Climate Grid Cell Count": len(indices),
                    "Climate Extraction Method": method,
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(
        ["Climate Geography Code", "Date"]
    ).reset_index(drop=True)


def fit_gamma_spi(baseline: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, str]:
    baseline = baseline[np.isfinite(baseline)]
    positive = baseline[baseline > 0]
    if len(positive) < 20 or np.isclose(positive.std(ddof=1), 0):
        mean = baseline.mean()
        sd = baseline.std(ddof=1)
        return (values - mean) / sd if sd > 0 else np.full_like(values, np.nan), "z fallback"
    zero_probability = float(np.mean(baseline <= 0))
    method = "gamma mle"
    try:
        shape, _, scale = gamma.fit(positive, floc=0)
    except Exception:
        mean = positive.mean()
        variance = positive.var(ddof=1)
        shape = mean * mean / variance
        scale = variance / mean
        method = "gamma moments fallback"
    probability = zero_probability + (1 - zero_probability) * gamma.cdf(
        np.maximum(values, 0), a=shape, loc=0, scale=scale
    )
    probability = np.clip(probability, 1e-8, 1 - 1e-8)
    result = norm.ppf(probability)
    result[~np.isfinite(values)] = np.nan
    return result, method


def add_monthly_shocks(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = monthly.copy()
    code = "Climate Geography Code"
    for window in SPI_WINDOWS:
        output[f"Accumulated Rainfall {window} Month mm"] = (
            output.groupby(code, sort=False)["Rainfall mm"]
            .rolling(window, min_periods=window)
            .sum()
            .reset_index(level=0, drop=True)
        )

    output["Monthly Rainfall Anomaly Z (1991-2020)"] = np.nan
    output["Monthly Dry Shock"] = False
    output["Monthly Extreme Wet Shock"] = False
    for window in SPI_WINDOWS:
        output[f"SPI {window} Month"] = np.nan

    validation_rows: list[dict[str, object]] = []
    baseline_mask = output["Year"].between(BASELINE_START, BASELINE_END)
    for (_, _), index in output.groupby([code, "Month"], sort=False).groups.items():
        idx = np.asarray(index, dtype=int)
        base_idx = idx[baseline_mask.iloc[idx].to_numpy()]
        rainfall_base = output.loc[base_idx, "Rainfall mm"].to_numpy(dtype=float)
        mean = rainfall_base.mean()
        sd = rainfall_base.std(ddof=1)
        q10, q90 = np.quantile(rainfall_base, [0.10, 0.90])
        rainfall = output.loc[idx, "Rainfall mm"].to_numpy(dtype=float)
        output.loc[idx, "Monthly Rainfall Anomaly Z (1991-2020)"] = (
            rainfall - mean
        ) / sd
        output.loc[idx, "Monthly Dry Shock"] = rainfall <= q10
        output.loc[idx, "Monthly Extreme Wet Shock"] = rainfall >= q90

        for window in SPI_WINDOWS:
            column = f"Accumulated Rainfall {window} Month mm"
            baseline = output.loc[base_idx, column].to_numpy(dtype=float)
            values = output.loc[idx, column].to_numpy(dtype=float)
            spi, method = fit_gamma_spi(baseline, values)
            output.loc[idx, f"SPI {window} Month"] = spi
            validation_rows.append(
                {
                    "SPI Window Months": window,
                    "Fit Method": method,
                }
            )

    for window in SPI_WINDOWS:
        spi = output[f"SPI {window} Month"]
        drought = spi.le(-1).astype("boolean")
        drought[spi.isna()] = pd.NA
        output[f"Drought Shock SPI {window}"] = drought
    validation = (
        pd.DataFrame(validation_rows)
        .value_counts(["SPI Window Months", "Fit Method"])
        .rename("Geography Calendar Month Fits")
        .reset_index()
    )
    return output, validation


def annual_from_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    code = "Climate Geography Code"
    annual = (
        monthly.groupby([code, "Climate Geography Name", "Year"], as_index=False)
        .agg(
            **{
                "Annual Rainfall mm": ("Rainfall mm", "sum"),
                "Observed Climate Months": ("Rainfall mm", "count"),
                "Climate Grid Cell Count": ("Climate Grid Cell Count", "first"),
                "Climate Extraction Method": ("Climate Extraction Method", "first"),
            }
        )
    )
    growing = (
        monthly[monthly["Month"].between(5, 10)]
        .groupby([code, "Year"])["Rainfall mm"]
        .sum()
        .rename("May October Rainfall mm")
        .reset_index()
    )
    annual = annual.merge(growing, on=[code, "Year"], how="left", validate="one_to_one")
    for variable, stem in [
        ("Annual Rainfall mm", "Annual Rainfall"),
        ("May October Rainfall mm", "May October Rainfall"),
    ]:
        annual[f"{stem} Anomaly Z (1991-2020)"] = np.nan
        annual[f"{stem} Dry Shock"] = False
        annual[f"{stem} Extreme Wet Shock"] = False
        for _, index in annual.groupby(code, sort=False).groups.items():
            idx = np.asarray(index, dtype=int)
            base_idx = idx[
                annual.loc[idx, "Year"].between(BASELINE_START, BASELINE_END).to_numpy()
            ]
            baseline = annual.loc[base_idx, variable].to_numpy(dtype=float)
            values = annual.loc[idx, variable].to_numpy(dtype=float)
            mean = baseline.mean()
            sd = baseline.std(ddof=1)
            q10, q90 = np.quantile(baseline, [0.10, 0.90])
            annual.loc[idx, f"{stem} Anomaly Z (1991-2020)"] = (
                values - mean
            ) / sd
            annual.loc[idx, f"{stem} Dry Shock"] = values <= q10
            annual.loc[idx, f"{stem} Extreme Wet Shock"] = values >= q90
    return annual


def geography_layers(boundary: gpd.GeoDataFrame) -> dict[str, tuple[gpd.GeoDataFrame, str, str]]:
    districts = boundary.dissolve(by="ADM2_PCODE", as_index=False).sort_values("ADM2_PCODE")
    provinces = boundary.dissolve(by="ADM1_PCODE", as_index=False).sort_values("ADM1_PCODE")
    return {
        "commune": (boundary, "ADM3_PCODE", "ADM3_EN"),
        "district": (districts, "ADM2_PCODE", "ADM2_EN"),
        "province": (provinces, "ADM1_PCODE", "ADM1_EN"),
    }


def attach_to_psu(
    crosswalk: pd.DataFrame,
    monthly_by_resolution: dict[str, pd.DataFrame],
    annual_by_resolution: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    pieces = []
    annual_columns = [
        "Annual Rainfall mm",
        "Observed Climate Months",
        "May October Rainfall mm",
        "Annual Rainfall Anomaly Z (1991-2020)",
        "Annual Rainfall Dry Shock",
        "Annual Rainfall Extreme Wet Shock",
        "May October Rainfall Anomaly Z (1991-2020)",
        "May October Rainfall Dry Shock",
        "May October Rainfall Extreme Wet Shock",
    ]
    monthly_columns = [
        "Rainfall mm",
        "Monthly Rainfall Anomaly Z (1991-2020)",
        "Monthly Dry Shock",
        "Monthly Extreme Wet Shock",
        "SPI 3 Month",
        "SPI 6 Month",
        "SPI 12 Month",
        "Drought Shock SPI 3",
        "Drought Shock SPI 6",
        "Drought Shock SPI 12",
    ]
    for resolution in ["commune", "district", "province"]:
        subset = crosswalk[
            crosswalk["Climate Geography Resolution"].eq(resolution)
        ].copy()
        subset["Survey Month Numeric"] = pd.to_numeric(
            subset["Survey Month"], errors="coerce"
        ).astype("Int64")
        annual = annual_by_resolution[resolution][
            ["Climate Geography Code", "Year"] + annual_columns
        ]
        joined = subset.merge(
            annual,
            left_on=["Climate Geography Code", "Survey Year"],
            right_on=["Climate Geography Code", "Year"],
            how="left",
            validate="many_to_one",
        ).drop(columns="Year")
        monthly = monthly_by_resolution[resolution][
            ["Climate Geography Code", "Year", "Month"] + monthly_columns
        ].rename(columns={column: f"Interview Month {column}" for column in monthly_columns})
        joined = joined.merge(
            monthly,
            left_on=["Climate Geography Code", "Survey Year", "Survey Month Numeric"],
            right_on=["Climate Geography Code", "Year", "Month"],
            how="left",
            validate="many_to_one",
        ).drop(columns=["Year", "Month"])
        pieces.append(joined)
    output = pd.concat(pieces, ignore_index=True, sort=False)
    output["Long Baseline Climate Link Matched"] = output["Annual Rainfall mm"].notna()
    output["Interview Month Climate Link Matched"] = output[
        "Interview Month Rainfall mm"
    ].notna()
    keep = [
        "Survey Year",
        "Survey Wave",
        "PSU",
        "Survey Month Numeric",
        "Climate Geography Resolution",
        "Climate Geography Code",
        "Climate Link Method",
        "Long Baseline Climate Link Matched",
        "Interview Month Climate Link Matched",
    ] + annual_columns + [f"Interview Month {column}" for column in monthly_columns]
    return output[keep].sort_values(["Survey Year", "PSU"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    exp = root / "data" / "exp" / "data-preprocessing"
    processed.mkdir(parents=True, exist_ok=True)
    exp.mkdir(parents=True, exist_ok=True)

    boundary = gpd.read_file(
        raw / "geography" / "cambodia_commune_boundaries_2018_2024.geojson"
    ).to_crs(4326)
    with xr.open_dataset(
        raw / "climate" / "chirps_v2_monthly_cambodia_1981_2021.nc"
    ) as dataset:
        times = pd.DatetimeIndex(dataset["time"].values)
        latitudes = dataset["latitude"].values.astype(float)
        longitudes = dataset["longitude"].values.astype(float)
        precip_flat = dataset["precip"].values.astype(float).reshape(len(times), -1)
    if times.min() != pd.Timestamp("1981-01-01") or times.max() != pd.Timestamp("2021-12-01"):
        raise ValueError("Unexpected CHIRPS time coverage")

    monthly_by_resolution: dict[str, pd.DataFrame] = {}
    annual_by_resolution: dict[str, pd.DataFrame] = {}
    validations = []
    for resolution, (geography, code, name) in geography_layers(boundary).items():
        monthly = extract_monthly(
            geography, code, name, times, latitudes, longitudes, precip_flat
        )
        monthly, fit_validation = add_monthly_shocks(monthly)
        annual = annual_from_monthly(monthly)
        monthly_by_resolution[resolution] = monthly
        annual_by_resolution[resolution] = annual
        monthly_path = processed / f"chirps_long_baseline_{resolution}_month_preprocessed.parquet"
        annual_path = processed / f"chirps_long_baseline_{resolution}_year_preprocessed.parquet"
        monthly.to_parquet(monthly_path, index=False)
        annual.to_parquet(annual_path, index=False)
        fit_validation.insert(0, "Climate Geography Resolution", resolution)
        validations.append(fit_validation)
        print(
            f"{resolution}: monthly_rows={len(monthly):,}, annual_rows={len(annual):,}",
            flush=True,
        )

    pd.concat(validations, ignore_index=True).to_csv(
        exp / "direction3_chirps_spi_fit_validation.csv", index=False
    )
    crosswalk = pd.read_parquet(
        processed / "direction3_historical_geography_crosswalk_preprocessed.parquet"
    )
    psu = attach_to_psu(crosswalk, monthly_by_resolution, annual_by_resolution)
    psu_path = processed / "direction3_psu_climate_shocks_preprocessed.parquet"
    psu.to_parquet(psu_path, index=False)
    if not psu["Long Baseline Climate Link Matched"].all():
        raise ValueError("One or more PSU-wave rows lack long-baseline annual climate measures")
    coverage = (
        psu.groupby("Survey Year")
        .agg(
            **{
                "PSU Wave Rows": ("PSU", "size"),
                "Annual Climate Linked Rows": ("Long Baseline Climate Link Matched", "sum"),
                "Interview Month Linked Rows": ("Interview Month Climate Link Matched", "sum"),
            }
        )
        .reset_index()
    )
    coverage.to_csv(exp / "direction3_chirps_shock_coverage.csv", index=False)
    print(f"{psu_path.name}: rows={len(psu):,}, columns={len(psu.columns):,}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
