#!/usr/bin/env python3
"""Test a survey-anchored, multi-season heat-NDVI-food pathway.

This exploratory diagnostic replaces the fixed May-October -> November-February
window with two linked tests at 5 km:

1. absolute heat during the 32 days strictly before a MODIS composite predicts
   the subsequent NDVI anomaly;
2. NDVI behavior across the 12 complete months before interview predicts food
   consumption conditional on heat and rainfall over the same completed year.

NDVI anomalies remain in natural index units relative to the same climate cell
and MODIS composite slot in 2001-2020.  The script distinguishes net annual
NDVI, downside-only NDVI loss, and the worst observed composite so that later
production can compensate the annual mean without erasing evidence of an
earlier adverse episode.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import duckdb
import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/exp/analysis/climate-welfare/rolling-multiseason-ndvi-food-test"

CSES = ROOT / "data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet"
MEMBERSHIP = ROOT / "data/processed/cses_village_buffer_grid_crosswalk_preprocessed.parquet"
GRID_CLIMATE = ROOT / "data/processed/cambodia_national_1km_to_climate_cell_preprocessed.parquet"
VEGETATION = ROOT / "data/processed/cambodia_national_modis_vegetation_16day"
TEMPERATURE = ROOT / "data/processed/cambodia_national_daily_temperature"
PRECIPITATION = ROOT / "data/processed/cambodia_national_daily_precipitation"

FOOD = "Real 2021 Food Consumption Value per Household Member Riels"
WEIGHT = "Household Survey Weight"
COMPOSITION = [
    "Female Household Member Share",
    "Mean Household Member Age Years",
    "Child Age 0-14 Share",
    "Older Age 65 Plus Share",
    "Household Dependency Ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-data", action="store_true")
    return parser.parse_args()


def q(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def build_analysis_frames(rebuild: bool) -> tuple[Path, Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path_a_out = OUTPUT / "village_composite_path_a_sample.parquet"
    household_out = OUTPUT / "household_rolling_12month_sample.parquet"
    if not rebuild and path_a_out.exists() and household_out.exists():
        return path_a_out, household_out

    temp_parent = OUTPUT / "duckdb-tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rolling-pathway-", dir=temp_parent) as temporary:
        database = Path(temporary) / "rolling_pathway.duckdb"
        con = duckdb.connect(database.as_posix())
        con.execute("SET threads = 4")
        con.execute("SET memory_limit = '8GB'")
        con.execute("SET preserve_insertion_order = false")
        con.execute(f"SET temp_directory = '{q(Path(temporary) / 'spill')}'")

        con.execute(f"""
          CREATE TABLE households AS
          SELECT *,
            make_date("Interview Calendar Year"::INTEGER, "Interview Month"::INTEGER, 1)
              AS interview_date,
            make_date("Interview Calendar Year"::INTEGER, "Interview Month"::INTEGER, 1)
              - INTERVAL 12 MONTH AS window_start,
            make_date("Interview Calendar Year"::INTEGER, "Interview Month"::INTEGER, 1)
              - INTERVAL 1 DAY AS window_end,
            "Survey Wave" || '_' || strftime(
              make_date("Interview Calendar Year"::INTEGER, "Interview Month"::INTEGER, 1),
              '%Y-%m'
            ) AS wave_month,
            floor(("Point Longitude" - 102.0) / 0.75)::VARCHAR || '_' ||
              floor(("Point Latitude" - 10.0) / 0.75)::VARCHAR AS spatial_block
          FROM read_parquet('{q(CSES)}')
          WHERE "Climate Ecology Link Available" = 1
            AND "Interview Month" BETWEEN 1 AND 12
            AND "Interview Calendar Year" BETWEEN 2007 AND 2021
            AND "{FOOD}" > 0
            AND "{WEIGHT}" > 0
        """)
        con.execute("""
          CREATE TABLE survey_windows AS
          SELECT DISTINCT "Village Code", interview_date, window_start, window_end
          FROM households
        """)
        con.execute(f"""
          CREATE TABLE membership AS
          SELECT m."Village Code", m."National Grid Cell ID"
          FROM read_parquet('{q(MEMBERSHIP)}') m
          SEMI JOIN (SELECT DISTINCT "Village Code" FROM households) h
            USING ("Village Code")
          WHERE m."Buffer Radius km" = 5
        """)
        con.execute(f"""
          CREATE TABLE climate_weights AS
          SELECT m."Village Code", g."Climate Cell ID", count(*)::DOUBLE AS grid_weight
          FROM membership m
          INNER JOIN read_parquet('{q(GRID_CLIMATE)}') g USING ("National Grid Cell ID")
          GROUP BY 1, 2
        """)

        print("Building climate-cell 16-day and monthly absolute-heat summaries...", flush=True)
        con.execute(f"""
          CREATE TABLE climate_16 AS
          WITH temperature AS (
            SELECT "Climate Cell ID", "Date", "Daily Maximum Temperature C"
            FROM read_parquet('{q(TEMPERATURE)}/**/*.parquet', hive_partitioning = true)
            WHERE "Date" BETWEEN DATE '2005-01-01' AND DATE '2021-12-31'
          ), precipitation AS (
            SELECT "Climate Cell ID", "Date", "Daily Precipitation mm"
            FROM read_parquet('{q(PRECIPITATION)}/**/*.parquet', hive_partitioning = true)
            WHERE "Date" BETWEEN DATE '2005-01-01' AND DATE '2021-12-31'
          )
          SELECT t."Climate Cell ID",
            make_date(year(t."Date"), 1, 1)
              + (floor((dayofyear(t."Date") - 1) / 16)::INTEGER * INTERVAL 16 DAY)
              AS composite_date,
            sum(CASE WHEN t."Daily Maximum Temperature C" >= 35 THEN 1 ELSE 0 END)
              AS heat_days_35,
            sum(p."Daily Precipitation mm") AS precipitation_mm,
            count(t."Daily Maximum Temperature C") AS valid_temperature_days,
            count(p."Daily Precipitation mm") AS valid_precipitation_days
          FROM temperature t
          INNER JOIN precipitation p USING ("Climate Cell ID", "Date")
          GROUP BY 1, 2
        """)
        con.execute(f"""
          CREATE TABLE climate_month AS
          WITH temperature AS (
            SELECT "Climate Cell ID", "Date", "Daily Maximum Temperature C"
            FROM read_parquet('{q(TEMPERATURE)}/**/*.parquet', hive_partitioning = true)
            WHERE "Date" BETWEEN DATE '2006-01-01' AND DATE '2021-12-31'
          ), precipitation AS (
            SELECT "Climate Cell ID", "Date", "Daily Precipitation mm"
            FROM read_parquet('{q(PRECIPITATION)}/**/*.parquet', hive_partitioning = true)
            WHERE "Date" BETWEEN DATE '2006-01-01' AND DATE '2021-12-31'
          )
          SELECT t."Climate Cell ID", date_trunc('month', t."Date")::DATE AS month,
            sum(CASE WHEN t."Daily Maximum Temperature C" >= 35 THEN 1 ELSE 0 END)
              AS heat_days_35,
            sum(p."Daily Precipitation mm") AS precipitation_mm,
            count(t."Daily Maximum Temperature C") AS valid_temperature_days,
            count(p."Daily Precipitation mm") AS valid_precipitation_days
          FROM temperature t
          INNER JOIN precipitation p USING ("Climate Cell ID", "Date")
          GROUP BY 1, 2
        """)
        con.execute("""
          CREATE TABLE village_climate_16 AS
          SELECT w."Village Code", c.composite_date,
            sum(c.heat_days_35 * w.grid_weight)
              / nullif(sum(w.grid_weight) FILTER (WHERE c.valid_temperature_days >= 15), 0)
              AS heat_days_35,
            sum(c.precipitation_mm * w.grid_weight)
              / nullif(sum(w.grid_weight) FILTER (WHERE c.valid_precipitation_days >= 15), 0)
              AS precipitation_mm,
            min(c.valid_temperature_days) AS minimum_valid_temperature_days,
            min(c.valid_precipitation_days) AS minimum_valid_precipitation_days
          FROM climate_weights w
          INNER JOIN climate_16 c USING ("Climate Cell ID")
          GROUP BY 1, 2
        """)
        con.execute("""
          CREATE TABLE village_climate_month AS
          SELECT w."Village Code", c.month,
            sum(c.heat_days_35 * w.grid_weight)
              / nullif(sum(w.grid_weight) FILTER (WHERE c.valid_temperature_days >= 28), 0)
              AS heat_days_35,
            sum(c.precipitation_mm * w.grid_weight)
              / nullif(sum(w.grid_weight) FILTER (WHERE c.valid_precipitation_days >= 28), 0)
              AS precipitation_mm,
            min(c.valid_temperature_days) AS minimum_valid_temperature_days,
            min(c.valid_precipitation_days) AS minimum_valid_precipitation_days
          FROM climate_weights w
          INNER JOIN climate_month c USING ("Climate Cell ID")
          GROUP BY 1, 2
        """)

        print("Building natural-unit NDVI anomalies by climate cell and composite slot...", flush=True)
        con.execute(f"""
          CREATE TABLE vegetation_cell_date AS
          SELECT g."Climate Cell ID", v."Composite Date", v."Year",
            v."MODIS Composite Slot",
            sum(v."Mean NDVI" * v."NDVI Valid Pixel Count")
              / nullif(sum(v."NDVI Valid Pixel Count") FILTER (
                  WHERE v."Mean NDVI" IS NOT NULL), 0) AS mean_ndvi,
            sum(v."NDVI Valid Pixel Count") AS valid_pixel_count
          FROM read_parquet('{q(VEGETATION)}/**/*.parquet', hive_partitioning = true) v
          INNER JOIN read_parquet('{q(GRID_CLIMATE)}') g USING ("National Grid Cell ID")
          SEMI JOIN (SELECT DISTINCT "Climate Cell ID" FROM climate_weights) w
            USING ("Climate Cell ID")
          WHERE v."Year" BETWEEN 2001 AND 2021
          GROUP BY 1, 2, 3, 4
        """)
        con.execute("""
          CREATE TABLE ndvi_baseline AS
          SELECT "Climate Cell ID", "MODIS Composite Slot",
            avg(mean_ndvi) AS baseline_mean_ndvi,
            count(mean_ndvi) AS baseline_valid_years
          FROM vegetation_cell_date
          WHERE "Year" BETWEEN 2001 AND 2020
          GROUP BY 1, 2
        """)
        con.execute("""
          CREATE TABLE village_ndvi AS
          WITH vegetation_measure AS (
            SELECT d."Climate Cell ID", d."Composite Date", d."Year",
              d.mean_ndvi,
              d.mean_ndvi - b.baseline_mean_ndvi AS ndvi_anomaly,
              d.valid_pixel_count
            FROM vegetation_cell_date d
            INNER JOIN ndvi_baseline b USING ("Climate Cell ID", "MODIS Composite Slot")
            WHERE d."Year" BETWEEN 2006 AND 2021
              AND b.baseline_valid_years >= 10
          )
          SELECT w."Village Code", a."Composite Date" AS composite_date,
            sum(a.mean_ndvi * w.grid_weight)
              / nullif(sum(w.grid_weight) FILTER (WHERE a.mean_ndvi IS NOT NULL), 0)
              AS mean_ndvi_absolute,
            sum(a.ndvi_anomaly * w.grid_weight)
              / nullif(sum(w.grid_weight) FILTER (WHERE a.ndvi_anomaly IS NOT NULL), 0)
              AS ndvi_anomaly,
            count(a.ndvi_anomaly) AS contributing_climate_cells
          FROM climate_weights w
          INNER JOIN vegetation_measure a USING ("Climate Cell ID")
          GROUP BY 1, 2
        """)

        print("Materializing the strictly ordered 32-day heat -> NDVI sample...", flush=True)
        con.execute(f"""
          COPY (
            WITH rolling_climate AS (
              SELECT *,
                sum(heat_days_35) OVER w AS prior_32day_heat_days_35,
                sum(precipitation_mm) OVER w AS prior_32day_precipitation_mm,
                sum(minimum_valid_temperature_days) OVER w AS prior_valid_temperature_days,
                sum(minimum_valid_precipitation_days) OVER w AS prior_valid_precipitation_days
              FROM village_climate_16
              WINDOW w AS (
                PARTITION BY "Village Code" ORDER BY composite_date
                ROWS BETWEEN 2 PRECEDING AND 1 PRECEDING
              )
            )
            SELECT n."Village Code", n.composite_date, n.ndvi_anomaly,
              greatest(-n.ndvi_anomaly, 0) AS ndvi_downside,
              c.prior_32day_heat_days_35,
              c.prior_32day_precipitation_mm,
              h.spatial_block,
              n.contributing_climate_cells
            FROM village_ndvi n
            INNER JOIN rolling_climate c USING ("Village Code", composite_date)
            INNER JOIN (
              SELECT "Village Code", min(spatial_block) AS spatial_block
              FROM households GROUP BY 1
            ) h USING ("Village Code")
            WHERE c.prior_valid_temperature_days >= 30
              AND c.prior_valid_precipitation_days >= 30
              AND EXISTS (
                SELECT 1 FROM survey_windows s
                WHERE s."Village Code" = n."Village Code"
                  AND n.composite_date >= s.window_start
                  AND n.composite_date + INTERVAL 15 DAY <= s.window_end
              )
            ORDER BY n."Village Code", n.composite_date
          ) TO '{q(path_a_out)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        print("Materializing household rolling-year heat, NDVI, and food sample...", flush=True)
        con.execute(f"""
          COPY (
            WITH ndvi_window AS (
              SELECT s."Village Code", s.interview_date,
                avg(n.mean_ndvi_absolute) AS annual_mean_ndvi_absolute,
                avg(n.ndvi_anomaly) AS annual_mean_ndvi_anomaly,
                avg(greatest(-n.ndvi_anomaly, 0)) AS annual_mean_ndvi_downside,
                min(n.ndvi_anomaly) AS annual_worst_ndvi_anomaly,
                count(n.ndvi_anomaly) AS valid_ndvi_composites
              FROM survey_windows s
              INNER JOIN village_ndvi n ON n."Village Code" = s."Village Code"
                AND n.composite_date >= s.window_start
                AND n.composite_date + INTERVAL 15 DAY <= s.window_end
              GROUP BY 1, 2
            ), climate_window AS (
              SELECT s."Village Code", s.interview_date,
                sum(c.heat_days_35) AS annual_heat_days_35,
                sum(c.precipitation_mm) AS annual_precipitation_mm,
                count(c.month) AS valid_climate_months
              FROM survey_windows s
              INNER JOIN village_climate_month c ON c."Village Code" = s."Village Code"
                AND c.month >= s.window_start
                AND c.month < s.interview_date
              WHERE c.minimum_valid_temperature_days >= 28
                AND c.minimum_valid_precipitation_days >= 28
              GROUP BY 1, 2
            )
            SELECT h.*, n.annual_mean_ndvi_absolute, n.annual_mean_ndvi_anomaly,
              n.annual_mean_ndvi_downside, n.annual_worst_ndvi_anomaly,
              n.valid_ndvi_composites, c.annual_heat_days_35,
              c.annual_precipitation_mm, c.valid_climate_months
            FROM households h
            INNER JOIN ndvi_window n USING ("Village Code", interview_date)
            INNER JOIN climate_window c USING ("Village Code", interview_date)
            WHERE n.valid_ndvi_composites >= 18
              AND c.valid_climate_months = 12
            ORDER BY h."Survey Year", h."Household ID"
          ) TO '{q(household_out)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        con.close()

    return path_a_out, household_out


def fit_absorbing(
    dependent: pd.Series,
    exog: pd.DataFrame,
    absorb: pd.DataFrame,
    clusters: pd.Series,
    weights: pd.Series | None = None,
) -> object:
    model = AbsorbingLS(
        dependent.astype(float),
        exog.astype(float),
        absorb=absorb,
        weights=None if weights is None else weights.astype(float),
        drop_absorbed=True,
    )
    return model.fit(
        cov_type="clustered",
        clusters=pd.Categorical(clusters).codes,
        debiased=True,
    )


def estimate_path_a(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.dropna().copy()
    data["NDVI anomaly (0.01 units)"] = data["ndvi_anomaly"] / 0.01
    data["NDVI downside (0.01 units)"] = data["ndvi_downside"] / 0.01
    data["Prior 32-day heat (10 days)"] = data["prior_32day_heat_days_35"] / 10.0
    data["Prior 32-day precipitation (100 mm)"] = data["prior_32day_precipitation_mm"] / 100.0
    exog_names = ["Prior 32-day heat (10 days)", "Prior 32-day precipitation (100 mm)"]
    absorb = pd.DataFrame(
        {
            "Village": data["Village Code"].astype("category"),
            "CompositeDate": pd.to_datetime(data["composite_date"]).astype("category"),
        },
        index=data.index,
    )
    rows: list[dict[str, object]] = []
    for outcome in ["NDVI anomaly (0.01 units)", "NDVI downside (0.01 units)"]:
        result = fit_absorbing(
            data[outcome], data[exog_names], absorb, data["spatial_block"]
        )
        for term in exog_names:
            rows.append(
                {
                    "stage": "A: strictly prior climate to subsequent NDVI",
                    "sample": "CSES-supported village-composites",
                    "outcome": outcome,
                    "term": term,
                    "coefficient": float(result.params[term]),
                    "standard_error": float(result.std_errors[term]),
                    "ci_lower_95": float(result.conf_int().loc[term, "lower"]),
                    "ci_upper_95": float(result.conf_int().loc[term, "upper"]),
                    "p_value": float(result.pvalues[term]),
                    "observations": int(result.nobs),
                    "villages": int(data["Village Code"].nunique()),
                    "spatial_blocks": int(data["spatial_block"].nunique()),
                    "fixed_effects": "village + MODIS composite date",
                }
            )
    return pd.DataFrame(rows)


def estimate_path_b(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["Log food"] = np.log(pd.to_numeric(data[FOOD], errors="coerce"))
    data["Annual heat (10 days)"] = data["annual_heat_days_35"] / 10.0
    data["Annual precipitation (1000 mm)"] = data["annual_precipitation_mm"] / 1000.0
    data["Annual mean NDVI (0.01 units)"] = data["annual_mean_ndvi_anomaly"] / 0.01
    data["Annual NDVI downside (0.01 units)"] = data["annual_mean_ndvi_downside"] / 0.01
    data["Worst NDVI anomaly (0.01 units)"] = data["annual_worst_ndvi_anomaly"] / 0.01
    required = [
        "Log food", "Annual heat (10 days)", "Annual precipitation (1000 mm)",
        "Annual mean NDVI (0.01 units)", "Annual NDVI downside (0.01 units)",
        "Worst NDVI anomaly (0.01 units)", WEIGHT, "District Code", "wave_month",
        "spatial_block", "Village Code",
    ]
    data = data.dropna(subset=required).copy()
    specifications = [
        ("Total heat response", None),
        ("Net annual NDVI pathway", "Annual mean NDVI (0.01 units)"),
        ("Downside-only NDVI pathway", "Annual NDVI downside (0.01 units)"),
        ("Worst-period NDVI pathway", "Worst NDVI anomaly (0.01 units)"),
    ]
    rows: list[dict[str, object]] = []
    for sample_name, sample in [
        ("All linked households", data),
        ("Agricultural households", data.loc[data["Agricultural Participation"].eq(True)].copy()),
    ]:
        absorb = pd.DataFrame(
            {
                "District": sample["District Code"].astype("category"),
                "WaveMonth": sample["wave_month"].astype("category"),
            },
            index=sample.index,
        )
        for specification, mediator in specifications:
            terms = ["Annual heat (10 days)", "Annual precipitation (1000 mm)"]
            if mediator is not None:
                terms.append(mediator)
            result = fit_absorbing(
                sample["Log food"], sample[terms], absorb,
                sample["spatial_block"], sample[WEIGHT]
            )
            for term in terms:
                rows.append(
                    {
                        "stage": "B: rolling-year NDVI to interview food",
                        "sample": sample_name,
                        "specification": specification,
                        "term": term,
                        "coefficient": float(result.params[term]),
                        "standard_error": float(result.std_errors[term]),
                        "ci_lower_95": float(result.conf_int().loc[term, "lower"]),
                        "ci_upper_95": float(result.conf_int().loc[term, "upper"]),
                        "p_value": float(result.pvalues[term]),
                        "observations": int(result.nobs),
                        "villages": int(sample["Village Code"].nunique()),
                        "spatial_blocks": int(sample["spatial_block"].nunique()),
                        "fixed_effects": "district + survey-wave-by-interview-month",
                        "survey_weighted": True,
                    }
                )
    return pd.DataFrame(rows)


def summarize(path_a: pd.DataFrame, path_b: pd.DataFrame, household: pd.DataFrame) -> dict[str, object]:
    heat_a = path_a.loc[path_a["term"].eq("Prior 32-day heat (10 days)")].copy()
    all_b = path_b.loc[path_b["sample"].eq("All linked households")].copy()
    total_heat = all_b.loc[
        all_b["specification"].eq("Total heat response")
        & all_b["term"].eq("Annual heat (10 days)")
    ].iloc[0]
    net = all_b.loc[
        all_b["specification"].eq("Net annual NDVI pathway")
        & all_b["term"].eq("Annual mean NDVI (0.01 units)")
    ].iloc[0]
    downside = all_b.loc[
        all_b["specification"].eq("Downside-only NDVI pathway")
        & all_b["term"].eq("Annual NDVI downside (0.01 units)")
    ].iloc[0]
    a_net = heat_a.loc[heat_a["outcome"].eq("NDVI anomaly (0.01 units)")].iloc[0]
    a_down = heat_a.loc[heat_a["outcome"].eq("NDVI downside (0.01 units)")].iloc[0]
    return {
        "design": "5 km rolling multi-season diagnostic",
        "households": int(len(household)),
        "villages": int(household["Village Code"].nunique()),
        "interview_months_included": sorted(household["Interview Month"].dropna().astype(int).unique().tolist()),
        "path_a_net_ndvi": {
            "coefficient_per_10_heat_days_in_prior_32_days": float(a_net["coefficient"]),
            "p_value": float(a_net["p_value"]),
        },
        "path_a_downside_ndvi": {
            "coefficient_per_10_heat_days_in_prior_32_days": float(a_down["coefficient"]),
            "p_value": float(a_down["p_value"]),
        },
        "annual_heat_food": {
            "log_coefficient_per_10_heat_days": float(total_heat["coefficient"]),
            "percent_change": float(100.0 * np.expm1(total_heat["coefficient"])),
            "p_value": float(total_heat["p_value"]),
        },
        "annual_net_ndvi_food": {
            "log_coefficient_per_0_01_ndvi": float(net["coefficient"]),
            "percent_change": float(100.0 * np.expm1(net["coefficient"])),
            "p_value": float(net["p_value"]),
        },
        "annual_downside_ndvi_food": {
            "log_coefficient_per_0_01_deficit": float(downside["coefficient"]),
            "percent_change": float(100.0 * np.expm1(downside["coefficient"])),
            "p_value": float(downside["p_value"]),
        },
        "interpretation_gate": {
            "short_run_damage_supported": bool(a_net["coefficient"] < 0 and a_net["p_value"] < 0.05),
            "short_run_downside_supported": bool(a_down["coefficient"] > 0 and a_down["p_value"] < 0.05),
            "net_annual_ndvi_food_supported": bool(net["coefficient"] > 0 and net["p_value"] < 0.05),
            "downside_ndvi_food_supported": bool(downside["coefficient"] < 0 and downside["p_value"] < 0.05),
        },
        "claim_limit": "exploratory longitudinal pathway diagnostic; not causal mediation",
    }


def main() -> None:
    args = parse_args()
    path_a_file, household_file = build_analysis_frames(args.rebuild_data)
    path_a_sample = pd.read_parquet(path_a_file)
    household_sample = pd.read_parquet(household_file)
    path_a = estimate_path_a(path_a_sample)
    path_b = estimate_path_b(household_sample)
    summary = summarize(path_a, path_b, household_sample)

    path_a.to_csv(OUTPUT / "path_a_heat_to_subsequent_ndvi.csv", index=False)
    path_b.to_csv(OUTPUT / "path_b_rolling_ndvi_to_food.csv", index=False)
    pd.DataFrame(
        [
            {
                "sample": "Village-composite path A",
                "observations": len(path_a_sample),
                "villages": path_a_sample["Village Code"].nunique(),
                "earliest_date": path_a_sample["composite_date"].min(),
                "latest_date": path_a_sample["composite_date"].max(),
            },
            {
                "sample": "Household rolling-year path B",
                "observations": len(household_sample),
                "villages": household_sample["Village Code"].nunique(),
                "earliest_date": household_sample["interview_date"].min(),
                "latest_date": household_sample["interview_date"].max(),
            },
        ]
    ).to_csv(OUTPUT / "sample_summary.csv", index=False)
    (OUTPUT / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "README.md").write_text(
        "# Rolling multi-season NDVI-food pathway diagnostic\n\n"
        "This is an exploratory 5 km test rather than a promoted manuscript result. "
        "Path A links heat in the 32 days strictly before a MODIS composite to the "
        "subsequent natural-unit NDVI anomaly with village and composite-date effects. "
        "Path B links NDVI behavior over the 12 completed months before interview to "
        "log real food consumption, conditional on rolling-year heat and rainfall, "
        "district effects, survey-wave-by-interview-month effects, survey weights, "
        "and spatial-block clustered uncertainty.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"Saved diagnostic outputs to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
