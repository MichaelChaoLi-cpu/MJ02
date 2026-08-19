"""Prepare WFP food-price shocks for the Cambodia conflict-shock study.

The script preserves the cleaned market-level observations and constructs two
complementary province-month series:

1. a 2003--2021 wholesale, low-quality mixed-rice series; and
2. a 2013--2021 broad retail food-price series aggregated across commodities.

Local price pressure is measured relative to the national median for the same
month and product. Missing observations are never imputed and no values are
winsorized.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMNS = {
    "date": "Date",
    "admin1": "Province Name",
    "admin2": "District Name",
    "market": "Market Name",
    "market_id": "Market ID",
    "latitude": "Market Latitude",
    "longitude": "Market Longitude",
    "category": "Food Category",
    "commodity": "Commodity Name",
    "commodity_id": "Commodity ID",
    "unit": "Unit",
    "priceflag": "Price Flag",
    "pricetype": "Price Type",
    "currency": "Currency",
    "price": "Price KHR",
    "usdprice": "Reported Price USD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args()


def clean_market_observations(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path).rename(columns=RAW_COLUMNS)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for column in [
        "Province Name",
        "District Name",
        "Market Name",
        "Food Category",
        "Commodity Name",
        "Unit",
        "Price Flag",
        "Price Type",
        "Currency",
    ]:
        data[column] = data[column].astype("string").str.strip()

    data["Price KHR"] = pd.to_numeric(data["Price KHR"], errors="coerce")
    data["Reported Price USD"] = pd.to_numeric(
        data["Reported Price USD"], errors="coerce"
    )
    data["Valid Positive KHR Price"] = (
        data["Date"].notna()
        & data["Province Name"].notna()
        & data["Commodity Name"].notna()
        & data["Price KHR"].gt(0)
        & data["Currency"].eq("KHR")
    )
    data["Log Price KHR"] = np.where(
        data["Valid Positive KHR Price"], np.log(data["Price KHR"]), np.nan
    )
    data["Year"] = data["Date"].dt.year.astype("Int64")
    data["Month"] = data["Date"].dt.month.astype("Int64")
    data["Year Month"] = data["Date"].dt.to_period("M").astype("string")

    keys = ["Market ID", "Commodity ID", "Unit", "Price Type"]
    valid = data["Valid Positive KHR Price"] & data["Year"].between(2003, 2021)
    stats = (
        data.loc[valid]
        .groupby(keys, dropna=False)["Log Price KHR"]
        .agg(
            **{
                "Series Observation Count": "size",
                "Series Mean": "mean",
                "Series SD": "std",
            }
        )
        .reset_index()
    )
    data = data.merge(stats, on=keys, how="left", validate="many_to_one")
    eligible = data["Series Observation Count"].ge(12) & data["Series SD"].gt(0)
    data["Within Series Standardized Log Price"] = np.where(
        eligible,
        (data["Log Price KHR"] - data["Series Mean"]) / data["Series SD"],
        np.nan,
    )
    data = data.drop(columns=["Series Mean", "Series SD"])
    return data.sort_values(["Date", "Province Name", "Market ID", "Commodity ID"])


def add_lag_changes(
    data: pd.DataFrame, keys: list[str], value_columns: list[str]
) -> pd.DataFrame:
    output = data.copy()
    lag = output[keys + ["Date"] + value_columns].copy()
    lag["Date"] = lag["Date"] + pd.DateOffset(years=1)
    lag = lag.rename(columns={column: f"{column} 12 Months Earlier" for column in value_columns})
    output = output.merge(lag, on=keys + ["Date"], how="left", validate="one_to_one")
    for column in value_columns:
        output[f"12 Month Change in {column}"] = (
            output[column] - output[f"{column} 12 Months Earlier"]
        )
    return output


def build_wholesale_rice(data: pd.DataFrame) -> pd.DataFrame:
    rice = data.loc[
        data["Valid Positive KHR Price"]
        & data["Year"].between(2003, 2021)
        & data["Commodity Name"].eq("Rice (mixed, low quality)")
        & data["Price Type"].eq("Wholesale")
        & data["Unit"].eq("KG")
    ].copy()
    province = (
        rice.groupby(["Province Name", "Date"], as_index=False)
        .agg(
            Wholesale_Rice_Price_KHR_per_kg=("Price KHR", "median"),
            Wholesale_Rice_Market_Count=("Market ID", "nunique"),
            Wholesale_Rice_Observation_Count=("Price KHR", "size"),
        )
        .rename(
            columns={
                "Wholesale_Rice_Price_KHR_per_kg": "Wholesale Rice Price KHR per kg",
                "Wholesale_Rice_Market_Count": "Wholesale Rice Market Count",
                "Wholesale_Rice_Observation_Count": "Wholesale Rice Observation Count",
            }
        )
    )
    national = (
        rice.groupby("Date", as_index=False)
        .agg(
            National_Wholesale_Rice_Price_KHR_per_kg=("Price KHR", "median"),
            National_Wholesale_Rice_Market_Count=("Market ID", "nunique"),
        )
        .rename(
            columns={
                "National_Wholesale_Rice_Price_KHR_per_kg": "National Wholesale Rice Price KHR per kg",
                "National_Wholesale_Rice_Market_Count": "National Wholesale Rice Market Count",
            }
        )
    )
    province = province.merge(national, on="Date", how="left", validate="many_to_one")
    province["Log Wholesale Rice Price KHR per kg"] = np.log(
        province["Wholesale Rice Price KHR per kg"]
    )
    province["Local Relative Log Wholesale Rice Price"] = (
        province["Log Wholesale Rice Price KHR per kg"]
        - np.log(province["National Wholesale Rice Price KHR per kg"])
    )
    province = add_lag_changes(
        province,
        ["Province Name"],
        ["Log Wholesale Rice Price KHR per kg", "Local Relative Log Wholesale Rice Price"],
    )
    province["Year"] = province["Date"].dt.year.astype("Int64")
    province["Month"] = province["Date"].dt.month.astype("Int64")
    return province.sort_values(["Province Name", "Date"])


def build_retail_food(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    retail = data.loc[
        data["Valid Positive KHR Price"]
        & data["Year"].between(2013, 2021)
        & data["Price Type"].eq("Retail")
        & data["Food Category"].ne("non-food")
    ].copy()

    commodity = (
        retail.groupby(
            ["Province Name", "Date", "Food Category", "Commodity Name", "Commodity ID", "Unit"],
            as_index=False,
        )
        .agg(
            Province_Commodity_Median_Price_KHR=("Price KHR", "median"),
            Commodity_Market_Count=("Market ID", "nunique"),
            Commodity_Observation_Count=("Price KHR", "size"),
        )
        .rename(
            columns={
                "Province_Commodity_Median_Price_KHR": "Province Commodity Median Price KHR",
                "Commodity_Market_Count": "Commodity Market Count",
                "Commodity_Observation_Count": "Commodity Observation Count",
            }
        )
    )
    national = (
        retail.groupby(["Date", "Commodity ID", "Unit"], as_index=False)
        .agg(
            National_Commodity_Median_Price_KHR=("Price KHR", "median"),
            National_Commodity_Market_Count=("Market ID", "nunique"),
        )
        .rename(
            columns={
                "National_Commodity_Median_Price_KHR": "National Commodity Median Price KHR",
                "National_Commodity_Market_Count": "National Commodity Market Count",
            }
        )
    )
    commodity = commodity.merge(
        national,
        on=["Date", "Commodity ID", "Unit"],
        how="left",
        validate="many_to_one",
    )
    commodity["Local Relative Log Retail Commodity Price"] = np.log(
        commodity["Province Commodity Median Price KHR"]
    ) - np.log(commodity["National Commodity Median Price KHR"])

    broad = (
        commodity.groupby(["Province Name", "Date"], as_index=False)
        .agg(
            Broad_Retail_Food_Local_Relative_Log_Price=(
                "Local Relative Log Retail Commodity Price",
                "mean",
            ),
            Broad_Retail_Food_Commodity_Count=("Commodity ID", "nunique"),
            Broad_Retail_Food_Market_Count=("Commodity Market Count", "max"),
            Broad_Retail_Food_Observation_Count=("Commodity Observation Count", "sum"),
        )
        .rename(
            columns={
                "Broad_Retail_Food_Local_Relative_Log_Price": "Broad Retail Food Local Relative Log Price",
                "Broad_Retail_Food_Commodity_Count": "Broad Retail Food Commodity Count",
                "Broad_Retail_Food_Market_Count": "Broad Retail Food Maximum Commodity Market Count",
                "Broad_Retail_Food_Observation_Count": "Broad Retail Food Observation Count",
            }
        )
    )
    broad["Broad Retail Food Coverage Adequate"] = broad[
        "Broad Retail Food Commodity Count"
    ].ge(2)
    broad.loc[
        ~broad["Broad Retail Food Coverage Adequate"],
        "Broad Retail Food Local Relative Log Price",
    ] = np.nan
    broad = add_lag_changes(
        broad,
        ["Province Name"],
        ["Broad Retail Food Local Relative Log Price"],
    )
    broad["Year"] = broad["Date"].dt.year.astype("Int64")
    broad["Month"] = broad["Date"].dt.month.astype("Int64")
    return commodity.sort_values(["Province Name", "Date", "Commodity ID"]), broad.sort_values(
        ["Province Name", "Date"]
    )


def attach_to_psu(
    psu: pd.DataFrame, rice: pd.DataFrame, retail: pd.DataFrame
) -> pd.DataFrame:
    output = psu[
        [
            "Survey Year",
            "Survey Wave",
            "PSU",
            "Matched Climate Province Name",
        ]
    ].copy()
    climate = pd.read_parquet(
        Path(psu.attrs["root"]) / "data/processed/direction3_psu_climate_shocks_preprocessed.parquet"
    )[["Survey Year", "Survey Wave", "PSU", "Survey Month Numeric"]]
    output = output.merge(
        climate,
        on=["Survey Year", "Survey Wave", "PSU"],
        how="left",
        validate="one_to_one",
    )
    output["Price Exposure Date"] = pd.NaT
    has_interview_month = output["Survey Month Numeric"].notna()
    output.loc[has_interview_month, "Price Exposure Date"] = pd.to_datetime(
        output.loc[has_interview_month, "Survey Year"].astype("string")
        + "-"
        + output.loc[has_interview_month, "Survey Month Numeric"]
        .astype("Int64")
        .astype("string")
        + "-15",
        errors="coerce",
    )

    rice_columns = [column for column in rice.columns if column not in {"Year", "Month"}]
    rice_for_merge = rice[rice_columns].rename(columns={"Province Name": "Matched Climate Province Name"})
    output = output.merge(
        rice_for_merge,
        left_on=["Matched Climate Province Name", "Price Exposure Date"],
        right_on=["Matched Climate Province Name", "Date"],
        how="left",
        validate="many_to_one",
    ).drop(columns="Date")
    output["Wholesale Rice Price Linked"] = output[
        "Wholesale Rice Price KHR per kg"
    ].notna()

    retail_columns = [column for column in retail.columns if column not in {"Year", "Month"}]
    retail_for_merge = retail[retail_columns].rename(
        columns={"Province Name": "Matched Climate Province Name"}
    )
    output = output.merge(
        retail_for_merge,
        left_on=["Matched Climate Province Name", "Price Exposure Date"],
        right_on=["Matched Climate Province Name", "Date"],
        how="left",
        validate="many_to_one",
    ).drop(columns="Date")
    output["Broad Retail Food Price Linked"] = output[
        "Broad Retail Food Local Relative Log Price"
    ].notna()
    return output


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    processed = root / "data/processed"
    diagnostics = root / "data/exp/data-preprocessing"
    processed.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)

    cleaned = clean_market_observations(root / "data/raw/food_prices/wfp_food_prices_khm.csv")
    rice = build_wholesale_rice(cleaned)
    retail_commodity, retail = build_retail_food(cleaned)

    psu = pd.read_parquet(processed / "direction3_psu_conflict_exposure_preprocessed.parquet")
    psu.attrs["root"] = str(root)
    psu_prices = attach_to_psu(psu, rice, retail)

    cleaned.to_parquet(processed / "wfp_market_food_prices_cleaned.parquet", index=False)
    rice.to_parquet(
        processed / "wfp_wholesale_rice_province_month_preprocessed.parquet", index=False
    )
    retail_commodity.to_parquet(
        processed / "wfp_retail_food_commodity_province_month_preprocessed.parquet",
        index=False,
    )
    retail.to_parquet(
        processed / "wfp_retail_food_province_month_preprocessed.parquet", index=False
    )
    psu_prices.to_parquet(
        processed / "direction3_psu_food_price_shocks_preprocessed.parquet", index=False
    )

    coverage = (
        psu_prices.groupby("Survey Year", as_index=False)
        .agg(
            PSU_Wave_Rows=("PSU", "size"),
            PSU_Rows_with_Interview_Month=("Price Exposure Date", lambda x: x.notna().sum()),
            Wholesale_Rice_Price_Linked_Rows=("Wholesale Rice Price Linked", "sum"),
            Broad_Retail_Food_Price_Linked_Rows=("Broad Retail Food Price Linked", "sum"),
        )
        .rename(
            columns={
                "PSU_Wave_Rows": "PSU Wave Rows",
                "PSU_Rows_with_Interview_Month": "PSU Rows with Interview Month",
                "Wholesale_Rice_Price_Linked_Rows": "Wholesale Rice Price Linked Rows",
                "Broad_Retail_Food_Price_Linked_Rows": "Broad Retail Food Price Linked Rows",
            }
        )
    )
    coverage.to_csv(
        diagnostics / "direction3_wfp_price_shock_coverage.csv", index=False
    )
    validation = pd.DataFrame(
        [
            {"Check": "Raw observations", "Value": len(cleaned)},
            {
                "Check": "Invalid or nonpositive KHR prices retained but not transformed",
                "Value": int((~cleaned["Valid Positive KHR Price"]).sum()),
            },
            {"Check": "Wholesale rice province-month rows", "Value": len(rice)},
            {"Check": "Wholesale rice provinces", "Value": rice["Province Name"].nunique()},
            {"Check": "Retail commodity province-month rows", "Value": len(retail_commodity)},
            {"Check": "Broad retail province-month rows", "Value": len(retail)},
            {
                "Check": "Broad retail rows below two-commodity threshold",
                "Value": int((~retail["Broad Retail Food Coverage Adequate"]).sum()),
            },
            {"Check": "PSU-wave rows", "Value": len(psu_prices)},
            {"Check": "Duplicate PSU-wave rows", "Value": int(psu_prices.duplicated(["Survey Year", "Survey Wave", "PSU"]).sum())},
        ]
    )
    validation.to_csv(diagnostics / "direction3_wfp_price_validation.csv", index=False)

    print(f"cleaned market observations: {len(cleaned):,}")
    print(f"wholesale rice province-month rows: {len(rice):,}")
    print(f"broad retail province-month rows: {len(retail):,}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
