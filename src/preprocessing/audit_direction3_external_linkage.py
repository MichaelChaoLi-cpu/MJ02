"""Audit spatial and temporal linkage of the approved Direction 3 sources.

Outputs are exploratory candidates under ``data/exp/feasibility-check``. The
script deliberately preserves separate bombing, prison, and burial components;
it does not choose or construct a final conflict-legacy index.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def spatial_join_points(points_path: Path, communes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    points = gpd.read_file(points_path).to_crs(communes.crs)
    joined = gpd.sjoin(
        points,
        communes[["com_code", "geometry"]],
        how="left",
        predicate="within",
    )
    return joined


def conflict_commune_candidate(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = root / "data" / "raw"
    communes = gpd.read_file(raw / "geography" / "odc_cambodia_communes_2014.gpkg")
    communes["commune_code"] = pd.to_numeric(communes["com_code"], errors="coerce").astype(
        "Int64"
    )
    base = communes[["commune_code", "com_name", "dis_code", "pro_code"]].copy()
    base = base.rename(
        columns={
            "com_name": "commune_name_2014",
            "dis_code": "district_code_2014",
            "pro_code": "province_code_2014",
        }
    )

    bombing = spatial_join_points(
        raw / "conflict" / "yale_cgeo_us_bombing_sites.geojson", communes
    )
    bombing["bombing_date"] = pd.to_datetime(bombing["DATE"], unit="ms", errors="coerce")
    bombing["coordinate_key"] = (
        bombing.geometry.x.round(5).astype(str) + "," + bombing.geometry.y.round(5).astype(str)
    )
    bombing_agg = (
        bombing.dropna(subset=["com_code"])
        .groupby("com_code", as_index=False)
        .agg(
            bombing_record_count=("OBJECTID", "size"),
            bombing_unique_location_count=("coordinate_key", "nunique"),
            bombing_unique_date_count=("bombing_date", "nunique"),
            bombing_first_date=("bombing_date", "min"),
            bombing_last_date=("bombing_date", "max"),
            bombing_reported_aircraft_count=("NUM_ACRFT", "sum"),
        )
        .rename(columns={"com_code": "commune_code"})
    )

    prisons = spatial_join_points(
        raw / "conflict" / "yale_cgeo_khmer_rouge_prisons.geojson", communes
    )
    prison_agg = (
        prisons.dropna(subset=["com_code"])
        .groupby("com_code", as_index=False)
        .agg(khmer_rouge_prison_count=("OBJECTID", "size"))
        .rename(columns={"com_code": "commune_code"})
    )

    burials = spatial_join_points(
        raw / "conflict" / "yale_cgeo_khmer_rouge_burials.geojson", communes
    )
    burial_agg = (
        burials.dropna(subset=["com_code"])
        .groupby("com_code", as_index=False)
        .agg(
            khmer_rouge_burial_site_count=("OBJECTID", "size"),
            khmer_rouge_reported_grave_count=("MASSGRAVES", "sum"),
            khmer_rouge_reported_body_count=("BODIES", "sum"),
        )
        .rename(columns={"com_code": "commune_code"})
    )

    candidate = base.merge(bombing_agg, on="commune_code", how="left", validate="one_to_one")
    candidate = candidate.merge(prison_agg, on="commune_code", how="left", validate="one_to_one")
    candidate = candidate.merge(burial_agg, on="commune_code", how="left", validate="one_to_one")
    count_columns = [column for column in candidate if column.endswith("_count")]
    candidate[count_columns] = candidate[count_columns].fillna(0)
    candidate["any_us_bombing_record"] = candidate["bombing_record_count"].gt(0)
    candidate["any_khmer_rouge_site"] = candidate[
        ["khmer_rouge_prison_count", "khmer_rouge_burial_site_count"]
    ].sum(axis=1).gt(0)

    source_summary = pd.DataFrame(
        [
            {
                "source": "Yale CGEO US bombing",
                "source_records": len(bombing),
                "records_with_2014_commune": int(bombing["com_code"].notna().sum()),
                "communes_with_record": int(candidate["any_us_bombing_record"].sum()),
            },
            {
                "source": "Yale CGEO Khmer Rouge prisons",
                "source_records": len(prisons),
                "records_with_2014_commune": int(prisons["com_code"].notna().sum()),
                "communes_with_record": int(candidate["khmer_rouge_prison_count"].gt(0).sum()),
            },
            {
                "source": "Yale CGEO Khmer Rouge burials",
                "source_records": len(burials),
                "records_with_2014_commune": int(burials["com_code"].notna().sum()),
                "communes_with_record": int(
                    candidate["khmer_rouge_burial_site_count"].gt(0).sum()
                ),
            },
        ]
    )
    return candidate, source_summary


def cses_linkage(root: Path, candidate: pd.DataFrame) -> pd.DataFrame:
    processed = root / "data" / "processed"
    psu = pd.read_parquet(
        processed / "direction3_historical_geography_crosswalk_preprocessed.parquet"
    )
    psu["climate_geography_code_numeric"] = pd.to_numeric(
        psu["Climate Geography Code"], errors="coerce"
    ).astype("Int64")
    available_codes = {
        "commune": set(candidate["commune_code"].dropna().astype(int)),
        "district": set(candidate["district_code_2014"].dropna().astype(int)),
        "province": set(candidate["province_code_2014"].dropna().astype(int)),
    }
    psu["conflict_geography_link_matched"] = [
        pd.notna(code) and int(code) in available_codes.get(resolution, set())
        for resolution, code in zip(
            psu["Climate Geography Resolution"],
            psu["climate_geography_code_numeric"],
            strict=True,
        )
    ]
    return (
        psu.groupby("Survey Year", dropna=False)
        .agg(
            psu_rows=("PSU", "size"),
            unique_psus=("PSU", "nunique"),
            unique_conflict_geography_codes=("climate_geography_code_numeric", "nunique"),
            commune_resolution_psu_rows=(
                "Climate Geography Resolution",
                lambda values: values.eq("commune").sum(),
            ),
            district_resolution_psu_rows=(
                "Climate Geography Resolution",
                lambda values: values.eq("district").sum(),
            ),
            province_resolution_psu_rows=(
                "Climate Geography Resolution",
                lambda values: values.eq("province").sum(),
            ),
            conflict_geography_linked_psu_rows=("conflict_geography_link_matched", "sum"),
        )
        .reset_index()
        .assign(
            conflict_geography_link_rate=lambda frame: frame[
                "conflict_geography_linked_psu_rows"
            ]
            / frame["psu_rows"]
        )
    )


def wfp_coverage(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(root / "data" / "raw" / "food_prices" / "wfp_food_prices_khm.csv")
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    prices["year"] = prices["date"].dt.year
    coverage = (
        prices.groupby("year")
        .agg(
            price_rows=("price", "size"),
            markets=("market_id", "nunique"),
            provinces=("admin1", "nunique"),
            districts=("admin2", "nunique"),
            commodities=("commodity_id", "nunique"),
            earliest_date=("date", "min"),
            latest_date=("date", "max"),
            retail_share=("pricetype", lambda values: values.eq("Retail").mean()),
        )
        .reset_index()
    )

    psu = pd.read_parquet(
        root / "data" / "processed" / "direction3_psu_year_climate_enhanced_preprocessed.parquet"
    )
    cses_provinces = {
        normalize_name(value): value
        for value in psu["Matched Climate Province Name"].dropna().unique()
    }
    wfp_provinces = {
        normalize_name(value): value for value in prices["admin1"].dropna().unique()
    }
    names = sorted(set(cses_provinces) | set(wfp_provinces))
    province_crosswalk = pd.DataFrame(
        {
            "normalized_name": names,
            "cses_province_name": [cses_provinces.get(name) for name in names],
            "wfp_province_name": [wfp_provinces.get(name) for name in names],
        }
    )
    province_crosswalk["exact_normalized_match"] = province_crosswalk[
        ["cses_province_name", "wfp_province_name"]
    ].notna().all(axis=1)
    return coverage, province_crosswalk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "data" / "exp" / "feasibility-check"
    output.mkdir(parents=True, exist_ok=True)

    candidate, source_summary = conflict_commune_candidate(root)
    candidate.to_csv(output / "historical_conflict_commune_candidate.csv", index=False)
    source_summary.to_csv(output / "historical_conflict_spatial_coverage.csv", index=False)

    cses = cses_linkage(root, candidate)
    cses.to_csv(output / "historical_conflict_cses_linkage_by_wave.csv", index=False)

    wfp, province_crosswalk = wfp_coverage(root)
    wfp.to_csv(output / "wfp_price_coverage_by_year.csv", index=False)
    province_crosswalk.to_csv(output / "wfp_cses_province_crosswalk_audit.csv", index=False)

    print(f"conflict_communes={len(candidate):,}")
    print(
        f"bombed_communes={int(candidate['any_us_bombing_record'].sum()):,}; "
        f"khmer_rouge_site_communes={int(candidate['any_khmer_rouge_site'].sum()):,}"
    )
    print(
        f"cgeo_points_linked={int(source_summary['records_with_2014_commune'].sum()):,}/"
        f"{int(source_summary['source_records'].sum()):,}"
    )
    print(
        f"cses_psu_rows_linked={int(cses['conflict_geography_linked_psu_rows'].sum()):,}/"
        f"{int(cses['psu_rows'].sum()):,}"
    )
    print(
        f"wfp_years={int(wfp['year'].min())}-{int(wfp['year'].max())}; "
        f"province_name_matches={int(province_crosswalk['exact_normalized_match'].sum()):,}/"
        f"{len(province_crosswalk):,}"
    )


if __name__ == "__main__":
    main()
