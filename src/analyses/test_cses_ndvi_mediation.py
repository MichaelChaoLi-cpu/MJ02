#!/usr/bin/env python3
"""Test temporally ordered NDVI mediation of the absolute-heat food response.

Exposure: post-onset days with daily maximum air temperature >=35 C in season s.
Mediator: November(s)-February(s+1) village-buffer NDVI.
Outcome: CSES food consumption observed in March-October of s+1.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS


CSES = Path("data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet")
ECOLOGY = Path("data/processed/cambodia_public_village_monsoon_ecology_panel_preprocessed.parquet")
OUTPUT = Path("data/exp/analysis/climate-welfare/cses-absolute-heat-ndvi-mediation")

ID = "National Village Point ID"
SEASON = "Last Complete May-October Season Year"
FOOD = "Real 2021 Food Consumption Value per Household Member Riels"
WEIGHT = "Household Survey Weight"
COMPOSITION = [
    "Female Household Member Share",
    "Mean Household Member Age Years",
    "Child Age 0-14 Share",
    "Older Age 65 Plus Share",
    "Household Dependency Ratio",
]
SHORT_EXPOSURES = ["Rain", "Onset", "DrySpell", "HeatDays10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def exposure_columns(candidate: str, threshold_c: int) -> dict[str, str]:
    return {
        "Village Buffer Mean May October Precipitation Total mm Anomaly Z": "Rain",
        f"Village Buffer Mean Wet-Season Onset DOY Candidate {candidate} Anomaly Z": "Onset",
        f"Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate {candidate} Anomaly Z": "DrySpell",
        (
            f"Village Buffer Mean Post-Onset Absolute Heat Day Count {threshold_c} C "
            f"Candidate {candidate}"
        ): "HeatDays10",
        "Village Buffer Mean November-February Mean NDVI Anomaly Z": "NDVI",
    }


def base_households(root: Path) -> pd.DataFrame:
    columns = [
        "Survey Year",
        "Interview Calendar Year",
        "Interview Month",
        "Village Code",
        "District Code",
        "Point Longitude",
        "Point Latitude",
        "Climate Ecology Link Available",
        ID,
        SEASON,
        FOOD,
        WEIGHT,
    ] + COMPOSITION
    frame = pd.read_parquet(root / CSES, columns=columns)
    frame = frame.loc[
        frame["Climate Ecology Link Available"].eq(1)
        & frame["Interview Month"].between(3, 10)
    ].copy()
    frame["LogFood"] = np.log(pd.to_numeric(frame[FOOD], errors="coerce"))
    frame["WaveMonth"] = (
        frame["Survey Year"].astype(str)
        + "_"
        + frame["Interview Month"].astype("Int64").astype(str)
    )
    frame["Block"] = (
        np.floor((frame["Point Longitude"] - 102.0) / 0.75).astype("Int64").astype(str)
        + "_"
        + np.floor((frame["Point Latitude"] - 10.0) / 0.75).astype("Int64").astype(str)
    )
    return frame


def build_sample(
    root: Path, base: pd.DataFrame, radius: int, candidate: str, threshold_c: int
) -> pd.DataFrame:
    mapping = exposure_columns(candidate, threshold_c)
    columns = [ID, "Year", "Buffer Radius km"] + list(mapping)
    ecology = pd.read_parquet(root / ECOLOGY, columns=columns)
    ecology = ecology.loc[ecology["Buffer Radius km"].eq(radius)].drop(columns="Buffer Radius km")
    ecology = ecology.rename(columns=mapping)
    ecology["HeatDays10"] = ecology["HeatDays10"] / 10.0
    sample = base.merge(ecology, left_on=[ID, SEASON], right_on=[ID, "Year"], how="left")
    required = [
        "LogFood",
        WEIGHT,
        "Village Code",
        "District Code",
        "WaveMonth",
        "Block",
        ID,
        "Year",
        "NDVI",
    ] + SHORT_EXPOSURES
    return sample.dropna(subset=required).copy()


def absorb_frame(sample: pd.DataFrame, village_effects: bool) -> pd.DataFrame:
    if village_effects:
        location = {"Village": sample["Village Code"].astype("category")}
    else:
        location = {"District": sample["District Code"].astype("category")}
    location["WaveMonth"] = sample["WaveMonth"].astype("category")
    return pd.DataFrame(location, index=sample.index)


def fit_model(
    dependent: pd.Series,
    exog: pd.DataFrame,
    absorb: pd.DataFrame,
    weights: pd.Series | None,
    clusters: pd.Series | None,
    clustered: bool,
) -> object:
    model = AbsorbingLS(
        dependent.astype(float),
        exog.astype(float),
        absorb=absorb,
        weights=None if weights is None else weights.astype(float),
        drop_absorbed=True,
    )
    if clustered:
        return model.fit(
            cov_type="clustered",
            clusters=pd.Categorical(clusters).codes,
            debiased=True,
        )
    return model.fit(cov_type="unadjusted", debiased=False)


def point_estimates(
    sample: pd.DataFrame,
    composition: bool = False,
    repeated_villages: bool = False,
) -> dict[str, float | int | str]:
    working = sample.copy()
    if repeated_villages:
        repeated = (
            working.groupby("Village Code", observed=True)["Survey Year"]
            .nunique()
            .loc[lambda values: values > 1]
            .index
        )
        working = working.loc[working["Village Code"].isin(repeated)].copy()
    controls = COMPOSITION if composition else []
    required = controls + SHORT_EXPOSURES + ["NDVI", "LogFood", WEIGHT]
    working = working.dropna(subset=required).copy()

    village_seasons = working[[ID, "Year", "Block", "NDVI"] + SHORT_EXPOSURES].drop_duplicates(
        [ID, "Year"]
    )
    path_a = fit_model(
        village_seasons["NDVI"],
        village_seasons[SHORT_EXPOSURES],
        pd.DataFrame(
            {
                "Village": village_seasons[ID].astype("category"),
                "Year": village_seasons["Year"].astype("category"),
            },
            index=village_seasons.index,
        ),
        None,
        village_seasons["Block"],
        True,
    )
    total_exog = SHORT_EXPOSURES + controls
    direct_exog = SHORT_EXPOSURES + ["NDVI"] + controls
    absorbed = absorb_frame(working, repeated_villages)
    total = fit_model(
        working["LogFood"],
        working[total_exog],
        absorbed,
        working[WEIGHT],
        working["Block"],
        True,
    )
    direct = fit_model(
        working["LogFood"],
        working[direct_exog],
        absorbed,
        working[WEIGHT],
        working["Block"],
        True,
    )

    a = float(path_a.params["HeatDays10"])
    b = float(direct.params["NDVI"])
    c = float(total.params["HeatDays10"])
    c_prime = float(direct.params["HeatDays10"])
    return {
        "path_a_heat_to_ndvi": a,
        "path_a_se": float(path_a.std_errors["HeatDays10"]),
        "path_a_p": float(path_a.pvalues["HeatDays10"]),
        "path_b_ndvi_to_food": b,
        "path_b_se": float(direct.std_errors["NDVI"]),
        "path_b_p": float(direct.pvalues["NDVI"]),
        "total_effect_c": c,
        "total_effect_se": float(total.std_errors["HeatDays10"]),
        "total_effect_p": float(total.pvalues["HeatDays10"]),
        "direct_effect_c_prime": c_prime,
        "direct_effect_se": float(direct.std_errors["HeatDays10"]),
        "direct_effect_p": float(direct.pvalues["HeatDays10"]),
        "indirect_effect_ab": a * b,
        "share_of_total_percent": 100.0 * a * b / c if c != 0 else np.nan,
        "absolute_attenuation": abs(c) - abs(c_prime),
        "households": int(len(working)),
        "villages": int(working["Village Code"].nunique()),
        "village_seasons": int(len(village_seasons)),
        "spatial_blocks": int(working["Block"].nunique()),
    }


def future_ndvi_placebo(root: Path, primary: pd.DataFrame) -> dict[str, float | int]:
    future = pd.read_parquet(
        root / ECOLOGY,
        columns=[ID, "Year", "Buffer Radius km", "Village Buffer Mean November-February Mean NDVI Anomaly Z"],
    )
    future = future.loc[future["Buffer Radius km"].eq(5)].drop(columns="Buffer Radius km")
    future = future.rename(
        columns={
            "Year": "FutureYear",
            "Village Buffer Mean November-February Mean NDVI Anomaly Z": "FutureNDVI",
        }
    )
    working = primary.copy()
    working["FutureYear"] = working["Year"] + 1
    working = working.merge(future, on=[ID, "FutureYear"], how="left")
    exog = SHORT_EXPOSURES + ["NDVI", "FutureNDVI"]
    working = working.dropna(subset=exog + ["LogFood", WEIGHT]).copy()
    result = fit_model(
        working["LogFood"],
        working[exog],
        absorb_frame(working, False),
        working[WEIGHT],
        working["Block"],
        True,
    )
    return {
        "future_ndvi_coefficient": float(result.params["FutureNDVI"]),
        "future_ndvi_se": float(result.std_errors["FutureNDVI"]),
        "future_ndvi_p": float(result.pvalues["FutureNDVI"]),
        "current_ndvi_coefficient": float(result.params["NDVI"]),
        "current_ndvi_p": float(result.pvalues["NDVI"]),
        "households": int(len(working)),
        "villages": int(working["Village Code"].nunique()),
    }


def bootstrap_draw(primary: pd.DataFrame, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    blocks = np.array(sorted(primary["Block"].unique()), dtype=object)
    sampled = rng.choice(blocks, size=len(blocks), replace=True)
    counts = pd.Series(sampled).value_counts()

    working = primary.loc[primary["Block"].isin(counts.index)].copy()
    multiplicity = working["Block"].map(counts).astype(float)
    village_seasons = working[[ID, "Year", "Block", "NDVI"] + SHORT_EXPOSURES].drop_duplicates(
        [ID, "Year"]
    )
    village_weights = village_seasons["Block"].map(counts).astype(float)

    path_a = fit_model(
        village_seasons["NDVI"],
        village_seasons[SHORT_EXPOSURES],
        pd.DataFrame(
            {
                "Village": village_seasons[ID].astype("category"),
                "Year": village_seasons["Year"].astype("category"),
            },
            index=village_seasons.index,
        ),
        village_weights,
        None,
        False,
    )
    direct = fit_model(
        working["LogFood"],
        working[SHORT_EXPOSURES + ["NDVI"]],
        absorb_frame(working, False),
        working[WEIGHT].astype(float) * multiplicity,
        None,
        False,
    )
    a = float(path_a.params["HeatDays10"])
    b = float(direct.params["NDVI"])
    return {"path_a": a, "path_b": b, "indirect_effect": a * b}


def bootstrap(primary: pd.DataFrame, draws: int, workers: int, seed: int) -> pd.DataFrame:
    seeds = np.random.SeedSequence(seed).generate_state(draws).astype(int).tolist()
    rows: list[dict[str, float] | None] = [None] * draws
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(bootstrap_draw, primary, draw_seed): index for index, draw_seed in enumerate(seeds)}
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            try:
                rows[index] = future.result()
            except Exception:
                rows[index] = None
            completed += 1
            if completed % 100 == 0 or completed == draws:
                print(f"bootstrap {completed}/{draws}", flush=True)
    frame = pd.DataFrame([row for row in rows if row is not None])
    frame.index.name = "draw"
    return frame.reset_index()


def bootstrap_summary(draws: pd.DataFrame) -> dict[str, float | int]:
    result: dict[str, float | int] = {"successful_draws": int(len(draws))}
    for column in ["path_a", "path_b", "indirect_effect"]:
        values = draws[column].dropna().to_numpy(float)
        result[f"{column}_bootstrap_mean"] = float(np.mean(values))
        result[f"{column}_ci_lower"] = float(np.quantile(values, 0.025))
        result[f"{column}_ci_upper"] = float(np.quantile(values, 0.975))
        result[f"{column}_two_sided_sign_probability"] = float(
            min(1.0, 2.0 * min(np.mean(values <= 0), np.mean(values >= 0)))
        )
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)

    base = base_households(root)
    specifications = [
        ("Candidate B, 35 C, 2 km", 2, "B", 35, False, False),
        ("Candidate B, 35 C, 5 km", 5, "B", 35, False, False),
        ("Candidate B, 35 C, 10 km", 10, "B", 35, False, False),
        ("Candidate A, 35 C, 5 km", 5, "A", 35, False, False),
        ("Candidate B, 33 C, 5 km", 5, "B", 33, False, False),
        ("Candidate B, 37 C, 5 km", 5, "B", 37, False, False),
        ("Candidate B, 35 C, 5 km, composition adjusted", 5, "B", 35, True, False),
        ("Candidate B, 35 C, 5 km, repeated-village fixed effects", 5, "B", 35, False, True),
    ]
    samples: dict[tuple[int, str, int], pd.DataFrame] = {}
    rows = []
    for name, radius, candidate, threshold_c, composition, repeated in specifications:
        key = (radius, candidate, threshold_c)
        if key not in samples:
            samples[key] = build_sample(root, base, radius, candidate, threshold_c)
        row = point_estimates(samples[key], composition=composition, repeated_villages=repeated)
        row.update(
            {
                "specification": name,
                "radius_km": radius,
                "candidate": candidate,
                "threshold_c": threshold_c,
                "composition_controls": composition,
                "repeated_village_effects": repeated,
            }
        )
        rows.append(row)
        print(
            f"{name}: a={row['path_a_heat_to_ndvi']:.4f}, b={row['path_b_ndvi_to_food']:.4f}, "
            f"ab={row['indirect_effect_ab']:.5f}, c={row['total_effect_c']:.4f}, "
            f"c'={row['direct_effect_c_prime']:.4f}",
            flush=True,
        )
    specification_summary = pd.DataFrame(rows)
    specification_summary.to_csv(output / "specification_summary.csv", index=False)

    primary = samples[(5, "B", 35)]
    placebo = future_ndvi_placebo(root, primary)
    pd.DataFrame([placebo]).to_csv(output / "future_ndvi_placebo.csv", index=False)
    print(
        f"Future NDVI placebo: coefficient={placebo['future_ndvi_coefficient']:.4f}, "
        f"p={placebo['future_ndvi_p']:.4f}",
        flush=True,
    )

    draws = bootstrap(primary, args.bootstrap_draws, args.workers, args.seed)
    draws.to_csv(output / "bootstrap_draws.csv", index=False)
    boot = bootstrap_summary(draws)
    primary_row = specification_summary.loc[
        specification_summary["specification"].eq("Candidate B, 35 C, 5 km")
    ].iloc[0].to_dict()
    gate = {
        "path_a_negative_and_clustered_p_below_0_05": bool(
            primary_row["path_a_heat_to_ndvi"] < 0 and primary_row["path_a_p"] < 0.05
        ),
        "path_b_positive_and_clustered_p_below_0_05": bool(
            primary_row["path_b_ndvi_to_food"] > 0 and primary_row["path_b_p"] < 0.05
        ),
        "indirect_bootstrap_ci_excludes_zero_and_is_negative": bool(
            boot["indirect_effect_ci_upper"] < 0
        ),
        "direct_effect_attenuates": bool(primary_row["absolute_attenuation"] > 0),
        "future_ndvi_placebo_p_at_least_0_05": bool(placebo["future_ndvi_p"] >= 0.05),
        "all_scale_definition_indirect_effects_negative": bool(
            specification_summary.loc[
                specification_summary["specification"].isin(
                    [
                        "Candidate B, 35 C, 2 km",
                        "Candidate B, 35 C, 5 km",
                        "Candidate B, 35 C, 10 km",
                        "Candidate A, 35 C, 5 km",
                        "Candidate B, 33 C, 5 km",
                        "Candidate B, 37 C, 5 km",
                    ]
                ),
                "indirect_effect_ab",
            ].lt(0).all()
        ),
    }
    gate["mediation_gate_passed"] = bool(all(gate.values()))
    summary = {
        "design": {
            "eligible_interview_months": "March-October",
            "primary_radius_km": 5,
            "primary_candidate": "B",
            "primary_absolute_threshold_c": 35,
            "heat_reporting_unit": "10 additional post-onset heat days",
            "bootstrap_draws_requested": args.bootstrap_draws,
            "bootstrap_unit": "fixed 0.75-degree spatial block",
        },
        "primary": primary_row,
        "future_ndvi_placebo": placebo,
        "bootstrap": boot,
        "gate": gate,
        "interpretation": (
            "Candidate NDVI mediation passed the frozen gate."
            if gate["mediation_gate_passed"]
            else "Candidate NDVI mediation did not pass the frozen gate; retain parallel response paths."
        ),
    }
    (output / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=lambda value: bool(value) if isinstance(value, np.bool_) else value),
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# CSES absolute-heat NDVI mediation diagnostic\n\n"
        "This diagnostic enforces the sequence post-onset absolute heat exposure, November-February "
        "NDVI, then CSES food consumption observed in March-October. It reports the path-a, path-b, "
        "total, direct, and indirect estimates; a future-NDVI placebo; scale and definition checks; "
        "and a 1,000-spatial-block-bootstrap indirect-effect interval.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved: {output.relative_to(root)}", flush=True)


if __name__ == "__main__":
    main()
