#!/usr/bin/env python3
"""Outcome-blind feasibility and power gate for temperature-shock extensions.

The script deliberately projects only treatment, geography, time, shock, and
outcome-*availability* fields from the processed panels. It never reads NPP or EVI
values. The calculations therefore determine whether the proposed tests are
identifiable and sufficiently powered before any temperature-response coefficient
is estimated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


ANNUAL_INPUT = Path(
    "data/processed/historical_boundary_annual_local_temperature_analysis_preprocessed.parquet"
)
HF_INPUT = Path(
    "data/processed/historical_boundary_16day_climate_vegetation_temperature_preprocessed.parquet"
)
OUTPUT_DIR = Path("data/exp/feasibility-check/temperature-extension")
YEARS = (2001, 2021)
SESOI = 0.20

ANNUAL_SHOCKS = {
    "Hot days": "May October Hot Day Count Anomaly Z",
    "Heatwave days": "May October Heatwave Day Count Anomaly Z",
    "Hot nights": "May October Hot Night Count Anomaly Z",
    "Compound hot-dry": "May October Compound Hot-Dry Intensity",
}
HF_SHOCKS = {
    "Hot days": "Interval Hot Day Count Anomaly Z",
    "Heatwave days": "Interval Heatwave Day Count Anomaly Z",
    "Hot nights": "Interval Hot Night Count Anomaly Z",
    "Compound hot-dry": "Compound Hot-Dry Intensity",
}

ANNUAL_COLUMNS = [
    "Village Code",
    "Year",
    "Commune Code",
    "Linked Climate Commune Code",
    "Historical Repression Side",
    "Higher-Repression Southwest Zone",
    "Signed Distance to Historical Repression Boundary km",
    "Absolute Distance to Historical Repression Boundary km",
    "Historical Boundary Segment",
    "NPP Complete 2001-2020 Baseline",
    *ANNUAL_SHOCKS.values(),
    "May October Cold Night Count Anomaly Z",
    "May October Hot Day Intensity",
    "Temperature Grid May October Dry Rainfall Intensity",
]
HF_COLUMNS = [
    "Village Code",
    "Composite Date",
    "Year",
    "Linked Climate Commune Code",
    "CHIRPS Cell ID",
    "Historical Repression Side",
    "Higher-Repression Southwest Zone",
    "Signed Distance to Historical Repression Boundary km",
    "Absolute Distance to Historical Repression Boundary km",
    "Historical Boundary Segment",
    "Cross-Side CHIRPS Cell",
    "EVI Valid Pixel Count",
    "Required Valid Pixel Count",
    "EVI Slot SD 2001-2020",
    "EVI Slot Valid Years 2001-2020",
    *HF_SHOCKS.values(),
    "Interval Cold Night Count Anomaly Z",
    "Hot Day Intensity",
    "Dry Rainfall Intensity",
]

ANNUAL_SCENARIOS = {
    "iid": {"iid": 1.00, "village_ar1": 0.00, "commune_time": 0.00, "district_time": 0.00},
    "moderate clustered dependence": {
        "iid": 0.40,
        "village_ar1": 0.25,
        "commune_time": 0.25,
        "district_time": 0.10,
    },
    "strong clustered dependence": {
        "iid": 0.20,
        "village_ar1": 0.25,
        "commune_time": 0.30,
        "district_time": 0.25,
    },
}
HF_SCENARIOS = {
    "iid": {"iid": 1.00, "village_ar1": 0.00, "side_cell_time": 0.00},
    "moderate clustered dependence": {
        "iid": 0.40,
        "village_ar1": 0.30,
        "side_cell_time": 0.30,
    },
    "strong clustered dependence": {
        "iid": 0.20,
        "village_ar1": 0.35,
        "side_cell_time": 0.45,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def categorical_codes(values: pd.Series) -> np.ndarray:
    return pd.Categorical(values).codes.astype(int)


def group_demean(matrix: np.ndarray, codes: np.ndarray) -> np.ndarray:
    n_groups = int(codes.max()) + 1
    sums = np.zeros((n_groups, matrix.shape[1]), dtype=float)
    np.add.at(sums, codes, matrix)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    return matrix - sums[codes] / counts[codes, None]


def absorb_fixed_effects(
    matrix: np.ndarray, groups: list[np.ndarray], tolerance: float = 1e-10
) -> np.ndarray:
    residual = matrix.astype(float, copy=True)
    for _ in range(1000):
        previous = residual.copy()
        for codes in groups:
            residual = group_demean(residual, codes)
        if np.max(np.abs(residual - previous)) < tolerance:
            return residual
    raise RuntimeError("Fixed-effect absorption did not converge")


def identifying_residual(
    data: pd.DataFrame,
    shock: str,
    fe_columns: list[str],
    compound: bool,
    hot_component: str,
    dry_component: str,
) -> np.ndarray:
    treatment = data["Higher-Repression Southwest Zone"].to_numpy(float)
    distance = data["Signed Distance to Historical Repression Boundary km"].to_numpy(float)
    shock_value = data[shock].to_numpy(float)
    target = treatment * shock_value

    if compound:
        hot = data[hot_component].to_numpy(float)
        dry = data[dry_component].to_numpy(float)
        base = [hot, dry, shock_value]
        nuisance = np.column_stack(
            base
            + [treatment * x for x in base]
            + [distance * x for x in base]
            + [treatment * distance * x for x in base]
        )
        # The target T*C is the sixth column in this hierarchy and is removed from nuisance.
        nuisance = np.delete(nuisance, 5, axis=1)
    else:
        nuisance = np.column_stack(
            [shock_value, distance * shock_value, treatment * distance * shock_value]
        )

    matrix = np.column_stack([target, nuisance])
    groups = [categorical_codes(data[column]) for column in fe_columns]
    within = absorb_fixed_effects(matrix, groups)
    target_within, nuisance_within = within[:, 0], within[:, 1:]
    keep = np.std(nuisance_within, axis=0) > 1e-12
    residual = target_within
    if keep.any():
        coefficient = np.linalg.lstsq(nuisance_within[:, keep], target_within, rcond=None)[0]
        residual = target_within - nuisance_within[:, keep] @ coefficient
    if float(np.dot(residual, residual)) < 1e-9:
        raise RuntimeError(f"No identifying variation remains for {shock}")
    return residual


def shared_group_variance(residual: np.ndarray, group: pd.Series) -> float:
    codes = categorical_codes(group)
    sums = np.bincount(codes, weights=residual, minlength=int(codes.max()) + 1)
    return float(np.dot(sums, sums))


def serial_variance(
    data: pd.DataFrame, residual: np.ndarray, time_column: str, rho: float = 0.50
) -> float:
    total = 0.0
    work = data[["Village Code", time_column]].copy()
    work["residual"] = residual
    for _, group in work.groupby("Village Code", observed=True, sort=False):
        values = group.sort_values(time_column)["residual"].to_numpy(float)
        # O(n) AR(1) quadratic form: diagonal plus lagged cross-products.
        value = float(np.dot(values, values))
        for lag in range(1, len(values)):
            contribution = float(np.dot(values[:-lag], values[lag:]))
            value += 2.0 * (rho**lag) * contribution
            if rho**lag < 1e-12:
                break
        total += value
    return total


def power_rows(
    data: pd.DataFrame,
    design: str,
    specification: str,
    shocks: dict[str, str],
    fe_columns: list[str],
    scenarios: dict[str, dict[str, float]],
    time_column: str,
    hot_component: str,
    dry_component: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, column in shocks.items():
        compound = label == "Compound hot-dry"
        residual = identifying_residual(
            data, column, fe_columns, compound, hot_component, dry_component
        )
        denominator = float(np.dot(residual, residual))
        components = {
            "iid": denominator,
            "village_ar1": serial_variance(data, residual, time_column),
        }
        if design == "annual":
            components["commune_time"] = shared_group_variance(
                residual,
                data["Linked Climate Commune Code"].astype(str)
                + "|"
                + data[time_column].astype(str),
            )
            components["district_time"] = shared_group_variance(
                residual,
                data["Commune Code"].astype(str).str.zfill(6).str[:4]
                + "|"
                + data[time_column].astype(str),
            )
        else:
            components["side_cell_time"] = shared_group_variance(
                residual,
                data["CHIRPS Cell ID"].astype(str)
                + "|"
                + data["Higher-Repression Southwest Zone"].astype(str)
                + "|"
                + data[time_column].astype(str),
            )

        iid_variance = components["iid"] / denominator**2
        shock_sd = float(data[column].std(ddof=1))
        for scenario, weights in scenarios.items():
            raw = sum(weights[key] * components[key] for key in components) / denominator**2
            variance = max(raw, iid_variance)
            se = float(np.sqrt(variance))
            mde_unit = float((norm.ppf(0.975) + norm.ppf(0.80)) * se)
            mde_sd = mde_unit * shock_sd
            rows.append(
                {
                    "design": design,
                    "specification": specification,
                    "shock": label,
                    "shock_column": column,
                    "dependence_scenario": scenario,
                    "rows": len(data),
                    "villages": data["Village Code"].nunique(),
                    "time_periods": data[time_column].nunique(),
                    "shock_sd": shock_sd,
                    "identifying_residual_sum_squares": denominator,
                    "standard_error_standardized_outcome_per_unit_shock": se,
                    "mde_80_standardized_outcome_per_unit_shock": mde_unit,
                    "mde_80_standardized_outcome_per_one_sd_shock": mde_sd,
                    "sesoi_standardized_outcome_per_one_sd_shock": SESOI,
                    "power_gate_pass": bool(mde_sd <= SESOI),
                    "raw_variance_inflation_relative_to_iid": raw / iid_variance,
                    "conservative_iid_floor_applied": bool(raw < iid_variance),
                }
            )
    return rows


def annual_support(data: pd.DataFrame, specification: str) -> dict[str, object]:
    village = data.drop_duplicates("Village Code")
    cross = village.groupby("Linked Climate Commune Code", observed=True)[
        "Higher-Repression Southwest Zone"
    ].nunique()
    return {
        "design": "annual",
        "specification": specification,
        "rows": len(data),
        "years": data["Year"].nunique(),
        "villages": village["Village Code"].nunique(),
        "southwest_villages": int(village["Higher-Repression Southwest Zone"].sum()),
        "west_villages": int((1 - village["Higher-Repression Southwest Zone"]).sum()),
        "climate_cells": data["Linked Climate Commune Code"].nunique(),
        "cross_side_climate_cells": int((cross == 2).sum()),
        "boundary_segments": data["Historical Boundary Segment"].nunique(),
    }


def hf_support(data: pd.DataFrame, specification: str) -> dict[str, object]:
    village = data.drop_duplicates("Village Code")
    cross = village.groupby("CHIRPS Cell ID", observed=True)[
        "Higher-Repression Southwest Zone"
    ].nunique()
    return {
        "design": "16-day EVI",
        "specification": specification,
        "rows": len(data),
        "years": data["Year"].nunique(),
        "villages": village["Village Code"].nunique(),
        "southwest_villages": int(village["Higher-Repression Southwest Zone"].sum()),
        "west_villages": int((1 - village["Higher-Repression Southwest Zone"]).sum()),
        "dates": data["Composite Date"].nunique(),
        "climate_cells": data["CHIRPS Cell ID"].nunique(),
        "cross_side_climate_cells": int((cross == 2).sum()),
        "boundary_segments": data["Historical Boundary Segment"].nunique(),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    annual = pd.read_parquet(root / ANNUAL_INPUT, columns=ANNUAL_COLUMNS)
    annual = annual.loc[
        annual["Year"].between(*YEARS)
        & annual["Absolute Distance to Historical Repression Boundary km"].le(5)
        & annual["NPP Complete 2001-2020 Baseline"].eq(1)
    ].copy()
    annual["Village Code"] = annual["Village Code"].astype("string").str.zfill(8)
    annual["Commune Code"] = annual["Commune Code"].astype("string").str.zfill(6)
    annual["segment_year"] = (
        annual["Historical Boundary Segment"].astype(str) + "|" + annual["Year"].astype(str)
    )
    annual["commune_year"] = (
        annual["Linked Climate Commune Code"].astype(str) + "|" + annual["Year"].astype(str)
    )
    required_annual = list(ANNUAL_SHOCKS.values()) + [
        "May October Hot Day Intensity",
        "Temperature Grid May October Dry Rainfall Intensity",
    ]
    annual = annual.dropna(subset=required_annual).sort_values(["Village Code", "Year"])

    cross_communes = (
        annual.drop_duplicates("Village Code")
        .groupby("Linked Climate Commune Code", observed=True)["Higher-Repression Southwest Zone"]
        .nunique()
    )
    cross_communes = set(cross_communes[cross_communes.eq(2)].index)
    annual_confirmation = annual.loc[
        annual["Linked Climate Commune Code"].isin(cross_communes)
    ].copy()

    annual_specs = {
        "main 5 km": (annual, ["Village Code", "segment_year"]),
        "within-climate-commune confirmation": (
            annual_confirmation,
            ["Village Code", "segment_year", "commune_year"],
        ),
    }
    support_rows: list[dict[str, object]] = []
    all_power: list[dict[str, object]] = []
    for specification, (frame, effects) in annual_specs.items():
        support_rows.append(annual_support(frame, specification))
        all_power.extend(
            power_rows(
                frame,
                "annual",
                specification,
                ANNUAL_SHOCKS,
                effects,
                ANNUAL_SCENARIOS,
                "Year",
                "May October Hot Day Intensity",
                "Temperature Grid May October Dry Rainfall Intensity",
            )
        )

    hf = pd.read_parquet(root / HF_INPUT, columns=HF_COLUMNS)
    hf["Composite Date"] = pd.to_datetime(hf["Composite Date"])
    # Reconstruct EVI outcome availability without loading Mean EVI or any anomaly value.
    hf["evi_available"] = (
        hf["EVI Valid Pixel Count"].ge(hf["Required Valid Pixel Count"])
        & hf["EVI Slot Valid Years 2001-2020"].ge(10)
        & hf["EVI Slot SD 2001-2020"].gt(0)
    )
    hf = hf.loc[
        hf["Year"].between(*YEARS)
        & hf["Absolute Distance to Historical Repression Boundary km"].le(5)
        & hf["Cross-Side CHIRPS Cell"].eq(1)
        & hf["evi_available"]
    ].copy()
    hf["Village Code"] = hf["Village Code"].astype("string").str.zfill(8)
    required_hf = list(HF_SHOCKS.values()) + ["Hot Day Intensity", "Dry Rainfall Intensity"]
    hf = hf.dropna(subset=required_hf)
    hf["cell_date"] = hf["CHIRPS Cell ID"].astype(str) + "|" + hf["Composite Date"].astype(str)
    hf["cell_commune_date"] = (
        hf["CHIRPS Cell ID"].astype(str)
        + "|"
        + hf["Linked Climate Commune Code"].astype(str)
        + "|"
        + hf["Composite Date"].astype(str)
    )
    cell_commune = (
        hf.drop_duplicates("Village Code")
        .groupby(["CHIRPS Cell ID", "Linked Climate Commune Code"], observed=True)[
            "Higher-Repression Southwest Zone"
        ]
        .nunique()
    )
    cross_cell_commune = set(cell_commune[cell_commune.eq(2)].index)
    pair_index = pd.MultiIndex.from_frame(hf[["CHIRPS Cell ID", "Linked Climate Commune Code"]])
    hf_confirmation = hf.loc[pair_index.isin(cross_cell_commune)].copy()

    hf_specs = {
        "main cross-side climate-cell": (hf, ["Village Code", "cell_date"]),
        "within-climate-commune confirmation": (
            hf_confirmation,
            ["Village Code", "cell_commune_date"],
        ),
    }
    for specification, (frame, effects) in hf_specs.items():
        frame = frame.sort_values(["Village Code", "Composite Date"]).reset_index(drop=True)
        support_rows.append(hf_support(frame, specification))
        all_power.extend(
            power_rows(
                frame,
                "16-day EVI",
                specification,
                HF_SHOCKS,
                effects,
                HF_SCENARIOS,
                "Composite Date",
                "Hot Day Intensity",
                "Dry Rainfall Intensity",
            )
        )

    support = pd.DataFrame(support_rows)
    power = pd.DataFrame(all_power)
    support.to_csv(output_dir / "temperature_support.csv", index=False)
    power.to_csv(output_dir / "temperature_blinded_power.csv", index=False)

    shock_frames = []
    for label, frame, shocks in [
        ("annual", annual, ANNUAL_SHOCKS),
        ("16-day EVI", hf, HF_SHOCKS),
    ]:
        correlation = frame[list(shocks.values())].corr()
        for row_name in correlation.index:
            for column_name in correlation.columns:
                shock_frames.append(
                    {
                        "design": label,
                        "shock_1": row_name,
                        "shock_2": column_name,
                        "correlation": correlation.loc[row_name, column_name],
                    }
                )
    pd.DataFrame(shock_frames).to_csv(output_dir / "shock_correlation.csv", index=False)

    strong = power.loc[power["dependence_scenario"].eq("strong clustered dependence")]
    gate_rows = []
    for (design, shock), group in strong.groupby(["design", "shock"], observed=True):
        records = group.set_index("specification")
        primary_name = "main 5 km" if design == "annual" else "main cross-side climate-cell"
        confirmation_name = "within-climate-commune confirmation"
        primary_mde = float(records.loc[primary_name, "mde_80_standardized_outcome_per_one_sd_shock"])
        confirmation_mde = float(
            records.loc[confirmation_name, "mde_80_standardized_outcome_per_one_sd_shock"]
        )
        gate_rows.append(
            {
                "design": design,
                "shock": shock,
                "predetermined_sesoi": SESOI,
                "primary_strong_dependence_mde": primary_mde,
                "confirmation_strong_dependence_mde": confirmation_mde,
                "activation_status": "pass"
                if max(primary_mde, confirmation_mde) <= SESOI
                else "fail",
                "effect_estimation_performed": False,
            }
        )
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(output_dir / "temperature_activation_gate.csv", index=False)

    summary = {
        "status": "outcome-blind feasibility gate complete",
        "analysis_years": list(YEARS),
        "bandwidth_km": 5,
        "predetermined_sesoi": SESOI,
        "outcome_values_read": [],
        "outcome_availability_fields_read": [
            "NPP Complete 2001-2020 Baseline",
            "EVI Valid Pixel Count",
            "Required Valid Pixel Count",
            "EVI Slot SD 2001-2020",
            "EVI Slot Valid Years 2001-2020",
        ],
        "effect_estimation_performed": False,
        "annual_dependence_scenarios": ANNUAL_SCENARIOS,
        "high_frequency_dependence_scenarios": HF_SCENARIOS,
        "compound_hierarchy": "hot, dry, and compound main terms plus all side and signed-distance lower-order interactions",
        "activation_rule": "Both the main and within-climate-commune strong-dependence 80% MDE must be no larger than 0.20 standardized outcome units per one-SD shock.",
        "activated_tests": gate.loc[gate["activation_status"].eq("pass"), ["design", "shock"]].to_dict("records"),
        "blocked_tests": gate.loc[gate["activation_status"].eq("fail"), ["design", "shock"]].to_dict("records"),
    }
    (output_dir / "README.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Outcome-Blind Temperature Extension Gate",
        "",
        "No NPP or EVI values were read and no response coefficient was estimated. Outcome",
        "availability was reconstructed from pre-existing completeness and vegetation-quality",
        "fields. The fixed smallest effect of substantive interest is 0.20 outcome SD per",
        "one-SD climate shock, matching the current AnaSOP equivalence threshold.",
        "",
        "## Strong-dependence 80% minimum detectable effects",
        "",
        "| design | specification | shock | MDE | gate |",
        "|---|---|---|---:|---|",
    ]
    for row in strong.sort_values(["design", "specification", "shock"]).itertuples(index=False):
        lines.append(
            f"| {row.design} | {row.specification} | {row.shock} | "
            f"{row.mde_80_standardized_outcome_per_one_sd_shock:.3f} | "
            f"{'pass' if row.power_gate_pass else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "A shock family is activated only when both its main and within-climate-commune",
            "specifications pass. Failure blocks effect estimation for that design-shock pair; it",
            "does not imply that the substantive effect is zero.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(support.to_string(index=False))
    print("\nActivation gate")
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
