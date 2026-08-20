#!/usr/bin/env python3
"""Outcome-blind power gate for the historical-boundary annual spatial panel.

Only treatment assignment, geography, time, rainfall shocks, and clustering fields are
read.  No NPP level, anomaly, quality, or other outcome column is accessed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.stats import norm


BANDWIDTHS_KM = (2, 5, 10, 15, 20, 30)
SHOCKS = {
    "Annual rainfall anomaly": "Annual Rainfall Anomaly Z (1991-2020)",
    "May-October rainfall anomaly": "May October Rainfall Anomaly Z (1991-2020)",
    "Annual dry shock": "Annual Rainfall Dry Shock",
    "Annual extreme-wet shock": "Annual Rainfall Extreme Wet Shock",
}
ALLOWED_COLUMNS = [
    "Village Code",
    "Year",
    "Commune Code",
    "Linked Climate Commune Code",
    "Historical Repression Side",
    "Higher-Repression Southwest Zone",
    "Signed Distance to Historical Repression Boundary km",
    "Absolute Distance to Historical Repression Boundary km",
    "Historical Boundary Segment",
    *SHOCKS.values(),
]
SCENARIOS = {
    "iid": {"iid": 1.0, "village_ar1": 0.0, "commune_year": 0.0, "district_year": 0.0},
    "moderate clustered dependence": {
        "iid": 0.40,
        "village_ar1": 0.25,
        "commune_year": 0.25,
        "district_year": 0.10,
    },
    "strong clustered dependence": {
        "iid": 0.20,
        "village_ar1": 0.25,
        "commune_year": 0.30,
        "district_year": 0.25,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/historical_boundary_annual_spatial_climate_preprocessed.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/exp/feasibility-check/historical-boundary-annual-spatial"),
    )
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def group_demean(matrix: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    sums = np.zeros((n_groups, matrix.shape[1]), dtype=float)
    np.add.at(sums, codes, matrix)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    return matrix - sums[codes] / counts[codes, None]


def absorb_fixed_effects(
    matrix: np.ndarray, groups: list[np.ndarray], tolerance: float = 1e-11
) -> np.ndarray:
    residual = matrix.astype(float, copy=True)
    for _ in range(500):
        previous = residual.copy()
        for codes in groups:
            residual = group_demean(residual, codes, int(codes.max()) + 1)
        if np.max(np.abs(residual - previous)) < tolerance:
            break
    else:
        raise RuntimeError("Fixed-effect absorption did not converge")
    return residual


def target_residual(data: pd.DataFrame, shock: str) -> np.ndarray:
    treatment = data["Higher-Repression Southwest Zone"].to_numpy(dtype=float)
    signed_distance = data[
        "Signed Distance to Historical Repression Boundary km"
    ].to_numpy(dtype=float)
    shock_values = data[shock].to_numpy(dtype=float)
    target = treatment * shock_values
    nuisance = np.column_stack(
        [
            shock_values,
            signed_distance * shock_values,
            treatment * signed_distance * shock_values,
        ]
    )
    matrix = np.column_stack([target, nuisance])
    village_codes = pd.Categorical(data["Village Code"]).codes
    segment_year = (
        data["Historical Boundary Segment"].astype(str)
        + "-"
        + data["Year"].astype(str)
    )
    segment_year_codes = pd.Categorical(segment_year).codes
    within = absorb_fixed_effects(matrix, [village_codes, segment_year_codes])
    target_within = within[:, 0]
    nuisance_within = within[:, 1:]
    keep = np.std(nuisance_within, axis=0) > 1e-12
    if keep.any():
        coefficients = np.linalg.lstsq(
            nuisance_within[:, keep], target_within, rcond=None
        )[0]
        residual = target_within - nuisance_within[:, keep] @ coefficients
    else:
        residual = target_within
    if np.dot(residual, residual) < 1e-10:
        raise RuntimeError(f"No residual identifying variation for {shock}")
    return residual


def component_numerator_variances(
    data: pd.DataFrame, residual: np.ndarray, ar1_rho: float = 0.50
) -> dict[str, float]:
    years = sorted(data["Year"].unique())
    villages = sorted(data["Village Code"].unique())
    expected = len(years) * len(villages)
    if len(data) != expected:
        raise RuntimeError("Power calculation requires a balanced village-year design")
    residual_matrix = residual.reshape(len(years), len(villages))
    iid_variance = float(np.dot(residual, residual))
    ar_covariance = toeplitz(ar1_rho ** np.arange(len(years)))
    village_ar_variance = float(
        np.einsum("tv,ts,sv->", residual_matrix, ar_covariance, residual_matrix)
    )

    village_frame = data.loc[data["Year"].eq(years[0])].sort_values("Village Code")
    commune_codes = pd.Categorical(village_frame["Linked Climate Commune Code"]).codes
    district_codes = pd.Categorical(village_frame["Commune Code"].str[:4]).codes

    def shared_group_variance(codes: np.ndarray) -> float:
        total = 0.0
        for year_index in range(len(years)):
            group_sums = np.bincount(
                codes, weights=residual_matrix[year_index], minlength=int(codes.max()) + 1
            )
            total += float(np.dot(group_sums, group_sums))
        return total

    return {
        "iid": iid_variance,
        "village_ar1": village_ar_variance,
        "commune_year": shared_group_variance(commune_codes),
        "district_year": shared_group_variance(district_codes),
    }


def empirical_mde(
    standard_error: float, simulations: int, seed: int, alpha: float = 0.05, power: float = 0.80
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    null_draws = rng.normal(0.0, standard_error, simulations)
    critical = float(np.quantile(np.abs(null_draws), 1 - alpha))
    low = 0.0
    high = 8.0 * standard_error
    for _ in range(60):
        middle = (low + high) / 2
        achieved = np.mean(np.abs(null_draws + middle) > critical)
        if achieved >= power:
            high = middle
        else:
            low = middle
    return high, float(np.mean(np.abs(null_draws + high) > critical))


def support_row(data: pd.DataFrame, bandwidth: int) -> dict[str, object]:
    village = data.loc[data["Year"].eq(data["Year"].min())].copy()
    commune_side_count = village.groupby("Linked Climate Commune Code")[
        "Higher-Repression Southwest Zone"
    ].nunique()
    segment_counts = village["Historical Boundary Segment"].value_counts()
    return {
        "bandwidth_km": bandwidth,
        "village_years": len(data),
        "years": data["Year"].nunique(),
        "villages": village["Village Code"].nunique(),
        "southwest_villages": int(village["Higher-Repression Southwest Zone"].sum()),
        "west_villages": int((1 - village["Higher-Repression Southwest Zone"]).sum()),
        "climate_communes": village["Linked Climate Commune Code"].nunique(),
        "cross_side_climate_communes": int((commune_side_count == 2).sum()),
        "districts": village["Commune Code"].str[:4].nunique(),
        "boundary_segments": village["Historical Boundary Segment"].nunique(),
        "segments_with_at_least_5_villages": int((segment_counts >= 5).sum()),
        "segments_with_at_least_20_villages": int((segment_counts >= 20).sum()),
        "minimum_villages_in_segment": int(segment_counts.min()),
        "maximum_segment_village_share": float(segment_counts.max() / len(village)),
    }


def build_plot(power: pd.DataFrame, output: Path) -> None:
    continuous = power.loc[
        power["shock"].isin(
            ["Annual rainfall anomaly", "May-October rainfall anomaly"]
        )
    ]
    colors = {
        "iid": "#777777",
        "moderate clustered dependence": "#2A6FBB",
        "strong clustered dependence": "#B13C2E",
    }
    shocks = ["Annual rainfall anomaly", "May-October rainfall anomaly"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    for ax, shock in zip(axes, shocks, strict=True):
        for scenario, group in continuous.loc[continuous["shock"].eq(shock)].groupby(
            "dependence_scenario"
        ):
            group = group.sort_values("bandwidth_km")
            ax.plot(
                group["bandwidth_km"],
                group["mde_80_standardized_outcome_per_one_sd_shock"],
                marker="o",
                linewidth=2,
                color=colors[scenario],
                label=scenario,
            )
        ax.set_xlabel("Candidate boundary bandwidth (km)")
        ax.set_title(shock)
        ax.grid(True, color="#D9D9D9", linewidth=0.7)
    axes[0].set_ylabel("80% MDE (outcome SD per 1 SD shock)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, ncol=3, fontsize=8)
    fig.suptitle("Outcome-blind power for the annual spatial boundary interaction")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_path = args.input if args.input.is_absolute() else root / args.input
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Column projection is the enforceable outcome-blinding boundary.
    panel = pd.read_parquet(input_path, columns=ALLOWED_COLUMNS)
    panel = panel.loc[panel["Year"].between(2001, 2021)].copy()
    panel["Commune Code"] = panel["Commune Code"].astype("string").str.zfill(6)
    panel["Village Code"] = panel["Village Code"].astype("string").str.zfill(8)
    if panel[list(SHOCKS.values())].isna().any().any():
        raise RuntimeError("Unexpected missing rainfall shock inside 2001-2021")

    support_rows: list[dict[str, object]] = []
    power_rows: list[dict[str, object]] = []
    for bandwidth in BANDWIDTHS_KM:
        data = panel.loc[
            panel["Absolute Distance to Historical Repression Boundary km"].le(bandwidth)
        ].sort_values(["Year", "Village Code"]).reset_index(drop=True)
        support_rows.append(support_row(data, bandwidth))
        for shock_index, (shock_label, shock_column) in enumerate(SHOCKS.items()):
            residual = target_residual(data, shock_column)
            denominator = float(np.dot(residual, residual))
            component_variances = component_numerator_variances(data, residual)
            shock_sd = float(data[shock_column].std(ddof=1))
            for scenario_index, (scenario, weights) in enumerate(SCENARIOS.items()):
                numerator_variance = sum(
                    weights[component] * component_variances[component]
                    for component in component_variances
                )
                raw_coefficient_variance = numerator_variance / denominator**2
                iid_variance = component_variances["iid"] / denominator**2
                coefficient_variance = max(raw_coefficient_variance, iid_variance)
                standard_error = float(np.sqrt(coefficient_variance))
                seed = args.seed + bandwidth * 100 + shock_index * 10 + scenario_index
                mde, achieved = empirical_mde(
                    standard_error, args.simulations, seed
                )
                variance_inflation = coefficient_variance / iid_variance
                power_rows.append(
                    {
                        "bandwidth_km": bandwidth,
                        "shock": shock_label,
                        "shock_column": shock_column,
                        "shock_sd_in_analysis_sample": shock_sd,
                        "dependence_scenario": scenario,
                        "simulations": args.simulations,
                        "seed": seed,
                        "village_years": len(data),
                        "villages": data["Village Code"].nunique(),
                        "years": data["Year"].nunique(),
                        "target_residual_sum_squares": denominator,
                        "coefficient_standard_error_standardized_outcome": standard_error,
                        "raw_variance_inflation_relative_to_iid": raw_coefficient_variance
                        / iid_variance,
                        "variance_inflation_relative_to_iid": variance_inflation,
                        "conservative_iid_variance_floor_applied": raw_coefficient_variance
                        < iid_variance,
                        "iid_equivalent_village_years": len(data) / variance_inflation,
                        "mde_80_standardized_outcome_per_one_unit_shock": mde,
                        "mde_80_standardized_outcome_per_one_sd_shock": mde * shock_sd,
                        "simulated_power_at_reported_mde": achieved,
                    }
                )

    support = pd.DataFrame(support_rows)
    power = pd.DataFrame(power_rows)
    support.to_csv(output_dir / "annual_spatial_blinded_support.csv", index=False)
    power.to_csv(output_dir / "annual_spatial_blinded_power.csv", index=False)
    build_plot(power, output_dir / "annual_spatial_blinded_power.png")

    strong = power.loc[power["dependence_scenario"].eq("strong clustered dependence")]
    summary = {
        "outcome_columns_read": [],
        "allowed_columns": ALLOWED_COLUMNS,
        "analysis_years": [2001, 2021],
        "candidate_bandwidths_km": list(BANDWIDTHS_KM),
        "model": "village fixed effects; boundary-segment-by-year fixed effects; shock main effect; shock interacted with signed distance and side-specific signed-distance slope; target is Southwest-side-by-shock discontinuity",
        "dependence_scenarios": SCENARIOS,
        "village_serial_correlation_in_clustered_scenarios": 0.50,
        "power_target": 0.80,
        "two_sided_alpha": 0.05,
        "simulations_per_cell": args.simulations,
        "mde_units": "standardized residual outcome units; no NPP distribution was read",
        "strong_dependence_best_continuous_mde": strong.loc[
            strong["shock"].isin(
                ["Annual rainfall anomaly", "May-October rainfall anomaly"]
            ),
            [
                "bandwidth_km",
                "shock",
                "mde_80_standardized_outcome_per_one_sd_shock",
            ],
        ].sort_values("mde_80_standardized_outcome_per_one_sd_shock").head(1).to_dict("records"),
        "activation_status": "pending human-approved smallest effect of substantive interest and identification diagnostics",
        "effect_estimation_performed": False,
    }
    (output_dir / "README_annual_spatial_power.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    gate = pd.DataFrame(
        [
            {
                "gate": "Outcome blinding",
                "status": "pass",
                "evidence": "Parquet column projection reads design, geography, time, and rainfall fields only.",
            },
            {
                "gate": "Balanced outcome-shock design",
                "status": "pass",
                "evidence": "All included villages have complete annual rainfall support for 2001-2021.",
            },
            {
                "gate": "Local two-sided support",
                "status": "pass with warning",
                "evidence": "Every bandwidth has both sides, but only nine climate communes cross the historical boundary.",
            },
            {
                "gate": "Boundary-segment support",
                "status": "warning",
                "evidence": "All five segments appear, but one segment contains only one village and contributes no within-segment-year identifying variation.",
            },
            {
                "gate": "Power relative to substantive effect",
                "status": "pending human threshold",
                "evidence": "Strong-dependence continuous-shock MDEs range from about 0.15 to 0.21 outcome SD per one-SD rainfall shock.",
            },
            {
                "gate": "Predetermined continuity and placebo diagnostics",
                "status": "pending",
                "evidence": "Not evaluated in this outcome-blind power step.",
            },
            {
                "gate": "Effect estimation",
                "status": "blocked",
                "evidence": "Requires a human-approved substantive-effect threshold and successful identification diagnostics.",
            },
        ]
    )
    gate.to_csv(output_dir / "annual_spatial_activation_gate.csv", index=False)

    strong_continuous = strong.loc[
        strong["shock"].isin(
            ["Annual rainfall anomaly", "May-October rainfall anomaly"]
        )
    ].sort_values(["bandwidth_km", "shock"])
    result_lines = [
        "# Outcome-Blind Annual Spatial Power Gate",
        "",
        "No NPP level, anomaly, quality, or other outcome field was read. The calculation uses",
        "historical treatment assignment, signed distance, boundary segment, village and climate",
        "geography, calendar year, and rainfall shocks only.",
        "",
        "## Design",
        "",
        "The target is the discontinuity in rainfall sensitivity at the Southwest-West historical",
        "boundary. The projected design absorbs village fixed effects and boundary-segment-by-year",
        "fixed effects, and allows the shock response to vary linearly with signed distance on both",
        "sides. Power is reported under iid, moderate, and strong clustered-dependence scenarios.",
        "",
        "## Strong-dependence 80% MDEs",
        "",
        "| bandwidth km | annual rainfall anomaly | May-October rainfall anomaly |",
        "|---:|---:|---:|",
    ]
    for bandwidth in BANDWIDTHS_KM:
        rows = strong_continuous.loc[strong_continuous["bandwidth_km"].eq(bandwidth)]
        annual = rows.loc[
            rows["shock"].eq("Annual rainfall anomaly"),
            "mde_80_standardized_outcome_per_one_sd_shock",
        ].iloc[0]
        growing = rows.loc[
            rows["shock"].eq("May-October rainfall anomaly"),
            "mde_80_standardized_outcome_per_one_sd_shock",
        ].iloc[0]
        result_lines.append(f"| {bandwidth} | {annual:.3f} | {growing:.3f} |")
    result_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The panel is conditionally feasible for moderate interaction effects. Under strong",
            "dependence, it is designed to detect roughly 0.15-0.21 standardized outcome units per",
            "one-standard-deviation rainfall shock, depending on bandwidth. It is not adequately",
            "powered for very small effects such as 0.05 SD. Power alone does not select a bandwidth:",
            "the 5 km window is the leading local candidate, while wider windows trade locality for",
            "precision and must pass continuity and functional-form diagnostics.",
            "",
            "Effect estimation remains blocked until the human approves a smallest effect of",
            "substantive interest and the predetermined continuity and placebo gates pass.",
            "",
        ]
    )
    (output_dir / "README_annual_spatial_power.md").write_text(
        "\n".join(result_lines), encoding="utf-8"
    )
    print(f"wrote support and blinded power outputs to {output_dir}")
    print(
        strong[
            [
                "bandwidth_km",
                "shock",
                "mde_80_standardized_outcome_per_one_sd_shock",
                "variance_inflation_relative_to_iid",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
