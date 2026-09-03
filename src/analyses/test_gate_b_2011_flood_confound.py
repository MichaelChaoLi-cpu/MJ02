#!/usr/bin/env python3
"""Test whether observed 2011 inundation explains the Gate B light decline."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mj02-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from estimate_cambodia_thailand_gate_b_longntl import WEATHER_CONTROLS
from freeze_cambodia_thailand_gate_b_protocol import OUT_DIR
from test_cambodia_thailand_gate_b_placebo_sites import make_weights
from validate_gate_b_with_ccnl_dmsp import (
    REFERENCE_YEAR,
    load_panel,
    scenario_samples,
)


ROOT = Path(__file__).resolve().parents[2]
FLOOD = ROOT / "data/processed/cambodia_national_2011_gfd_flood_exposure_preprocessed.parquet"
FLOOD_SHARE = "2011 Maximum Flooded Share Excluding Permanent Water"
FLOOD_ANY = "2011 Any Satellite Observed Flooding"
CLEAR_SHARE = "Event 3853 Clear Observation Share"
OUTCOME_SPECS = (
    ("LongNTL", "Asinh Annual NPP-VIIRS-like Radiance"),
    ("CCNL", "Asinh CCNL DMSP Corrected DN"),
)
SPECIFICATIONS = (
    "Full-sample baseline",
    "Flood-observed baseline",
    "Flood-adjusted",
    "Observed dry cells",
    "Below 10% flooded",
)
COEFFICIENTS = OUT_DIR / "gate_b_2011_flood_confound_coefficients.csv"
OVERLAP = OUT_DIR / "gate_b_2011_flood_overlap_diagnostics.csv"
SUPPORT = OUT_DIR / "gate_b_2011_flood_confound_support.csv"
FIGURE = OUT_DIR / "gate_b_2011_flood_confound.png"
METADATA = OUT_DIR / "gate_b_2011_flood_confound_metadata.json"
SUMMARIES = OUT_DIR / "gate_b_2011_flood_confound_model_summaries.txt"


def reweight(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    output = sample.copy()
    treated = output["Continuous Conflict Dose"].gt(0).to_numpy()
    if treated.sum() == 0 or (~treated).sum() == 0:
        raise RuntimeError("Flood restriction removed a treatment arm")
    weights, maximum_smd, treated_block_ess = make_weights(output, treated)
    output["Gate B Binary Support Overlap Weight"] = weights
    support = {
        "Grid Cells": len(output),
        "Treated Grid Cells": int(treated.sum()),
        "Control Grid Cells": int((~treated).sum()),
        "Treated Spatial Blocks": output.loc[treated, "Spatial Block ID"].nunique(),
        "Treated 10 km Block ESS": treated_block_ess,
        "Maximum Absolute Weighted SMD": maximum_smd,
    }
    return output, support


def specification_samples(
    base_sample: pd.DataFrame,
    flood: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, dict[str, float]]]:
    merged = base_sample.merge(flood, on="National Grid Cell ID", validate="one_to_one")
    treated = merged["Continuous Conflict Dose"].gt(0)
    base_support = {
        "Grid Cells": len(merged),
        "Treated Grid Cells": int(treated.sum()),
        "Control Grid Cells": int((~treated).sum()),
        "Treated Spatial Blocks": merged.loc[treated, "Spatial Block ID"].nunique(),
        "Treated 10 km Block ESS": np.nan,
        "Maximum Absolute Weighted SMD": np.nan,
    }
    samples: dict[str, tuple[pd.DataFrame, dict[str, float]]] = {
        "Full-sample baseline": (merged, base_support)
    }
    observed = merged.loc[merged[CLEAR_SHARE].gt(0)].copy()
    observed_weighted, observed_support = reweight(observed)
    samples["Flood-observed baseline"] = (observed_weighted, observed_support)
    samples["Flood-adjusted"] = (observed_weighted.copy(), observed_support.copy())
    dry = merged.loc[merged[CLEAR_SHARE].ge(0.8) & merged[FLOOD_SHARE].eq(0)].copy()
    samples["Observed dry cells"] = reweight(dry)
    low_flood = merged.loc[
        merged[CLEAR_SHARE].ge(0.8) & merged[FLOOD_SHARE].lt(0.10)
    ].copy()
    samples["Below 10% flooded"] = reweight(low_flood)
    return samples


def fit_model(frame: pd.DataFrame, outcome: str, flood_adjusted: bool):
    model_frame = frame.set_index(["National Grid Cell ID", "Year"]).sort_index()
    years = model_frame.index.get_level_values("Year").to_numpy()
    dose_parameters = []
    for year in (2011, 2012, 2013):
        column = f"Dose x {year}"
        model_frame[column] = (
            model_frame["Continuous Conflict Dose"] * (years == year).astype(float)
        )
        dose_parameters.append(column)
    exog_columns = dose_parameters.copy()
    if flood_adjusted:
        for year in (2011, 2012, 2013):
            flood_column = f"Flood Share x {year}"
            clear_column = f"Clear Observation x {year}"
            model_frame[flood_column] = model_frame[FLOOD_SHARE] * (years == year).astype(float)
            model_frame[clear_column] = model_frame[CLEAR_SHARE] * (years == year).astype(float)
            exog_columns.extend([flood_column, clear_column])
    exog = model_frame[[*exog_columns, *WEATHER_CONTROLS]].copy()
    exog.insert(0, "Constant", 1.0)
    other_effects = pd.DataFrame(
        {"Sector Year": pd.Categorical(model_frame["Sector Year"]).codes},
        index=model_frame.index,
    )
    clusters = pd.DataFrame(
        {"Spatial Block": pd.Categorical(model_frame["Spatial Block ID"]).codes},
        index=model_frame.index,
    )
    result = PanelOLS(
        model_frame[outcome],
        exog,
        weights=model_frame["Gate B Binary Support Overlap Weight"],
        entity_effects=True,
        other_effects=other_effects,
        drop_absorbed=True,
        check_rank=False,
    ).fit(cov_type="clustered", clusters=clusters)
    return result, dose_parameters


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    finite = values.notna() & weights.notna()
    return float(np.average(values.loc[finite], weights=weights.loc[finite]))


def overlap_diagnostics(sample: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name, group in sample.assign(
        Group=np.where(sample["Continuous Conflict Dose"].gt(0), "Within 60 km", "Frontier control")
    ).groupby("Group", sort=False):
        observed = group[CLEAR_SHARE].gt(0)
        rows.append(
            {
                "Scenario": scenario,
                "Diagnostic": "weighted exposure summary",
                "Group": group_name,
                "Grid Cells": len(group),
                "Weighted Clear-Observed Share": weighted_mean(
                    observed.astype(float), group["Gate B Binary Support Overlap Weight"]
                ),
                "Weighted Any Flood Share among Observed": weighted_mean(
                    group.loc[observed, FLOOD_ANY],
                    group.loc[observed, "Gate B Binary Support Overlap Weight"],
                ),
                "Weighted Mean Maximum Flooded Share among Observed": weighted_mean(
                    group.loc[observed, FLOOD_SHARE],
                    group.loc[observed, "Gate B Binary Support Overlap Weight"],
                ),
                "Dose Coefficient": np.nan,
                "Dose Standard Error": np.nan,
                "Dose p-value": np.nan,
            }
        )

    observed_sample = sample.loc[sample[CLEAR_SHARE].gt(0)].copy()
    observed_sample, _ = reweight(observed_sample)
    sector = pd.get_dummies(
        observed_sample["Border Analysis Sector"], prefix="Sector", drop_first=True, dtype=float
    )
    x = pd.concat(
        [
            pd.Series(1.0, index=observed_sample.index, name="Constant"),
            observed_sample[["Continuous Conflict Dose"]],
            sector,
        ],
        axis=1,
    )
    for outcome_name, outcome in (
        ("maximum flooded share", FLOOD_SHARE),
        ("any observed flooding", FLOOD_ANY),
    ):
        result = sm.WLS(
            observed_sample[outcome],
            x,
            weights=observed_sample["Gate B Binary Support Overlap Weight"],
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": observed_sample["Spatial Block ID"]},
        )
        rows.append(
            {
                "Scenario": scenario,
                "Diagnostic": f"weighted sector-adjusted {outcome_name} regression",
                "Group": "Dose contrast",
                "Grid Cells": len(observed_sample),
                "Weighted Clear-Observed Share": 1.0,
                "Weighted Any Flood Share among Observed": np.nan,
                "Weighted Mean Maximum Flooded Share among Observed": np.nan,
                "Dose Coefficient": float(result.params["Continuous Conflict Dose"]),
                "Dose Standard Error": float(result.bse["Continuous Conflict Dose"]),
                "Dose p-value": float(result.pvalues["Continuous Conflict Dose"]),
            }
        )
    return pd.DataFrame(rows)


def plot_coefficients(coefficients: pd.DataFrame) -> None:
    primary = coefficients.loc[coefficients["Year"].eq(2011)].copy()
    scenarios = ["All 2011 event sectors", "Preah Vihear only"]
    colors = {"LongNTL": "#2C6E9B", "CCNL": "#B54A3A"}
    y = np.arange(len(SPECIFICATIONS))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), sharex=True, sharey=True)
    for axis, scenario in zip(axes, scenarios, strict=True):
        part = primary.loc[primary["Scenario"].eq(scenario)]
        for offset, product in zip((-0.12, 0.12), ("LongNTL", "CCNL"), strict=True):
            product_part = part.loc[part["Product"].eq(product)].set_index("Specification").loc[
                list(SPECIFICATIONS)
            ]
            axis.errorbar(
                product_part["Standardized Estimate"],
                y + offset,
                xerr=1.96 * product_part["Standardized Standard Error"],
                fmt="o",
                color=colors[product],
                capsize=3,
                label=product,
            )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(scenario)
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, list(SPECIFICATIONS))
    axes[0].invert_yaxis()
    axes[0].set_xlabel("2011 dose effect relative to 2010 (baseline-sample SD)")
    axes[1].set_xlabel("2011 dose effect relative to 2010 (baseline-sample SD)")
    axes[1].legend(frameon=False)
    fig.suptitle("Does observed 2011 inundation explain the border-conflict light decline?")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    flood = pd.read_parquet(FLOOD)
    panel = load_panel()
    scenario_data = scenario_samples()
    coefficient_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    overlap_frames: list[pd.DataFrame] = []
    summaries: list[str] = []

    for scenario, (base_sample, base_support_original) in scenario_data.items():
        base_with_flood = base_sample.merge(flood, on="National Grid Cell ID", validate="one_to_one")
        overlap_frames.append(overlap_diagnostics(base_with_flood, scenario))
        samples = specification_samples(base_sample, flood)
        for specification, (sample, support) in samples.items():
            if specification == "Full-sample baseline":
                support["Treated 10 km Block ESS"] = base_support_original[
                    "treated_10km_block_ess"
                ]
                support["Maximum Absolute Weighted SMD"] = base_support_original[
                    "maximum_absolute_weighted_smd"
                ]
            support_rows.append({"Scenario": scenario, "Specification": specification, **support})
        for product, outcome in OUTCOME_SPECS:
            baseline_frame = panel.merge(
                samples["Full-sample baseline"][0][
                    [
                        "National Grid Cell ID",
                        "Border Analysis Sector",
                        "Spatial Block ID",
                        "Gate B Binary Support Overlap Weight",
                        "Continuous Conflict Dose",
                        FLOOD_SHARE,
                        CLEAR_SHARE,
                    ]
                ],
                on="National Grid Cell ID",
                validate="many_to_one",
            )
            baseline_sd = float(
                baseline_frame.loc[baseline_frame["Year"].eq(REFERENCE_YEAR), outcome].std(ddof=1)
            )
            for specification in SPECIFICATIONS:
                sample = samples[specification][0]
                frame = panel.merge(
                    sample[
                        [
                            "National Grid Cell ID",
                            "Border Analysis Sector",
                            "Spatial Block ID",
                            "Gate B Binary Support Overlap Weight",
                            "Continuous Conflict Dose",
                            FLOOD_SHARE,
                            CLEAR_SHARE,
                        ]
                    ],
                    on="National Grid Cell ID",
                    validate="many_to_one",
                )
                frame["Sector Year"] = (
                    frame["Border Analysis Sector"] + "__" + frame["Year"].astype(str)
                )
                required = [outcome, *WEATHER_CONTROLS]
                if specification == "Flood-adjusted":
                    required.extend([FLOOD_SHARE, CLEAR_SHARE])
                frame = frame.dropna(subset=required).copy()
                result, parameters = fit_model(
                    frame,
                    outcome,
                    flood_adjusted=specification == "Flood-adjusted",
                )
                summaries.append(
                    f"\n{'=' * 92}\n{scenario} | {product} | {specification}\n{'=' * 92}\n{result.summary}\n"
                )
                for year, parameter in zip((2011, 2012, 2013), parameters, strict=True):
                    estimate = float(result.params[parameter])
                    standard_error = float(result.std_errors[parameter])
                    coefficient_rows.append(
                        {
                            "Scenario": scenario,
                            "Product": product,
                            "Outcome": outcome,
                            "Specification": specification,
                            "Year": year,
                            "Reference Year": REFERENCE_YEAR,
                            "Estimate": estimate,
                            "Standard Error": standard_error,
                            "Lower 95 CI": estimate - 1.96 * standard_error,
                            "Upper 95 CI": estimate + 1.96 * standard_error,
                            "p-value": float(result.pvalues[parameter]),
                            "Baseline-Sample Reference-Year SD": baseline_sd,
                            "Standardized Estimate": estimate / baseline_sd,
                            "Standardized Standard Error": standard_error / baseline_sd,
                            "Standardized Lower 95 CI": (estimate - 1.96 * standard_error)
                            / baseline_sd,
                            "Standardized Upper 95 CI": (estimate + 1.96 * standard_error)
                            / baseline_sd,
                            "Grid Cells": frame["National Grid Cell ID"].nunique(),
                            "Observations": len(frame),
                        }
                    )

    coefficients = pd.DataFrame(coefficient_rows)
    overlap = pd.concat(overlap_frames, ignore_index=True)
    support = pd.DataFrame(support_rows)
    coefficients.to_csv(COEFFICIENTS, index=False)
    overlap.to_csv(OVERLAP, index=False)
    support.to_csv(SUPPORT, index=False)
    SUMMARIES.write_text("".join(summaries), encoding="utf-8")
    plot_coefficients(coefficients)
    metadata = {
        "status": "candidate Gate B flood-confound experiment",
        "flood_source": "Global Flood Database v1.4 events 3850 and 3853",
        "primary_exposure": FLOOD_SHARE,
        "observation_quality": CLEAR_SHARE,
        "specifications": list(SPECIFICATIONS),
        "flood_adjustment": "event maximum flooded share and event-3853 clear-observation share interacted separately with 2011, 2012, and 2013",
        "dry_cell_rule": "event-3853 clear observation share >= 0.8 and no detected non-permanent-water inundation in either event",
        "low_flood_rule": "event-3853 clear observation share >= 0.8 and maximum flooded share < 0.10",
        "weights": "baseline frozen Gate B weights or re-estimated outcome-independent overlap weights after each flood observation/exclusion rule",
        "fixed_effects": "national 1 km grid cell and one-degree border-sector by year",
        "weather_controls": list(WEATHER_CONTROLS),
        "inference": "10 km spatial-block clustered covariance",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        coefficients.loc[
            coefficients["Year"].eq(2011),
            [
                "Scenario",
                "Product",
                "Specification",
                "Standardized Estimate",
                "Standardized Standard Error",
                "Standardized Lower 95 CI",
                "Standardized Upper 95 CI",
                "p-value",
                "Grid Cells",
            ],
        ].to_string(index=False)
    )
    print("\nFlood overlap diagnostics")
    print(overlap.to_string(index=False))
    print("\nSupport")
    print(support.to_string(index=False))


if __name__ == "__main__":
    main()
