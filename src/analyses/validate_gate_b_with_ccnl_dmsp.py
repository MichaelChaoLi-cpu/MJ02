#!/usr/bin/env python3
"""Validate the Gate B 2011 signal with the independent CCNL DMSP product."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mj02-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from diagnose_cambodia_thailand_common_support import PANEL
from estimate_cambodia_thailand_gate_b_longntl import WEATHER_CONTROLS
from freeze_cambodia_thailand_gate_b_protocol import OUT_DIR, load_features
from test_cambodia_thailand_gate_b_placebo_sites import make_weights
from test_cambodia_thailand_gate_b_sector_robustness import (
    distance_to_sites,
    event_sector_coordinates,
)


ROOT = Path(__file__).resolve().parents[2]
CCNL = ROOT / "data/processed/cambodia_national_ccnl_dmsp_annual_preprocessed.parquet"
COEFFICIENTS = OUT_DIR / "gate_b_ccnl_product_replication_coefficients.csv"
CORRELATIONS = OUT_DIR / "gate_b_ccnl_longntl_overlap_diagnostics.csv"
FIGURE = OUT_DIR / "gate_b_ccnl_product_replication.png"
METADATA = OUT_DIR / "gate_b_ccnl_product_replication_metadata.json"
SUMMARIES = OUT_DIR / "gate_b_ccnl_product_replication_model_summaries.txt"
YEARS = (2010, 2011, 2012, 2013)
REFERENCE_YEAR = 2010
OUTCOME_SPECS = (
    ("LongNTL", "Asinh Annual NPP-VIIRS-like Radiance", "asinh radiance"),
    ("CCNL", "Asinh CCNL DMSP Corrected DN", "asinh corrected DN"),
    ("CCNL", "CCNL DMSP Corrected DN", "corrected DN"),
    ("CCNL", "Any Positive CCNL DMSP Corrected DN", "positive-light indicator"),
)


def scenario_samples() -> dict[str, tuple[pd.DataFrame, dict[str, float]]]:
    base = load_features()
    sectors = event_sector_coordinates()
    preah = [(name, sites) for name, sites in sectors if "Preah Vihear" in name]
    if len(preah) != 1:
        raise RuntimeError(f"Expected one Preah Vihear sector, found {len(preah)}")
    site_sets = {
        "All 2011 event sectors": np.vstack([sites for _, sites in sectors]),
        "Preah Vihear only": preah[0][1],
    }
    output: dict[str, tuple[pd.DataFrame, dict[str, float]]] = {}
    strict_control = base["Candidate All 2008-2011 Nearest Event Distance km"].gt(60).to_numpy()
    for scenario, sites in site_sets.items():
        distance = distance_to_sites(base, sites)
        treated = distance <= 60
        keep = treated | strict_control
        sample = base.loc[keep].copy()
        sample_distance = distance[keep]
        sample_treated = sample_distance <= 60
        weights, maximum_smd, treated_block_ess = make_weights(sample, sample_treated)
        sample["Gate B Binary Support Overlap Weight"] = weights
        sample["Continuous Conflict Dose"] = np.maximum(0, 1 - sample_distance / 60)
        support = {
            "grid_cells": float(len(sample)),
            "treated_grid_cells": float(sample_treated.sum()),
            "treated_spatial_blocks": float(sample.loc[sample_treated, "Spatial Block ID"].nunique()),
            "treated_10km_block_ess": treated_block_ess,
            "maximum_absolute_weighted_smd": maximum_smd,
        }
        output[scenario] = (sample, support)
    return output


def load_panel() -> pd.DataFrame:
    ccnl = pd.read_parquet(CCNL)
    longntl = pd.read_parquet(
        PANEL,
        columns=[
            "National Grid Cell ID",
            "Year",
            "Asinh Annual NPP-VIIRS-like Radiance",
            *WEATHER_CONTROLS,
        ],
    )
    longntl = longntl.loc[longntl["Year"].isin(YEARS)].copy()
    panel = ccnl.merge(
        longntl,
        on=["National Grid Cell ID", "Year"],
        validate="one_to_one",
    )
    return panel


def fit_model(frame: pd.DataFrame, outcome: str):
    model_frame = frame.set_index(["National Grid Cell ID", "Year"]).sort_index()
    interaction_columns = []
    for year in YEARS:
        if year == REFERENCE_YEAR:
            continue
        column = f"Dose x {year}"
        model_frame[column] = (
            model_frame["Continuous Conflict Dose"]
            * (model_frame.index.get_level_values("Year").to_numpy() == year).astype(float)
        )
        interaction_columns.append(column)
    exog = model_frame[[*interaction_columns, *WEATHER_CONTROLS]].copy()
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
    return result, interaction_columns


def overlap_diagnostics(panel: pd.DataFrame, sample: pd.DataFrame, scenario: str) -> pd.DataFrame:
    columns = [
        "National Grid Cell ID",
        "Gate B Binary Support Overlap Weight",
    ]
    frame = panel.merge(sample[columns], on="National Grid Cell ID", validate="many_to_one")
    rows = []
    for year, part in frame.groupby("Year", sort=True):
        rows.append(
            {
                "Scenario": scenario,
                "Year": int(year),
                "Grid Cells": len(part),
                "Pearson Correlation of Asinh Products": part[
                    ["Asinh CCNL DMSP Corrected DN", "Asinh Annual NPP-VIIRS-like Radiance"]
                ].corr(method="pearson").iloc[0, 1],
                "Spearman Correlation of Asinh Products": part[
                    ["Asinh CCNL DMSP Corrected DN", "Asinh Annual NPP-VIIRS-like Radiance"]
                ].corr(method="spearman").iloc[0, 1],
                "CCNL Positive Share": part["Any Positive CCNL DMSP Corrected DN"].mean(),
                "LongNTL Positive Share": (part["Asinh Annual NPP-VIIRS-like Radiance"] > 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_primary(coefficients: pd.DataFrame) -> None:
    primary = coefficients.loc[
        coefficients["Outcome"].isin(
            ["Asinh Annual NPP-VIIRS-like Radiance", "Asinh CCNL DMSP Corrected DN"]
        )
    ].copy()
    scenarios = ["All 2011 event sectors", "Preah Vihear only"]
    products = ["LongNTL", "CCNL"]
    colors = {"LongNTL": "#2C6E9B", "CCNL": "#B54A3A"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)
    for axis, scenario in zip(axes, scenarios, strict=True):
        part = primary.loc[primary["Scenario"].eq(scenario)]
        for offset, product in zip((-0.08, 0.08), products, strict=True):
            product_part = part.loc[part["Product"].eq(product)].set_index("Year").loc[[2011, 2012, 2013]]
            x = np.arange(3) + offset
            axis.errorbar(
                x,
                product_part["Standardized Estimate"],
                yerr=1.96 * product_part["Standardized Standard Error"],
                fmt="o-",
                color=colors[product],
                capsize=3,
                linewidth=1.4,
                label=product,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(np.arange(3), ["2011", "2012", "2013"])
        axis.set_title(scenario)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Dose effect relative to 2010 (2010 cross-sectional SD)")
    axes[1].legend(frameon=False)
    fig.suptitle("Independent DMSP-product replication of the Gate B nighttime-light signal")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    samples = scenario_samples()
    rows: list[dict[str, object]] = []
    summaries: list[str] = []
    diagnostics: list[pd.DataFrame] = []
    support_metadata: dict[str, dict[str, float]] = {}

    for scenario, (sample, support) in samples.items():
        support_metadata[scenario] = support
        diagnostics.append(overlap_diagnostics(panel, sample, scenario))
        model_frame = panel.merge(
            sample[
                [
                    "National Grid Cell ID",
                    "Border Analysis Sector",
                    "Spatial Block ID",
                    "Gate B Binary Support Overlap Weight",
                    "Continuous Conflict Dose",
                ]
            ],
            on="National Grid Cell ID",
            validate="many_to_one",
        )
        model_frame["Sector Year"] = (
            model_frame["Border Analysis Sector"] + "__" + model_frame["Year"].astype(str)
        )
        for product, outcome, scale in OUTCOME_SPECS:
            complete = model_frame.dropna(subset=[outcome, *WEATHER_CONTROLS]).copy()
            pre_sd = float(complete.loc[complete["Year"].eq(REFERENCE_YEAR), outcome].std(ddof=1))
            if not np.isfinite(pre_sd) or pre_sd <= 0:
                raise RuntimeError(f"Invalid 2010 SD for {scenario}: {outcome}")
            result, parameters = fit_model(complete, outcome)
            summaries.append(f"\n{'=' * 88}\n{scenario} | {product} | {outcome}\n{'=' * 88}\n{result.summary}\n")
            for year, parameter in zip((2011, 2012, 2013), parameters, strict=True):
                estimate = float(result.params[parameter])
                standard_error = float(result.std_errors[parameter])
                rows.append(
                    {
                        "Scenario": scenario,
                        "Product": product,
                        "Outcome": outcome,
                        "Outcome Scale": scale,
                        "Year": year,
                        "Reference Year": REFERENCE_YEAR,
                        "Estimate": estimate,
                        "Standard Error": standard_error,
                        "Lower 95 CI": estimate - 1.96 * standard_error,
                        "Upper 95 CI": estimate + 1.96 * standard_error,
                        "p-value": float(result.pvalues[parameter]),
                        "Reference-Year SD": pre_sd,
                        "Standardized Estimate": estimate / pre_sd,
                        "Standardized Standard Error": standard_error / pre_sd,
                        "Standardized Lower 95 CI": (estimate - 1.96 * standard_error) / pre_sd,
                        "Standardized Upper 95 CI": (estimate + 1.96 * standard_error) / pre_sd,
                        "Grid Cells": complete["National Grid Cell ID"].nunique(),
                        "Observations": len(complete),
                    }
                )

    coefficients = pd.DataFrame(rows)
    correlations = pd.concat(diagnostics, ignore_index=True)
    coefficients.to_csv(COEFFICIENTS, index=False)
    correlations.to_csv(CORRELATIONS, index=False)
    SUMMARIES.write_text("".join(summaries), encoding="utf-8")
    plot_primary(coefficients)
    metadata = {
        "status": "interim independent-product validation; native F18 validation remains deferred",
        "source": "CCNL V1, Zenodo DOI 10.5281/zenodo.6644980",
        "years": list(YEARS),
        "reference_year": REFERENCE_YEAR,
        "primary_comparison": "asinh CCNL corrected DN versus asinh LongNTL on identical 2010-2013 support",
        "fixed_effects": "national 1 km grid cell and one-degree border-sector by calendar year",
        "weather_controls": list(WEATHER_CONTROLS),
        "weights": "scenario-specific outcome-independent binary overlap weights using the frozen Gate B predictors",
        "inference": "10 km spatial-block clustered covariance",
        "standardization": "unweighted 2010 cross-sectional outcome SD within each scenario sample",
        "support": support_metadata,
        "limitations": [
            "2010 is the only pre-conflict year in the frozen four-year replication window",
            "CCNL is corrected rather than native F18 stable-lights DN",
            "CCNL has no native cloud-free observation-count band",
            "positive CCNL light is sparse in the affected rural border cells",
        ],
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        coefficients.loc[
            coefficients["Outcome"].isin(
                ["Asinh Annual NPP-VIIRS-like Radiance", "Asinh CCNL DMSP Corrected DN"]
            ),
            [
                "Scenario",
                "Product",
                "Year",
                "Standardized Estimate",
                "Standardized Standard Error",
                "Standardized Lower 95 CI",
                "Standardized Upper 95 CI",
                "p-value",
            ],
        ].to_string(index=False)
    )
    print("\nProduct overlap")
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
