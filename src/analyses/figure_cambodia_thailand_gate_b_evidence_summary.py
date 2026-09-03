#!/usr/bin/env python3
"""Summarize supporting and weakening Gate B LongNTL evidence in one figure."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mj02-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/exp/experiment-design/cambodia-thailand-gate-b"


def main() -> None:
    periods = pd.read_csv(OUT_DIR / "gate_b_longntl_continuous_dose_coefficients.csv")
    sectors = pd.read_csv(OUT_DIR / "gate_b_longntl_sector_and_inner_zone_robustness.csv")
    placebos = pd.read_csv(OUT_DIR / "gate_b_longntl_placebo_site_estimates.csv")
    period_order = [
        "Early pre-period (2000-2004)",
        "Precursor escalation (2008-2010)",
        "Conflict year (2011)",
        "Early recovery (2012-2014)",
        "Medium recovery (2015-2019)",
        "Long recovery (2020-2024)",
    ]
    periods = periods.set_index("Period").loc[period_order].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.35, 1, 1]})

    axis = axes[0]
    x = np.arange(len(periods))
    axis.errorbar(
        x,
        periods["Standardized Estimate"],
        yerr=1.96 * periods["Standardized Standard Error"],
        fmt="o-",
        color="#2C6E9B",
        capsize=3,
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(1.5, color="#777777", linestyle="--", linewidth=0.8)
    axis.set_xticks(x, ["Early pre", "2008-10", "2011", "2012-14", "2015-19", "2020-24"], rotation=30)
    axis.set_ylabel("Standardized LongNTL dose effect")
    axis.set_title("A  Time and recovery")
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1]
    scenarios = [
        "All 2011 event sectors",
        "All sectors excluding 0-5 km",
        "Only Kantharalak district / Preah Vihear temple",
        "Only Phanom Dong Rak district / Ta Kwai and Ta Muen temple area",
    ]
    labels = ["All sectors", "Exclude 0-5 km", "Preah Vihear", "Ta Moan / Ta Krabey"]
    part = sectors.set_index("Scenario").loc[scenarios]
    y = np.arange(len(part))
    axis.errorbar(
        part["Conflict year (2011) Estimate"],
        y,
        xerr=1.96 * part["Conflict year (2011) Standard Error"],
        fmt="o",
        color="#8E3B46",
        capsize=3,
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("2011 standardized dose effect")
    axis.set_title("B  Sector dependence")
    axis.grid(axis="x", alpha=0.2)

    axis = axes[2]
    valid = placebos.loc[placebos["Passes Estimation Support"]].sort_values(
        "Conflict Year Standardized Estimate"
    )
    actual = float(periods.loc[periods["Period"].eq("Conflict year (2011)"), "Standardized Estimate"].iloc[0])
    axis.scatter(
        valid["Conflict Year Standardized Estimate"],
        np.arange(1, len(valid) + 1),
        color="#5B8C5A",
        label="Placebo locations",
    )
    axis.axvline(actual, color="#8E3B46", linewidth=2, label="Actual locations")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("2011 standardized dose effect")
    axis.set_ylabel("Ordered placebo placement")
    axis.set_title("C  Spatial placebo rank")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.2)

    fig.suptitle("Gate B evidence: a localized 2011 activity signal with limited causal support")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate_b_longntl_evidence_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
