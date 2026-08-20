#!/usr/bin/env python3
"""Create a blinded coverage and quality briefing for VIIRS VNL V2.1.

The script deliberately summarizes the full exported rectangle without loading the
historical boundary or constructing treatment-side indicators. It therefore cannot
reveal boundary discontinuities while candidate-outcome quality is being screened.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parents[2]
INPUT = (
    ROOT
    / "data/raw/independent_validation/viirs_vnl_v21"
    / "viirs_vnl_v21_kampong_speu_boundary_2013_2021.tif"
)
OUTPUT = ROOT / "data/exp/data-briefing/viirs_vnl_v21"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"

YEARS = list(range(2013, 2022))
METRICS = ["average_masked", "median_masked", "cf_cvg", "cvg"]


def quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability)) if values.size else np.nan


def summarize_band(
    values: np.ndarray, valid_mask: np.ndarray, year: int, metric: str, band: int
) -> dict[str, float | int | str]:
    finite = valid_mask & np.isfinite(values)
    observed = values[finite]
    return {
        "band": band,
        "year": year,
        "metric": metric,
        "total_pixels": int(values.size),
        "valid_pixels": int(finite.sum()),
        "valid_share": float(finite.mean()),
        "zero_share_among_valid": float(np.mean(observed == 0)) if observed.size else np.nan,
        "min": float(observed.min()) if observed.size else np.nan,
        "p01": quantile(observed, 0.01),
        "p05": quantile(observed, 0.05),
        "p25": quantile(observed, 0.25),
        "median": quantile(observed, 0.50),
        "mean": float(observed.mean()) if observed.size else np.nan,
        "p75": quantile(observed, 0.75),
        "p95": quantile(observed, 0.95),
        "p99": quantile(observed, 0.99),
        "max": float(observed.max()) if observed.size else np.nan,
        "sd": float(observed.std(ddof=1)) if observed.size > 1 else np.nan,
    }


def make_coverage_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), constrained_layout=True)

    data = summary.loc[summary["metric"] == "average_masked"]
    axes[0].plot(
        data["year"],
        100 * (1 - data["zero_share_among_valid"]),
        marker="o",
        color="#1f4e79",
    )
    axes[0].set(title="Pixels with nonzero radiance", xlabel="Year", ylabel="Nonzero pixels (%)")
    axes[0].set_ylim(0, 12)
    axes[0].grid(alpha=0.25)

    for metric, color, label in [
        ("cf_cvg", "#38761d", "Cloud-free observations"),
        ("cvg", "#674ea7", "Total observations"),
    ]:
        data = summary.loc[summary["metric"] == metric]
        axes[1].plot(data["year"], data["median"], marker="o", color=color, label=label)
    axes[1].set(title="Annual observation support", xlabel="Year", ylabel="Median observations per pixel")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xticks(YEARS)
        axis.tick_params(axis="x", rotation=45)

    fig.savefig(FIGURES / "viirs_vnl_v21_coverage_by_year.png", dpi=220)
    plt.close(fig)


def make_distribution_figure(dataset: rasterio.io.DatasetReader) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(YEARS)))
    bins = np.linspace(0, np.log1p(30), 70)

    for index, (year, color) in enumerate(zip(YEARS, colors, strict=True)):
        band = index * len(METRICS) + 1
        array = dataset.read(band, masked=True)
        values = np.asarray(array.compressed(), dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        axis.hist(
            np.log1p(values),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.1,
            alpha=0.85,
            color=color,
            label=str(year),
        )

    axis.set(
        title="Annual VIIRS radiance distributions (positive pixels only)",
        xlabel="log(1 + annual mean radiance)",
        ylabel="Density",
    )
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=3, frameon=False, title="Year")
    fig.savefig(FIGURES / "viirs_vnl_v21_radiance_distributions.png", dpi=220)
    plt.close(fig)


def write_readme(summary: pd.DataFrame, dataset: rasterio.io.DatasetReader) -> None:
    radiance = summary[summary["metric"].isin(["average_masked", "median_masked"])]
    cloud_free = summary[summary["metric"] == "cf_cvg"]
    bounds = dataset.bounds
    lines = [
        "# VIIRS VNL V2.1 blinded data briefing",
        "",
        "This briefing screens coverage and measurement quality only. It does not load the",
        "historical repression boundary, label treatment sides, choose a bandwidth, or estimate",
        "any treatment effect.",
        "",
        "## Raster structure",
        "",
        f"- Years: {YEARS[0]}-{YEARS[-1]} ({len(YEARS)} annual layers).",
        f"- Grid: {dataset.width} x {dataset.height} pixels; {dataset.count} Float32 bands.",
        f"- CRS: {dataset.crs}; bounds: ({bounds.left:.6f}, {bounds.bottom:.6f}, "
        f"{bounds.right:.6f}, {bounds.top:.6f}).",
        "- Band order within each year: average_masked, median_masked, cf_cvg, cvg.",
        "",
        "## Coverage screen",
        "",
        f"- Radiance valid-pixel shares range from {100 * radiance['valid_share'].min():.2f}% "
        f"to {100 * radiance['valid_share'].max():.2f}% across years and radiance summaries.",
        f"- Nonzero annual-mean radiance covers "
        f"{100 * (1 - summary.loc[summary['metric'] == 'average_masked', 'zero_share_among_valid'].max()):.2f}% "
        f"to {100 * (1 - summary.loc[summary['metric'] == 'average_masked', 'zero_share_among_valid'].min()):.2f}% "
        "of pixels across years.",
        f"- The median annual cloud-free count ranges from {cloud_free['median'].min():.1f} "
        f"to {cloud_free['median'].max():.1f} observations per pixel.",
        "- Zero radiance is common and substantively important in this predominantly rural area;",
        "  the distribution figure conditions on positive pixels only for legibility.",
        "- Negative radiance can occur after background correction and is retained in tables.",
        "",
        "## Outputs",
        "",
        "- `tables/band_dictionary.csv`: deterministic mapping from GeoTIFF band to year and metric.",
        "- `tables/band_quality_summary.csv`: coverage and distribution statistics for every band.",
        "- `figures/viirs_vnl_v21_coverage_by_year.png`: valid-pixel and observation-count diagnostics.",
        "- `figures/viirs_vnl_v21_radiance_distributions.png`: full-rectangle radiance distributions.",
        "",
        "These diagnostics are exploratory and are not final manuscript evidence.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    dictionary: list[dict[str, int | str]] = []

    with rasterio.open(INPUT) as dataset:
        expected_bands = len(YEARS) * len(METRICS)
        if dataset.count != expected_bands:
            raise ValueError(f"Expected {expected_bands} bands, found {dataset.count}")

        for year_index, year in enumerate(YEARS):
            for metric_index, metric in enumerate(METRICS):
                band = year_index * len(METRICS) + metric_index + 1
                array = dataset.read(band, masked=True)
                values = np.asarray(array.data, dtype=float)
                valid_mask = ~np.ma.getmaskarray(array)
                rows.append(summarize_band(values, valid_mask, year, metric, band))
                dictionary.append({"band": band, "year": year, "metric": metric})

        summary = pd.DataFrame(rows)
        pd.DataFrame(dictionary).to_csv(TABLES / "band_dictionary.csv", index=False)
        summary.to_csv(TABLES / "band_quality_summary.csv", index=False, float_format="%.8g")
        make_coverage_figure(summary)
        make_distribution_figure(dataset)
        write_readme(summary, dataset)

    print(f"Read 1 raster with {len(rows)} band-year metrics")
    print(f"Wrote briefing to {OUTPUT}")


if __name__ == "__main__":
    main()
