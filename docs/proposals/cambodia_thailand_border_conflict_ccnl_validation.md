# Independent CCNL validation of the Cambodia–Thailand Gate B signal

Status: **directional independent-product replication; causal promotion threshold not met**  
Analysis date: **2026-08-22**

## 1. Purpose

The original Gate B result used the CNN-harmonized LongNTL series and found a
sharp 2011 nighttime-activity decline near the conflict locations, concentrated
around Preah Vihear. This validation asks a narrower measurement question:
does the 2011 spatial pattern survive in a separately processed DMSP-OLS-derived
product?

The validation uses the public CCNL V1 dataset (Zenodo DOI
`10.5281/zenodo.6644980`), which was published as a data descriptor in
*Scientific Data*. CCNL corrects the DMSP stable-light series for interannual
inconsistency, saturation, and blooming. It is independent of the current
LongNTL processing chain, but it is not native F18 data.

## 2. Frozen comparison

Exact source pixels for 2010–2013 were remotely read from the public Zenodo
GeoTIFFs, cropped with a guard band around Cambodia, and area-averaged to the
stable national EPSG:32648 1 km grid. The model then holds the following fixed
across products:

- the 60 km continuous conflict-distance dose;
- the outcome-independent Gate B overlap weights;
- grid-cell and one-degree border-sector-by-year fixed effects;
- annual precipitation, extreme-rainfall, dry-spell, and maximum-temperature
  controls;
- 10 km spatial-block clustered covariance;
- 2010 as the reference year.

LongNTL is re-estimated on exactly the same 2010–2013 window and support so that
the product comparison does not mix different time windows or samples.

## 3. Main results

### Asinh nighttime-light outcomes

| Scenario | Product | 2011 estimate (2010 SD) | 95% CI | 2012 estimate | 2013 estimate |
|---|---|---:|---:|---:|---:|
| All 2011 event sectors | LongNTL | -0.641 | [-1.272, -0.010] | -0.249 | 0.020 |
| All 2011 event sectors | CCNL | -0.290 | [-0.637, 0.058] | -0.168 | -0.147 |
| Preah Vihear only | LongNTL | -0.595 | [-1.091, -0.099] | -0.240 | 0.037 |
| Preah Vihear only | CCNL | -0.294 | [-0.687, 0.098] | -0.118 | -0.181 |

The independent CCNL estimate has the same negative direction and the same
2011 timing as LongNTL, but its standardized magnitude is about one-half as
large and its clustered 95% confidence interval includes zero.

### Alternative CCNL scales

For Preah Vihear in 2011, the corrected-DN level estimate is -0.095
(SE 0.074), while the linear-probability estimate for any positive CCNL light
is -0.122 (SE 0.075). Both remain negative, so the asinh result is not solely a
consequence of the transformation or a few extremely bright cells. Neither
estimate is conventionally significant.

CCNL is sparse in this rural frontier: only about 3.2% of the Preah validation
sample has positive CCNL light in 2010. This low signal density is a substantive
measurement limitation and explains part of the wider uncertainty.

## 4. Spatial-placebo evidence

The exact frozen two-sector translation universe contains 11 eligible placebo
locations. All 11 pass the predeclared balance and effective-block thresholds.
The actual CCNL 2011 estimate (-0.290 SD) is more negative than every placebo
estimate; the closest placebo is -0.258 SD. The finite-sample one-sided rank
p-value is therefore:

\[
p = \frac{1+0}{1+11}=0.0833.
\]

This is the smallest p-value available with 11 placebos. It improves on the
LongNTL spatial-placebo p-value of 0.167, but the short frontier does not permit
a conventional 0.05 design-based rejection.

## 5. Product overlap

Across the frozen Gate B sample, the annual Pearson correlation between the two
asinh products ranges from 0.390 to 0.727; the 2011 correlation is 0.727
(Spearman 0.750). The products therefore share meaningful spatial information
without being mechanically identical.

## 6. What this validates—and what it does not

The validation weakens the hypothesis that the 2011 decline was manufactured
only by the CNN-harmonized LongNTL product. A separately corrected DMSP product
finds a smaller but directionally consistent decline, and the actual conflict
placement is the most negative location in the frozen placebo set.

It still does not prove that armed conflict caused the decline. CCNL and LongNTL
can both respond to the same contemporaneous local shock, including flooding,
displacement, or access disruption. The validation has only one pre-conflict
year, CCNL does not supply native F18 cloud-free observation counts, and the
affected area has very sparse stable lighting.

The appropriate disposition is therefore:

- upgrade the measurement assessment from **single-product signal** to
  **directionally replicated independent-product signal**;
- retain Gate B as a candidate diagnostic rather than a causal main result;
- leave Gate C and the current AnaSOP unchanged;
- obtain native F18 stable lights and cloud-free coverage when EOG login is
  available;
- treat observed 2011 inundation as tested and not explanatory of the signal;
  see `docs/proposals/cambodia_thailand_border_conflict_flood_confound.md`;
- prioritize independently geocoded displacement, road-closure, or damaged-
  settlement evidence as the next unattended test of contemporaneous local
  disruption.

## 7. Reproducible artifacts

- Acquisition: `src/preprocessing/acquire_ccnl_dmsp_cambodia.py`
- Preprocessing: `src/preprocessing/preprocess_ccnl_dmsp_cambodia.py`
- Product replication: `src/analyses/validate_gate_b_with_ccnl_dmsp.py`
- Frozen spatial placebos: `src/analyses/test_gate_b_ccnl_placebo_sites.py`
- Processed panel: `data/processed/cambodia_national_ccnl_dmsp_annual_preprocessed.parquet`
- Coefficients: `data/exp/experiment-design/cambodia-thailand-gate-b/gate_b_ccnl_product_replication_coefficients.csv`
- Spatial inference: `data/exp/experiment-design/cambodia-thailand-gate-b/gate_b_ccnl_placebo_site_inference.csv`
- Figure: `data/exp/experiment-design/cambodia-thailand-gate-b/gate_b_ccnl_product_replication.png`
