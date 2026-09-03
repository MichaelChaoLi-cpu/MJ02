# Gate B results: Cambodia–Thailand border-conflict experiment

Status: **suggestive localized signal; causal promotion threshold not met**  
Analysis date: **2026-08-22**  
Primary outcome: **annual asinh LongNTL, 2000–2024**

## 1. What was estimated

The approved Gate B design compares Cambodian cells within the 0–10, 10–20, 20–40, and 40–60 km rings around verified 2011 conflict events with outcome-blind weighted cells more than 60 km from all 2008–2011 events along the same national frontier. Primary weights balance fixed geography and 1991–2007 climate but exclude outcome history. The main model absorbs grid-cell effects and one-degree border-sector-by-year effects and controls for annual precipitation, extreme rainfall, dry-spell, and maximum-temperature anomalies. Standard errors are clustered by frozen 10 km spatial blocks.

The generalized five-category weighting attempt was rejected before effect estimation because the 0–10 km ring retained only 2.5 effective spatial blocks and a maximum weighted covariate difference of 2.45 SD. The repaired design instead balances the entire 0–60 km exposure support against frontier controls and uses the rings only to describe the pre-specified distance gradient. This reduces the maximum support-level covariate difference to 0.036 SD.

## 2. Supporting evidence

The continuous-distance model produces the following standardized estimates relative to 2005–2007:

| Period | Estimate (pre-period SD) | 95% CI |
|---|---:|---:|
| 2000–2004 placebo pre-period | -0.001 | [-0.006, 0.004] |
| 2008–2010 precursor/escalation | -0.053 | [-0.120, 0.014] |
| 2011 conflict year | -0.641 | [-1.230, -0.052] |
| 2012–2014 early recovery | -0.125 | [-0.247, -0.003] |
| 2015–2019 medium recovery | -0.006 | [-0.191, 0.180] |
| 2020–2024 long recovery | -0.099 | [-0.337, 0.138] |

The ring estimates show a broadly attenuating 2011 pattern: -0.395 SD at 0–10 km, -0.287 SD at 10–20 km, -0.296 SD at 20–40 km, and -0.181 SD at 40–60 km. The earliest placebo period is close to zero, and the 2008–2010 coefficients do not establish an anticipatory decline. Excluding the immediate 0–5 km zone leaves the continuous 2011 estimate essentially unchanged (-0.640 SD).

## 3. Evidence that prevents causal promotion

### Spatial placebo inference

Eleven outcome-blind placebo placements preserve the along-border separation of the two actual conflict sectors, remain at least 120 km from the actual sectors, pass fixed cell/block support thresholds, and achieve maximum weighted SMD below 0.10. Ten have 2011 estimates above the actual -0.641 SD estimate, but one reaches -0.900 SD. The one-sided rank p-value is therefore 0.167. With this short frontier and only two actual conflict sectors, the randomization distribution cannot deliver conventional design-based significance.

### Leave-one-sector estimates

| Scenario | 2011 estimate (SD) | SE | Maximum weighted SMD |
|---|---:|---:|---:|
| Both conflict sectors | -0.641 | 0.300 | 0.036 |
| Preah Vihear only | -0.553 | 0.212 | 0.055 |
| Ta Moan/Ta Krabey only | 0.273 | 0.487 | 0.227 |

The pooled result is driven by Preah Vihear. The western sector does not independently reproduce it and lacks adequate support under the same weighting rule.

### Local annual-shock absorption

Replacing border-sector-by-year effects and weather controls with climate-cell-by-year effects attenuates the actual 2011 continuous-dose estimate to -0.305 SD with an SE of 0.388. Placebo estimates become extremely unstable because little distance-dose variation remains within approximately 5 km climate cells. This specification is weakly identified rather than a decisive refutation, but it does not provide positive confirmation.

### Dated vegetation response

Sixteen-day EVI and NDVI were differenced from each cell and seasonal-slot's 2001–2007 mean. Neither Preah Vihear's February conflict window nor Ta Moan/Ta Krabey's late-April conflict window shows a coherent negative vegetation response. A western-sector EVI loss instead appears in the September monsoon window. The annual LongNTL signal is therefore domain-specific and cannot be corroborated as an immediate ecological disruption.

## 4. Gate B verdict

The experiment establishes a reproducible and temporally sharp **Preah Vihear-specific nighttime-activity signal in 2011 followed by apparent recovery**. It does not yet establish that the 2011 conflict caused a general decline across Cambodia's affected border sectors. The strongest weakening facts are the single-sector dependence and the spatial-placebo rank p-value of 0.167.

The correct disposition is:

- retain the result as a candidate diagnostic;
- do not activate the conflict-by-climate-sensitivity Gate C;
- do not replace the current AnaSOP paper design;
- seek an independent measurement capable of distinguishing conflict from a local 2011 shock.

Independent CCNL DMSP-derived lights and observed 2011 inundation have now been
tested below. The main remaining additions are native F18 stable lights with
cloud-free coverage and independently geocoded displacement, road-closure, or
damaged-settlement records around Preah Vihear. These would address measurement
coverage and the remaining contemporaneous local-disruption mechanisms directly.

### Independent-product update (2026-08-22)

An unattended validation using the public CCNL DMSP-derived product for
2010–2013 produces a directionally consistent 2011 estimate: -0.290 SD for all
event sectors and -0.294 SD for Preah Vihear, with both clustered confidence
intervals crossing zero. The actual placement is more negative than all 11
frozen spatial placebos (one-sided rank p=0.0833, the smallest attainable with
this placebo count). This weakens a LongNTL-specific processing explanation but
does not resolve contemporaneous local confounding or replace native F18
validation. Full details are in
`docs/proposals/cambodia_thailand_border_conflict_ccnl_validation.md`.

### Observed-inundation update (2026-08-22)

Global Flood Database events 3850 and 3853 were aggregated to the national 1 km
grid and linked to the frozen Gate B sample. In Preah Vihear, conflict-distance
exposure does not predict maximum observed flooded share (p=0.767) or any
observed flooding (p=0.871). The 2011 coefficients remain negative after direct
flood adjustment and among well-observed cells with no detected flooding:
-0.600 SD for LongNTL and -0.527 SD for CCNL in the dry-cell specification.
This rules out mapped 2011 inundation as the explanation for the light pattern,
but it does not resolve the one-sector design or unobserved displacement and
access-disruption confounding. Full details are in
`docs/proposals/cambodia_thailand_border_conflict_flood_confound.md`.

### Precursor-timing update (2026-08-22)

An annual Preah Vihear specification separates the verified 2008, 2009, and
2011 clash years. The two small precursor-year coefficients are -0.000 and
0.003 SD and are jointly compatible with zero (p=0.819), whereas the 2011
coefficient is -0.570 SD. The 2011 coefficient differs from the precursor-year
mean by -0.570 SD (p=0.0096) and is followed by attenuation in 2012 and near-
recovery in 2013. This improves timing alignment, although a modest 2010 decline
and the very small number of clash years prevent interpreting the pattern as a
causal conflict-intensity response. Full details are in
`docs/proposals/cambodia_thailand_border_conflict_precursor_timing.md`.

## 5. Reproducible artifacts

- Main estimator: `src/analyses/estimate_cambodia_thailand_gate_b_longntl.py`
- Spatial placebos: `src/analyses/test_cambodia_thailand_gate_b_placebo_sites.py`
- Climate-cell-by-year test: `src/analyses/test_cambodia_thailand_gate_b_climate_cell_time_fe.py`
- Sector robustness: `src/analyses/test_cambodia_thailand_gate_b_sector_robustness.py`
- Vegetation timing: `src/analyses/diagnose_cambodia_thailand_gate_b_2011_vegetation_timing.py`
- Observed-flood confound test: `src/analyses/test_gate_b_2011_flood_confound.py`
- Precursor-timing test: `src/analyses/test_gate_b_precursor_timing.py`
- Evidence summary figure: `data/exp/experiment-design/cambodia-thailand-gate-b/gate_b_longntl_evidence_summary.png`
- Machine-readable outputs: `data/exp/experiment-design/cambodia-thailand-gate-b/`
