# Gate A diagnostic: Cambodia–Thailand border-conflict experiment

Status: **partial design support; post-conflict outcomes remain unopened for effect estimation**  
Diagnostic date: **2026-08-22**  
Primary event phase: **2011 UCDP conflict events**  
Precursor role: **2008–2009 events reserved for robustness**

## 1. Question answered at this gate

This diagnostic does not estimate whether the conflict changed nighttime activity or vegetation. It asks the logically prior question: are exposed and unexposed Cambodian locations sufficiently comparable before the conflict, and are there enough independent spatial units to make a later effect estimate informative?

All weighting and support calculations use only fixed geography, predetermined covariates, and information observed no later than 2007. No outcome after 2007 was loaded. The treatment radii—10, 20, 40, and 60 km—were frozen before the diagnostics.

## 2. Inputs and exclusions

- National analytical grid: 179,072 one-kilometre cells.
- Design-eligible grid: 179,045 cells.
- Excluded cells: 27 coastal or raster-edge cells with incomplete 1991–2007 climate histories; none is treated under any candidate radius.
- Treatment: proximity to the 2011 UCDP event coordinates.
- Strict controls: more than 60 km from every 2008–2011 candidate event.
- Primary comparison reservoir for the revised design: strict controls within 60 km of the Cambodia–Thailand frontier.
- Balance estimator: logistic-propensity overlap weights, with nearest-neighbour matching retained only as a diagnostic.
- Outcome-independent pretrend weights: fixed geography and 1991–2007 climate only.
- Pretrend models: cell and year fixed effects with 10 km spatial-block clustered covariance; 2007 is the reference year.

## 3. Common-support result

| Binary radius and reservoir | Treated cells | Treated communes | Maximum weighted SMD | Treated/control 10 km block ESS | Screening MDE (SD) | Gate A interpretation |
|---|---:|---:|---:|---:|---:|---|
| 10 km, local annulus | 379 | 4 | 0.152 | 5.6 / 10.9 | 1.45 | Local interpretation is plausible, but independent support and power are inadequate. |
| 20 km, local annulus | 1,383 | 8 | 0.281 | 12.0 / 7.6 | 1.30 | Balance and power are inadequate for a standalone binary treatment. |
| 40 km, frontier controls | 5,068 | 24 | 0.077 | 24.4 / 19.1 | 0.86 | Strong cell-level balance, but still low power after spatial dependence and too broad for homogeneous direct exposure. |
| 60 km, frontier controls | 10,605 | 53 | 0.025 | 61.8 / 55.2 | 0.52 | Best support and power, but substantively diffuse; suitable as a comparison domain, not as a binary direct-exposure definition. |

The screening MDE is `(1.96 + 0.84) × sqrt(1 / treated block ESS + 1 / control block ESS)`. It is a conservative design-screening approximation, not the final model-specific power calculation.

The result exposes a structural trade-off. Narrow radii correspond more closely to direct fighting, artillery, displacement, and access disruption, but contain too few independent places. Wider radii provide overlap and power, but mix directly exposed and only geographically proximate cells. Selecting 60 km because it has the smallest MDE would therefore change the scientific treatment rather than merely improve precision.

## 4. Outcome-independent pretrend result

The weights for this test exclude LongNTL and NPP. This avoids manufacturing parallel pretrends by balancing directly on the same outcome histories later used to assess them.

| Candidate design | LongNTL joint p | Maximum absolute LongNTL lead (SD) | NPP joint p | Maximum absolute NPP lead (SD) | Interpretation |
|---|---:|---:|---:|---:|---|
| 10 km, local annulus | 0.867 | 0.203 | <0.001 | 0.172 | No joint LongNTL rejection, but individual deviations are large; NPP shows a material early gap. |
| 40 km, frontier controls | 0.324 | 0.002 | 0.001 | 0.054 | LongNTL is exceptionally close; NPP is jointly non-parallel but deviations are below 0.10 SD. |
| 60 km, frontier controls | 0.288 | 0.001 | 0.005 | 0.023 | LongNTL is exceptionally close; NPP is jointly non-parallel but deviations are very small. |

The NPP p-values at 40 and 60 km should not be read alone. The annual differences relative to 2007 are small—at most 0.054 and 0.023 SD—even though their covariance makes the joint Wald test reject. They require explicit trend adjustment and sensitivity analysis, but they are not evidence of a large pre-existing productivity divergence. By contrast, the 10 km design combines material pre-period deviations with very low spatial effective sample size.

## 5. Gate A decision

The original plan of estimating four separate binary DiDs at 10, 20, 40, and 60 km is rejected before inspecting effects. It would either rely on an underpowered local comparison or redefine exposure broadly enough to dilute its physical meaning.

The data do support a narrower question:

> Along the Cambodian side of the Thailand frontier, did post-2011 outcomes change more near verified conflict sites than farther away, and did the change attenuate with distance?

This is a spatial dose-response event study. It does not treat every cell within 40 or 60 km as receiving the same intervention.

## 6. Proposed frozen Gate B specification

Subject to human approval, the direct-effect analysis will use the following specification.

1. **Sample:** Cambodian cells within 60 km of the Cambodia–Thailand border. Cells more than 60 km from every 2008–2011 event form the frontier control reservoir. The 60 km border restriction is a sample-support rule, not the treatment.
2. **Exposure:** mutually exclusive distance rings of 0–10, 10–20, 20–40, 40–60, and over 60 km from verified 2011 events, supplemented by one pre-specified monotone continuous-distance dose. The 0–20 km contrast represents the most plausible direct-exposure range; outer rings identify attenuation and spillovers.
3. **Timing:** 2000–2007 clean pre-period, 2008–2010 precursor/escalation period, 2011 conflict year, 2012–2014 early recovery, 2015–2019 medium-run recovery, and 2020–2024 long-run recovery. The 2008–2009 GED events remain a separately labelled robustness exposure.
4. **Primary outcome:** annual asinh LongNTL. It has the strongest holdout-pretrend result and full 2000–2024 coverage.
5. **Secondary outcome:** annual NPP anomaly, with a pre-specified differential-trend adjustment and a no-adjustment sensitivity estimate. NPP cannot by itself establish household welfare or crop yield.
6. **Supporting outcomes:** 16-day EVI and NDVI recovery dynamics, activated only after the annual design and inference are frozen.
7. **Fixed effects:** grid-cell effects, calendar-year effects, and pre-declared border-sector-by-year effects where within-sector distance support remains adequate. These compare distance rings facing the same sector-specific annual shocks.
8. **Weights:** outcome-independent overlap weights based on fixed geography, terrain, roads, land cover, baseline population, and 1991–2007 climate. Pre-outcome histories are used only in a declared sensitivity weighting scheme.
9. **Inference:** 10 km spatial blocks for descriptive covariance, plus spatial randomization inference over a pre-declared universe of eligible placebo conflict sites. Cell-clustered p-values will not be treated as primary evidence.
10. **Falsification:** pre-2008 placebo dates, unaffected frontier placebo sites, leave-one-conflict-sector-out estimates, and exclusion of the immediate 0–5 km zone.

The primary estimand is the post-2011 change in each inner ring relative to the over-60-km frontier controls, together with a joint test that the effect attenuates with distance. The analysis will report the 0–10 and 10–20 km coefficients even if imprecise; it will not replace them with the 60 km coefficient because the latter is more precise.

## 7. Stop and promotion rules

The design remains a candidate analysis unless all of the following hold:

- the placebo-site universe provides credible design-based inference despite the small number of actual conflict sectors;
- event-study leads remain below a pre-specified material threshold, provisionally 0.10 outcome SD;
- the direct-effect path has interpretable timing or bounds effects tightly enough to be informative;
- results are not driven by one sector, one event coordinate, or one reconstructed outcome family.

Climate-sensitivity interactions remain Gate C. They will not be estimated merely because a direct-effect coefficient is significant. Gate C requires a credible direct-effect design and enough variation to distinguish a changed climate response from a persistent outcome-level gap.

## 8. Reproducible artifacts

- Common-support script: `src/analyses/diagnose_cambodia_thailand_common_support.py`
- Holdout-pretrend script: `src/analyses/diagnose_cambodia_thailand_holdout_pretrends.py`
- Diagnostic directory: `data/exp/experiment-design/cambodia-thailand-common-support/`
- Core balance summary: `common_support_summary.csv`
- Full covariate balance: `covariate_balance.csv`
- Holdout coefficients: `holdout_pretrend_coefficients.csv`
- Holdout joint tests: `holdout_pretrend_joint_tests.csv`
- Review figure: `holdout_pretrend_diagnostics.png`
- Machine-readable decision summary: `design_gate_recommendation.csv`
