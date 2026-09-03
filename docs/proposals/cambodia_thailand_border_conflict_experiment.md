# Experimental proposal: Cambodia–Thailand border conflict, local development, and climate sensitivity

Status: **Gate B executed; a Preah Vihear-specific 2011 LongNTL decline remains suggestive. The first approved continuous-dose NPP drought-amplification model is negative and conventionally significant (-0.0431 kg C/m2, p = 0.0396), passes the post-result directional Monte Carlo rule, and is negative in both sector decompositions. However, the predetermined fixed distance-ring diagnostic produces four small positive coefficients, no joint rejection (p = 0.7249), and no inner-versus-outer difference (p = 0.9807). The NPP finding is therefore dependent on the linear distance-dose functional form and is not ready for causal promotion. Covariance-scale and dynamic diagnostics may characterize it but cannot replace the missing distance gradient; the design remains outside AnaSOP Sections 1–2.**  
Research type: **applied empirical research**  
Candidate conflict period: **2008–2011 escalation; 2011 is the UCDP-coded conflict year**, with event-specific dates and locations to be frozen only after source verification

## 1. Proposed central question

Did localized exposure to the 2008–2011 Cambodia–Thailand border escalation, particularly the UCDP-coded fighting in 2011, cause a detectable loss or slower recovery in local economic activity and vegetation productivity, and did exposed places subsequently become more sensitive to drought, extreme rainfall, or heat?

This question has two logically ordered parts:

1. **Direct conflict effect:** did activity or vegetation change when fighting occurred and during the subsequent recovery period?
2. **Climate-sensitivity effect:** conditional on a credible direct-effect design, did the response to later weather extremes become larger in exposed places?

The second part will not be promoted as a main result if the first part has inadequate support, power, or identification. A null direct effect does not prove that the conflict caused no harm to households; it only bounds effects on the satellite-observed domains and spatial scales studied.

## 2. Why this is not a simple border comparison

The conflict was localized around disputed border sectors rather than assigned randomly across Cambodia. Cells near the Thai border differ structurally from interior cells in elevation, slope, roads, markets, population density, and land cover. Therefore:

- “within X km of Thailand” is not itself a defensible treatment;
- the year 2008 is not a single permanent treatment date, because clashes recurred at different sites through 2011;
- all of Cambodia is retained as a control reservoir, but only cells with adequate pre-conflict comparability enter the causal comparison;
- distance bands describe exposure gradients; they do not independently create causal identification.

Official records confirm escalating tension near Preah Vihear in July 2008, armed incidents in October 2008, repeated clashes in February 2011, and renewed fighting in April 2011. UCDP Dyadic 26.1 identifies the Cambodia–Thailand government dyad as conflict 294 / dyad 634, but codes only 2011 within 2000–2024 because the annual dyadic series uses a 25-fatality activation threshold. UCDP GED nevertheless retains lethal events in below-threshold years for a dyad that becomes active in another year. The event extract consequently contains one record on 15 October 2008, one on 3 April 2009, and twelve records from February–May 2011. The design must distinguish the 2011 active conflict year from 2008–2009 precursor events while retaining both in declared sensitivity analyses.

## 3. Research questions and falsification criteria

### RQ1. Direct local disruption

Did conflict-exposed grid cells experience lower nighttime activity, NPP, EVI, or NDVI during and after verified clashes relative to comparable unexposed cells?

- Supporting evidence: stable pre-trends, a decline beginning no earlier than the conflict, and a coherent dose gradient by verified event proximity or severity.
- Weakening evidence: pre-2008 divergence, effects at placebo sites or placebo dates, or estimates driven by a single event coordinate or buffer.

### RQ2. Recovery path

If a decline occurred, how quickly and completely did affected places recover between 2012 and 2024?

- Supporting evidence: event-time estimates trace an interpretable loss and rebound path across independent outcome systems.
- Weakening evidence: no measurable direct decline, sensor-specific patterns, or intervals too wide to distinguish recovery from persistent loss.

### RQ3. Subsequent climate sensitivity

After conflict exposure, did drought, extreme rainfall, or heat produce larger reductions in vegetation or nighttime activity?

- Supporting evidence: the exposure-by-hazard response changes after verified fighting, while pre-conflict placebo interactions are compatible with zero.
- Weakening evidence: the interaction predates conflict, depends on total rainfall rather than independently constructed extremes, or fails across nearby bandwidths.

### RQ4. Spatial reach

How rapidly do direct and climate-conditioned effects attenuate with distance from verified conflict sites?

- Supporting evidence: a monotone or otherwise interpretable pattern across fixed 0–10, 10–20, 20–40, and 40–60 km rings, with continuous-distance estimates agreeing.
- Weakening evidence: isolated significance in one selected ring, unstable signs, or apparent effects farther away with no corresponding local effect.

### RQ5. Domain specificity

Are any effects primarily economic, ecological, or both?

- Supporting evidence: consistent timing within a domain and transparent differences across LongNTL, NPP, EVI, and NDVI.
- Weakening evidence: treating NPP as crop yield, nighttime light as household consumption, or rainfall extremes as observed flood inundation.

## 4. Data already constructed

### Nationwide analytical support

- Stable Cambodia 1 km grid: 179,072 cells linked to province, district, and commune.
- LongNTL annual panel: 2000–2024, 4,476,800 cell-years.
- MODIS annual NPP: 2001–2024, 4,297,728 cell-years.
- MODIS EVI/NDVI: 2001–2024, 98,310,528 cell-composites.
- CHIRPS daily precipitation and CHIRTS-daily temperatures: 1991–2024 on 6,252 climate cells.
- Annual climate hazards: 212,568 climate-cell-years.
- Sixteen-day climate hazards: 4,889,064 climate-cell-composites.
- Joined annual satellite-climate panel: 2000–2024, 4,476,800 grid-cell-years.
- UCDP Cambodia–Thailand dyad-year extract: conflict 294 / dyad 634; 2011 is the only active conflict year during 2000–2024.
- UCDP GED candidate-event extract: 14 lethal event records at four coded coordinate locations—one inactive-year record in 2008 (best deaths 2), one inactive-year record in 2009 (best deaths 2), and twelve active-year records in February–May 2011 (best deaths 29; low 29; high 36) across the Preah Vihear and Ta Moan/Ta Krabey sectors.
- Outcome-blind national conflict-exposure features: nearest-event distance and event/death doses within 10, 20, 40, and 60 km for all events, the 2008–2009 precursor phase, the 2011 active conflict year, and higher-spatial-precision events.
- Pre-conflict MODIS MCD12Q1 land-cover shares and stability over 2001–2007 for all 179,072 national grid cells.
- Cambodia–Thailand shared-border geometry and grid-cell distance/sector assignments, using documented Cambodia and Thailand government-origin geometries.
- GHS-POP 2000 baseline population for all 179,072 grid cells, plus a WorldPop 2000 robustness layer with more than 99.9% grid coverage.
- Public AidData Cambodia road-project corridors, work types, and completion years, used only to flag known post-2007 road construction and not as a complete road network.
- Copernicus DEM GLO-30 elevation and slope summaries for all 179,072 national grid cells, acquired anonymously from the Amazon Web Services open-data archive.
- gROADSv1 historical-road access proxies for all 179,072 grid cells, including an AidData-corridor-exclusion sensitivity version; the clipped source contains 971 Cambodia road features totalling approximately 13,527 km.

### Constructed hazard families

- Dryness: monsoon precipitation anomaly, maximum consecutive dry days, and dry-rainfall intensity.
- Extreme rainfall: maximum one-day and five-day rainfall, extreme-wet-day counts, and wet-rainfall intensity.
- Heat: mean and maximum temperature, hot days, hot nights, heatwave days, and heat intensity.
- Compound hazard: joint hot-and-dry intensity.

Rainfall extremes are meteorological hazards and must not be described as observed floods. Observed inundation remains a separate, temporally limited validation source.

### Approved predetermined covariate stack

The source hierarchy below was approved before outcome estimation. Variables will be
aggregated to the stable national 1 km grid and used for common-support assessment,
matching or weighting, and heterogeneity checks. Their time-invariant levels are absorbed
by cell fixed effects and will not be mechanically added as separate regressors.

| Covariate family | Primary source and construction | Robustness or audit source | Timing rule |
|---|---|---|---|
| Terrain | Copernicus DEM GLO-30 Public 2021 release; derive mean elevation, elevation dispersion, mean slope, upper-tail slope, and steep-terrain share | JAXA ALOS AW3D30 Version 4.1 when direct JAXA access becomes available | Time invariant; source year does not create post-treatment variation |
| International boundary | Existing Cambodia government-origin OCHA/MEF administrative geometry, dissolved to the national outline and paired with Thailand geometry from the same or a documented compatible source | GADM boundary and alternative shared-border geometry | Geometry is used for distance and sector assignment, not as treatment |
| Roads | gROADSv1 inter-settlement network, clipped to Cambodia, with source-vintage uncertainty reported | AidData Cambodia road-project lines and their completion/work-type fields; contemporary OSM/ODC only for descriptive comparison | The main access measure must approximate the pre-conflict network; known post-2007 new roads are flagged or removed in sensitivity analysis |
| Land cover | MODIS MCD12Q1 Version 6.1 IGBP classes summarized over 2001–2007 | Alternative class groupings and stability thresholds | Freeze pre-conflict shares; later land cover is an outcome or mechanism, not a control |
| Population | GHS-POP/GPW year-2000 gridded population | WorldPop 2000 and, if subsequently available, pre-conflict Cambodian census aggregates | Freeze the baseline level; later modeled population is not a time-varying control |

Readable grid-level variables are: `Mean Elevation m`, `Elevation SD m`, `Mean Slope
Degrees`, `Slope P90 Degrees`, `Steep Terrain Share`, `Distance to Cambodia Thailand
Border km`, `Nearest Border Sector`, `Historical Road Density Proxy km per km2`, `Distance to
Nearest Historical Road Proxy km`, `Baseline Cropland Share`, `Baseline Forest Share`,
`Baseline Grass Shrub Share`, `Baseline Built Share`, `Baseline Water Wetland Share`,
`Baseline Land Cover Stability Share`, `Baseline Population 2000`, `Log Baseline Population
2000`, and `Baseline Population Density per km2`.

### Road-source and satellite-validation rule

No new government road-data application will be made for this design. The published
gROADSv1 source is the primary historical road layer, while the public AidData Cambodia
road-project GeoJSON is used to identify known post-2007 road construction, rehabilitation,
and upgrading. The latter contains project corridors rather than a complete national road
network and therefore cannot replace gROADSv1.

Satellite imagery will not be used to generate the primary road network from scratch. A
pre-specified validation can instead test whether candidate major-road segments are visible
in dry-season Landsat 5/7 composites from 2005–2007. The validation will compare spectral
and linear-feature contrast along candidate centerlines with adjacent shoulders and classify
only major-road presence, with an explicit uncertain category. Landsat's 30 m pixels cannot
reliably recover narrow rural roads. Sentinel-2 or contemporary high-resolution imagery
begins after treatment and may be used only to validate current geometry, never to define a
pre-conflict matching covariate. Any satellite-derived road score remains an appendix
robustness measure unless it passes a blinded manual-validation threshold frozen before
outcome analysis.

### Data and decisions still required before causal estimation

1. Human approval of the revised frontier dose-response design described in the Gate A diagnostic report.
2. A frozen spatial-precision rule for the 2011 event coordinates and a documented protocol for the 2008–2009 precursor incidents omitted from the UCDP dyad-year eligibility record. Any supplemented event must come from pre-declared primary official sources and remain separate from the UCDP treatment definition.
3. A pre-estimation inference protocol appropriate to the small number of independent conflict sectors, including the eligible placebo-site universe and spatial-block construction.

The national outcome panel, 2011 UCDP exposure layer, predetermined covariates, and outcome-blind common-support diagnostics are complete. Detailed Gate A results and the proposed design revision are recorded in `docs/proposals/cambodia_thailand_border_conflict_gate_a_diagnostic.md`.

## 5. Conflict-exposure construction

The raw UCDP GED file will remain unchanged. A reproducible script will:

1. retain events dated 2000–2024 inside Cambodia or a narrow transboundary bounding box;
2. identify Cambodia–Thailand state-based events using actors/dyads rather than place-name keywords alone;
3. retain event coordinate precision, date precision, and low/best/high fatality estimates;
4. reconcile the candidate list against official UN or ICJ chronology;
5. flag, rather than silently discard, spatially or temporally imprecise records;
6. freeze a primary UCDP 2011 event set before examining outcome estimates;
7. construct a separately flagged 2008–2009 official-chronology precursor set for sensitivity analysis, never silently pooling it with UCDP events.

For each national 1 km cell, construct:

- distance to nearest verified conflict event;
- event counts within 10, 20, 40, and 60 km;
- best-estimate fatalities within the same radii;
- inverse-distance or kernel-weighted conflict dose;
- first verified UCDP exposure year and event-specific exposure history;
- mutually exclusive distance rings: 0–10, 10–20, 20–40, 40–60, and over 60 km.

The 20 km ring is a substantively motivated candidate because official reporting indicates artillery reaching roughly that distance in February 2011, but it will not be the sole specification. This radius is a design input to be frozen before outcome estimation, not selected from results.

## 6. Samples and comparison groups

### Analytical frontier and exposure support

The proposed Gate B sample contains Cambodian grid cells within 60 km of the Cambodia–Thailand border. Exposure is a distance dose within this frontier sample, not a single treated-area boundary. The constructed layer contains 379 cells within 10 km of a verified 2011 event, 1,383 within 20 km, 5,068 within 40 km, and 10,605 within 60 km. Alternative precursor exposure based on the 2008–2009 GED and official records remains a secondary definition. The 40 and 60 km boundaries define analytical support and outer exposure rings; Gate A rejected interpreting either as a homogeneous direct treatment.

### Candidate control reservoir

The primary reservoir is composed of cells within 60 km of the Cambodia–Thailand border and more than 60 km from every 2008–2011 candidate event. Secondary diagnostics may draw unexposed Cambodian cells from:

- other international-border sectors with no verified conflict event;
- cells farther from the affected sites in the same northern provinces; and
- interior cells with similar pre-2008 outcome and climate histories.

### Matching and weighting

Primary overlap weights will be estimated without any outcome history. Predictors are:

- 1991–2007 precipitation and temperature normals and variability;
- latitude, longitude, distance to border, elevation, slope, baseline land cover, roads, and population;
- province or broad agro-ecological region.

An explicitly labelled sensitivity weighting scheme may additionally balance 2000–2007 LongNTL and 2001–2007 NPP levels and trends. Overlap weighting is the primary Gate B method because it achieved the strongest balance while preserving more spatial support than nearest-neighbour matching. Coarsened exact matching, entropy balancing, or synthetic difference-in-differences weights may be compared as robustness checks. Lack of common support is a stop condition, not a reason to extrapolate.

## 7. Estimation sequence

### Gate A. Exposure validity and design support

Before outcome estimation:

- map every event and its precision;
- report cell counts and effective climate cells in every ring;
- test covariate balance and 2000–2007 outcome pre-trends;
- estimate minimum detectable effects under spatial clustering;
- freeze primary exposure, sample, matching variables, and inference method.

Gate A was executed without loading outcomes after 2007. The result is **partial support, not approval of the original binary-radius design**. At 10 and 20 km, the number of spatially independent treated units is too small for a credible standalone binary DiD. At 40 and 60 km, overlap weighting produces strong predetermined-covariate balance and the pre-conflict outcome differences are small in substantive magnitude, but these radii are too broad to be interpreted as homogeneous direct physical exposure. They therefore define the support for a continuous or ring-based frontier dose-response design rather than alternative binary treatments. See the Gate A diagnostic report for numerical results and the revised estimand.

### Gate B. Direct-effect event study

For annual outcome Y in cell i and year t, estimate a weighted distance-ring event study with cell fixed effects and calendar-year fixed effects:

`Y(i,t) = cell FE + year FE + sum over distance rings r and event times k [beta(r,k) × ring(i,r) × event-time(t,k)] + weather controls + error(i,t)`

The omitted distance category is the frontier control reservoir more than 60 km from every candidate event. A continuous monotone distance-dose model supplements rather than replaces the declared ring estimates.

Gate B has now been executed. The frozen continuous-dose model estimates a 2011 LongNTL decline of 0.641 pre-period SD at the event-location-versus-60-km contrast, followed by a smaller 0.125 SD early-recovery gap and no clear medium-run gap. This temporal pattern is concentrated in the Preah Vihear sector: its leave-one-sector estimate is -0.553 SD, whereas the Ta Moan/Ta Krabey estimate is positive, imprecise, and fails the balance threshold. One of eleven support-passing spatial placebo placements produces a larger negative 2011 estimate, yielding a one-sided rank p-value of 0.167. Climate-cell-by-year fixed effects attenuate and destabilize the estimate, and 16-day EVI/NDVI do not show a coherent negative response in the dated conflict windows. Gate B therefore identifies a localized signal requiring independent validation, not a promoted causal effect. Full results are in `docs/proposals/cambodia_thailand_border_conflict_gate_b_results.md`.

Primary outcomes:

1. asinh LongNTL radiance;
2. annual NPP anomaly Z.

Supporting outcomes:

3. sixteen-day EVI anomaly;
4. sixteen-day NDVI anomaly.

The reference period is pre-escalation. Event time will be aggregated to avoid sparse annual coefficients, provisionally: 2000–2004, 2005–2007, 2008–2010 precursor/escalation, 2011 UCDP conflict, 2012–2014 early recovery, 2015–2019 medium-run recovery, and 2020–2024 long-run recovery. Site-specific timing will replace these bins where event coverage permits.

### Gate C. Climate-sensitivity change

Only after Gate B and adequate power, estimate:

`Y(i,t) = cell FE + year FE + hazard(c,t) + exposure(i) × hazard(c,t) + post-conflict(t) × exposure(i) × hazard(c,t) + controls + error(i,t)`

The coefficient of interest is the three-way interaction. It asks whether conflict exposure changed the marginal response to the same standardized climate hazard. It does **not** ask whether exposed places are simply poorer on average.

For sixteen-day vegetation, replace annual year effects with composite-date effects and use distributed hazard leads/lags. Climate-cell-by-time fixed effects will be used where within-climate-cell exposure support is adequate; this compares differently exposed 1 km cells facing the same measured weather realization.

The drought-first Gate C data and prospective power audit are now complete. The annual
panel contains 927,600 grid-cell-years and supplies strong temporal hazard support, but the
pre-specified power gate fails. At an absolute 0.20 outcome-SD interaction, power is 33.8%
for LongNTL and 39.3% for NPP under the primary rho = 0.5 calibration; the corresponding
80% minimum detectable effects are 0.363 and 0.332 SD. The target interaction retains only
19.8% of its fixed-effect-absorbed variance after residualizing lower-order terms (VIF 5.05).
Accordingly, post-conflict outcome coefficients remain unopened. See
`docs/proposals/cambodia_thailand_border_conflict_gate_c_power.md`.

### Fixed effects and controls

- Cell fixed effects absorb all time-invariant cell attributes; they are not a list of demographic controls.
- Calendar-year or composite-date fixed effects absorb national temporal shocks.
- Matched-set-by-year effects will be tested for annual models.
- Climate-cell-by-year or climate-cell-by-composite effects are preferred for hazard interactions where support permits.
- Time-varying controls are restricted to predetermined or clearly exogenous measures; post-conflict demographics are not inserted mechanically because they may be mediators.

### Inference

Because treatment arises from a small number of conflict sectors, conventional cell-clustered standard errors are insufficient. Primary inference will use spatial randomization inference or permutation across comparable border sectors, supplemented by spatial-block bootstrap or Conley-type covariance estimates. Results will report effect sizes, confidence intervals, effective clusters, and design-based p-values.

## 8. Robustness and falsification plan

- Fixed distance bands: 10, 20, 40, and 60 km.
- Continuous-distance dose and alternative fatality weighting.
- Leave-one-event-site and leave-one-sector-out checks.
- Exclude the immediate 0–5 km displacement/destruction zone, then model it separately.
- Placebo conflict dates before 2008.
- Placebo event sites on unaffected border sectors.
- Pre-conflict exposure-by-hazard interaction test.
- Separate reconstructed LongNTL years from observed-composite years.
- Compare NPP with EVI/NDVI without treating agreement as automatic.
- Sensitivity to spatial precision and date precision filters.
- Equivalence tests where null results are central, with bounds frozen from outcome-SD and substantive-scale calculations before estimation.

## 9. Decision rule for whether this becomes the paper

Promote this direction only if all five conditions hold:

1. verified events provide more than one usable site or exposure cohort;
2. treated and control cells have credible pre-conflict common support;
3. the direct-effect design has adequate spatially adjusted power;
4. event-study leads do not show material pre-trends;
5. at least one primary outcome supplies a robust direct effect, recovery path, or tightly bounded null of substantive interest.

If these gates fail, retain the conflict analysis as a diagnostic comparison and return to the nationwide historical-conflict-by-climate design. The nationwide panel remains valuable in either case.

## 10. Immediate execution order

1. **Done:** finish and checksum the UCDP GED 26.1 download.
2. **Done:** build and manually audit the Cambodia–Thailand event extract.
3. **Done:** construct grid-level conflict distances and doses.
4. **Done:** construct border distance, terrain, roads, land cover, and baseline population.
5. **Done:** produce the outcome-blind support, balance, power, and holdout-pretrend report.
6. **Done:** approve and freeze the frontier dose-response specification.
7. **Done:** run the direct-effect event study, spatial placebo test, leave-one-sector checks, local-time fixed effects, and high-frequency vegetation timing diagnostic.
8. **Done:** replicate the 2011 direction with independent CCNL DMSP-derived lights.
9. **Done:** test and exclude mapped 2011 inundation as the explanation for the Preah Vihear light decline.
10. **Done:** separate the small 2008–2009 Preah Vihear precursor clashes from the larger 2011 episode; the light decline is concentrated in 2011.
11. **Done:** organize the drought-first Gate C annual panel and audit hazard support without inspecting the target coefficient.
12. **Done:** run the prospective design-rank and power gate using pre-conflict outcome variance only.
13. **Done:** complete the first human review, remove LongNTL from Gate C, and approve the NPP-only main specification.
14. **Done:** open the first NPP target coefficient without specification search; the natural-unit rainfall-deficit model and mandatory dry-spell confirmation are both negative and conventionally significant.
15. **Done:** run the frozen spatial-placebo experiment; the actual estimate ranks third-most negative among 12 actual-or-placebo assignments and fails spatial validation (one-sided rank p = 0.25).
16. **Done:** redesign and freeze 1,000 accepted random two-sector assignments with variable separation and run the exact NPP model for every assignment; the actual estimate passes the approved one-sided 5% rule (p = 0.0340), while the equal-tail two-sided p-value is 0.0679.
17. **Done:** by human decision, supersede the under-resolved 11-placement exercise and use only the 1,000-draw Monte Carlo in the active spatial evidence chain; retain superseded files solely for audit.
18. **Done:** estimate the separate-sector and joint common-sample NPP decomposition. Both sector coefficients are negative; their joint-model difference is not significant (p = 0.724), but neither sector-only estimate is individually precise and the Ta Moan/Ta Krabey standalone comparison fails the 0.10 balance benchmark.
19. **Done:** estimate the fixed 0–10, 10–20, 20–40, and 40–60 km NPP gradient without selecting a favorable ring. All four coefficients are small and positive, are jointly compatible with zero (p = 0.7249), and the inner-minus-outer contrast is essentially zero (p = 0.9807); the continuous result fails this functional-form diagnostic.
20. **Current decision:** pause before opening covariance sensitivity at 5, 10, and 20 km or the early-, medium-, and late-post-conflict dynamics.

## 11. Source anchors for treatment chronology

- UCDP GED 26.1 download and codebook: <https://ucdp.uu.se/downloads/>
- UCDP GED 26.1 codebook: <https://ucdp.uu.se/downloads/ged/ged261.pdf>
- UCDP API documentation and version/filter definitions: <https://ucdp.uu.se/apidocs/>
- UN statement on border tension near Preah Vihear, 21 July 2008: <https://www.un.org/sg/en/content/former-secretary-general/statement/2008-07-21/statement-attributable-the-spokesperson-for-the-secretary-general-cambodiathailand-tensions-the-border-near-the-preah-vihear-temple>
- UN statement on Cambodia–Thailand clashes, 15 October 2008: <https://press.un.org/en/2008/sgsm11865.doc.htm>
- UN statement on repeated clashes, 4–6 February 2011: <https://press.un.org/en/2011/sgsm13393.doc.htm>
- UN statement on renewed fighting, April 2011: <https://press.un.org/en/2011/sgsm13522.doc.htm>
- ICJ provisional-measures order, 18 July 2011: <https://www.icj-cij.org/node/103723>
