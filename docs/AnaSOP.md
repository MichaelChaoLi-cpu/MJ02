# AnaSOP
Analysis Standard Operating Procedure

## 1. Research Objective

### Working Title

Regional concentration of dry-spell sensitivity and household economic relevance in Cambodia

### Central Research Question

Where do prolonged dry spells impose the largest constraints on annual cropland net primary
productivity (NPP), and do those ecologically sensitive regions also exhibit stronger household
economic relevance of local cropland productivity?

This is an **applied** environmental and agricultural-geography study. It integrates national
daily climate records and satellite-derived cropland productivity with repeated household surveys
to determine whether a modest national average conceals geographically concentrated ecological
and household-economic relevance. The study retains two separately estimated empirical stages:

1. annual climate exposures to village-buffer cropland NPP; and
2. prior-year village-buffer cropland NPP to household consumption.

The principal contribution is the comparison of these two evidence dimensions across a frozen,
outcome-blind regional geography. The dimensions are reported side by side and are never multiplied
into an indirect-effect estimate.

### Supporting Research Questions

#### Supporting Point 1

- Research question: What is the national natural-unit magnitude of the association between prolonged dry spells and annual cropland NPP, including the NPP change associated with the observed P10-to-P90 dry-spell contrast?
- Role relative to central point: Establish the national ecological benchmark against which spatially concentrated sensitivity is interpreted.
- Success condition: The national village-year panel has traceable support, the dry-spell coefficient has a stable adverse direction, and its observed-exposure translation can be reported in natural NPP units.
- Failure condition: The analytical panel is not reproducible, the coefficient changes direction across the primary measurement choices, or the observed-exposure translation is too imprecise to interpret.

#### Supporting Point 2

- Research question: How strongly does dry-spell sensitivity vary across the six frozen SKATER regions and across continuous local coefficient surfaces?
- Role relative to central point: Determine whether the national benchmark conceals spatially concentrated ecological sensitivity.
- Success condition: Outcome-blind regions meet the frozen support gates, regional slopes reject a common-slope restriction, and the broad spatial pattern is compatible with adequately supported continuous estimates.
- Failure condition: Regionalisation depends on outcomes, regions lack analytical support, regional slopes are compatible with one common slope, or continuous patterns are unstable under adjacent bandwidths.

#### Supporting Point 3

- Research question: Is prior-year cropland NPP associated more robustly with household food consumption than with total consumption after household-composition and socioeconomic adjustment?
- Role relative to central point: Establish whether local cropland productivity has measurable household economic relevance in the linked survey data.
- Success condition: Exact interview-year linkage is reproducible and the food-consumption association retains its positive direction and precision under the expanded control block and declared exposure sensitivities.
- Failure condition: Household linkage or consumption harmonisation is not comparable across waves, or the food-consumption estimate loses direction or precision under the declared checks.

#### Supporting Point 4

- Research question: Which regions combine relatively strong ecological dry-spell sensitivity with relatively strong food-consumption relevance, and which regions show discordant evidence requiring a different monitoring strategy?
- Role relative to central point: Convert the two separately estimated evidence dimensions into a transparent regional monitoring and adaptation typology.
- Success condition: At least some outcome-blind regions remain jointly elevated relative to the national ecological and household-reference estimates under expanded controls and adequate sample support.
- Failure condition: Candidate priority regions disappear under the declared controls, depend on an arbitrary composite score, or lack sufficient ecological or household observations.

#### Supporting Point 5

- Research question: Does the dry-spell conclusion remain stable when annual rainfall is omitted or orthogonalised relative to Rx5day and when MODIS NPP quality support is explicitly restricted or adjusted?
- Role relative to central point: Test whether the ecological result is an artefact of rainfall collinearity, wet-year retrieval conditions, or one NPP-quality threshold.
- Success condition: The dry-spell coefficient retains its adverse direction and comparable magnitude when rainfall is omitted or residualised and when NPP quality is adjusted or restricted.
- Failure condition: The dry-spell coefficient changes direction materially, loses interpretable support, or appears only under one rainfall or NPP-quality treatment.

### Scope of Analysis

- **Geographic scope:** Cambodia, using mapped household-survey villages and national public village
  points.
- **Stage-1 period:** 2001-2021, determined by the harmonised annual cropland-NPP panel.
- **Stage-2 period:** 2007-2021 survey waves; 2004 remains a consumption-construction diagnostic.
- **Primary spatial scale:** 5 km village-centred buffers, with 2 km and 10 km sensitivities.
- **Stage-1 focal exposure:** annual maximum consecutive dry days below 1 mm. Heat days, heat
  degree-days, Rx5day, and annual precipitation remain model components or secondary results.
- **Household outcome hierarchy:** total consumption remains the originally prespecified benchmark;
  food consumption is the focal robust household-relevance result in the revised narrative. This
  post-result distinction is reported transparently rather than presented as preregistered.
- **Regionalisation:** one frozen six-region SKATER partition selected from outcome-blind candidate
  diagnostics before regional outcome estimation.
- **Regional prioritisation:** ecological sensitivity and household economic relevance are displayed
  as separate dimensions with uncertainty and sample support; no composite causal-loss estimate is
  constructed.
- **Continuous spatial heterogeneity:** geographically weighted residual regression remains the
  continuous check for Stage 1; Stage-2 local surfaces remain appendix diagnostics.

### Study Design Declaration

- Research type: applied
- Stage 1 is a national village-year panel with village and year fixed effects.
- Stage 2 is a survey-weighted repeated household cross-section linked to the most recent completed
  annual cropland-NPP observation.
- Regression variables and effect translations remain in natural units. Standardisation is used
  only inside the spatial clustering algorithm.
- Heat-day count and heat degree-days are alternative heat specifications and are never entered
  together.
- Regional priority evidence is based on transparent regional coefficients, confidence intervals,
  natural-unit translations, and analytical support rather than a significance-selected score.
- GeoDetector, temporal sample splitting, and multiplication of Stage-1 and Stage-2 coefficients
  remain excluded.

## 2. Theoretical Background  /  Conceptual Framework  /  Problem Formulation

### Research Gap

National climate-productivity estimates are useful benchmarks but can have limited operational
meaning when ecological sensitivity varies sharply across space. A small national coefficient can
average together places with almost no response and places where the same dry-spell contrast is
associated with several times the national productivity change. Uniform national risk rankings may
therefore fail to identify where additional agricultural monitoring, water-management assessment,
or household protection evidence is most needed.

Remote-sensing studies and household-welfare studies also usually occupy separate evidence systems.
Satellite NPP provides repeated ecological measurement but cannot establish whether the measured
productivity is economically relevant to households. Household surveys measure food and total
consumption directly but provide intermittent spatial observations and do not by themselves locate
environmental production constraints. Integrating the two sources at a common village-centred scale
can identify regions in which the two evidence dimensions coincide, without requiring them to be
treated as a single mediated coefficient.

Rainfall measurement creates an additional interpretive problem. Rx5day and annual precipitation
are strongly correlated, and annual MODIS NPP quality depends partly on gap-filled optical inputs
under unfavourable atmospheric conditions. A negative coefficient on annual precipitation in a
joint model is therefore not self-interpreting. The regional dry-spell contribution is credible only
if it remains stable when rainfall quantity, rainfall concentration, and NPP quality support are
handled transparently.

### Conceptual Framework

The revised framework contains three evidence dimensions:

1. **Ecological sensitivity:** annual dry-spell persistence is associated with within-village
   changes in accumulated cropland carbon productivity.
2. **Household economic relevance:** prior-year local cropland productivity is associated with
   household consumption, with food consumption providing the more specification-stable outcome.
3. **Regional concentration:** predetermined geographic characteristics organise places in which
   ecological sensitivity and household relevance may coincide or diverge.

The actionable output is a regional evidence typology. Regions with stronger negative dry-spell
slopes and stronger positive food-consumption slopes become priorities for additional agricultural
monitoring and adaptation assessment; discordant regions motivate different diagnostic questions.
NPP remains an ecological carbon-productivity measure rather than crop yield, and the regional
coincidence of the two dimensions is interpreted as prioritisation evidence rather than a mediated
causal effect.

### Hypotheses and Weakening Evidence

- **H1 — reproducible dry-spell constraint:** longer annual dry spells are associated with lower
  annual cropland NPP nationally and across declared measurement and covariance checks. The claim is
  weakened if its direction changes when annual rainfall is omitted or orthogonalised, or when NPP
  quality support is restricted.
- **H2 — substantively important regional heterogeneity:** the national dry-spell coefficient masks
  regional magnitudes that differ materially in natural units and reject a common regional slope.
  The claim is weakened if regional contrasts disappear after common-model estimation or are not
  compatible with adequately supported continuous surfaces.
- **H3 — household food-consumption relevance:** higher prior-year cropland NPP is positively
  associated with real per-capita food consumption after household-composition and socioeconomic
  adjustment. Total consumption remains a broader, more specification-sensitive benchmark. The
  claim is weakened if the food estimate loses direction or precision under the expanded control
  block and declared exposure sensitivities.
- **H4 — regional concentration of joint relevance:** at least some outcome-blind regions combine
  relatively strong ecological sensitivity with relatively strong food-consumption relevance under
  common regional-interaction models. The claim is weakened if candidate priority regions disappear
  under expanded household controls or if apparent ranking depends on an arbitrary composite score.

No arbitrary equivalence threshold or outcome-standard-deviation rule is used. Magnitudes,
confidence intervals, observed exposure contrasts, analytical support, and stability across declared
specifications are reported directly.

## 3. Data Overview

### Data Sources and Scope

The data sources combine daily gridded maximum temperature and precipitation, annual satellite
NPP and land cover, repeated national household surveys and consumer-price indices, national
public village locations, and predetermined terrain, cropland, population, and road context. The
data scope is national Cambodia: climate coverage spans 1991-2024, cropland NPP spans 2001-2021,
and the spatially linked household analysis uses survey waves from 2007-2021, with 2004 retained
only as a nominal consumption diagnostic.

### Evidence Architecture

| Readable data name | Period | Observation | Coverage and missingness | Quality controls | Research use | Limitation |
|---|---|---|---|---|---|---|
| National Village Climate-Cropland Productivity Panel | 2001-2021 | public village point-year, with 2 km, 5 km, and 10 km buffer variants | 59,135 complete 5 km strict-cropland rows; 186,858 village-buffer-year NPP rows across three radii for 2,966 points | observed-cell counts, valid-day counts, cropland pixel support, recoded QC diagnostics, and filled-days percentage are retained | Stage-1 measurement audit, national fixed-effects estimation, regional heterogeneity, continuous spatial heterogeneity, and robustness | NPP measures annual cropland carbon productivity rather than crop yield or farm revenue |
| Interview-Aligned Household Consumption and Cropland Productivity Data | 2007-2021 | household-wave linked to the most recent completed village-buffer NPP year | 44,290 eligible households linked to 5 km strict-cropland NPP; 43,120 total-consumption and 43,365 food-consumption complete cases | exact interview calendar year, prior NPP year, survey weight, outcome construction, linkage, and complete-case flags are retained | Stage-2 linkage audit, national and regional NPP-consumption estimation, robustness, and continuous diagnostics | repeated cross-sections do not follow the same household over time, and unmatched households cannot enter spatially linked models |
| Predetermined Village Spatial Context and Regionalisation Data | baseline or long-run | mapped village point and spatial-neighbour relation | national public village locations with terrain, slope, cropland, population, roads, long-run climate, and one frozen six-region assignment | outcome variables, regression residuals, coefficient signs, and p-values are excluded from construction; every region passes the frozen support gates | Outcome-blind SKATER regionalisation, regional interaction models, and map interpretation | analytical regions can cross administrative boundaries and describe spatial heterogeneity rather than administrative jurisdictions |

### Coverage and Linkage Rules

- Climate data are not spatially or temporally imputed. Sixteen temperature cells and three
  precipitation cells have no source values; village buffers average the remaining observed cells
  and retain valid-day and included-cell counts.
- Three pairs of current village codes map to the same physical public village point. Stage 1
  collapses those duplicates before estimation so the same physical location is not double
  weighted.
- Strict cropland NPP is the primary ecological measure. Inclusive agriculture is retained as a
  land-cover sensitivity.
- Stage 2 uses the actual interview calendar year minus one. The 2019 survey is correctly split
  between interviews conducted in 2019 and 2020, which link to 2018 and 2019 NPP, respectively.
- Strict-cropland and inclusive-agriculture NPP are materialised at 2, 5, and 10 km. The 5 km
  measure remains primary; the narrower and wider buffers are scale sensitivities.
- Unmatched households remain in the consumption release but cannot enter spatially linked NPP
  models.

### Data Strengths

- Climate shocks are derived from daily observations and expressed in natural units.
- The 35 C threshold is absolute and does not mechanically classify a fixed share of every local
  temperature history as extreme.
- Annual cropland NPP uses native MODIS pixels inside explicit village-centred buffers and retains
  pixel-count and quality-control information.
- CSES food and non-food recall periods are harmonised item by item rather than added without time
  conversion.
- Actual interview-year alignment prevents the cross-calendar-year 2019 fieldwork from receiving
  the same prior-year NPP exposure mechanically.
- The same public village coordinates support panel estimation, spatial regionalisation, and local
  coefficient mapping.

### Material Limitations

- CSES is a repeated cross-section, not a household panel.
- NPP is not crop yield, farm revenue, or crop-specific production.
- Consumption instruments change across waves, particularly in 2019. Survey-wave fixed effects
  and instrument-regime diagnostics are therefore mandatory.
- CSES 2004 lacks comparable CPI and imputed-rent support and is excluded from the real-price main
  outcome.
- Spatial regionalisation and GWR describe heterogeneity; they do not independently identify
  causal mechanisms.
- A positive relationship in both stages is consistent with, but does not prove, a mediated causal
  pathway from climate through NPP to household consumption.

## 4. Variable Construction  /  Key Variables

### Stage-1 Climate and Ecological Variables

| Readable variable | Readable data name | Unit | Construction | Role | Final variable |
|---|---|---|---|---|---|
| Annual Heat Days at or Above 35 C | National Village Climate-Cropland Productivity Panel | days/year | count of calendar days with daily maximum temperature at or above 35 C | primary heat exposure; report per 10 days | yes |
| Annual Heat Degree-Days Above 35 C | National Village Climate-Cropland Productivity Panel | degree-C days/year | sum of daily maximum temperature exceedance above 35 C | alternative heat-intensity exposure; report per 10 degree-days | yes |
| Annual Maximum Consecutive Five-Day Precipitation Rx5day | National Village Climate-Cropland Productivity Panel | mm | maximum rolling sum of precipitation over five consecutive calendar days | extreme-rainfall exposure; report per 10 mm | yes |
| Annual Maximum Consecutive Dry Days Below 1 mm | National Village Climate-Cropland Productivity Panel | days | longest annual run with daily precipitation below 1 mm | dry-spell exposure; report per 10 days | yes |
| Annual Precipitation Total | National Village Climate-Cropland Productivity Panel | mm/year | sum of daily precipitation over the calendar year | rainfall-quantity control; report per 100 mm | yes |
| Annual Strict-Cropland Mean NPP | National Village Climate-Cropland Productivity Panel | kg C/m2/year | mean annual MODIS NPP among same-year strict-cropland pixels within the village buffer | primary stage-1 outcome and input to the timed stage-2 exposure; report per 0.1 kg C/m2 | yes |
| Annual Inclusive-Agriculture Mean NPP | National Village Climate-Cropland Productivity Panel | kg C/m2/year | corresponding mean including cropland-natural vegetation mosaics | land-cover sensitivity and input to its timed stage-2 counterpart | yes |
| NPP Pixel Support | National Village Climate-Cropland Productivity Panel | count/share | candidate pixels, valid pixels, valid-pixel share, and recoded QC diagnostics | sample-quality gate | yes |
| Mean Annual Strict-Cropland NPP Filled-Days Percentage | National Village Climate-Cropland Productivity Panel | percent | mean percentage of growing-season days for which gap-filled FPAR/LAI inputs contribute to annual strict-cropland NPP pixels within the village buffer; lower values indicate less gap filling | continuous quality adjustment and outcome-independent P50/P75 quality-support sensitivity | yes |

Heat-day count and heat degree-days have a 5 km correlation of 0.93. They are therefore estimated
in separate heat specifications. Regression values are not standardised.

### Stage-2 Household Variables

| Readable variable | Readable data name | Unit | Construction | Role | Final variable |
|---|---|---|---|---|---|
| Real 2021 Annual Total Consumption per Capita | Interview-Aligned Household Consumption and Cropland Productivity Data | 2021 riels/person/year | annualised food, recall non-food, housing services, and where required education, deflated by the applicable CPI and divided by household size | primary stage-2 outcome; log used in regression | yes |
| Real 2021 Annual Food Consumption per Capita | Interview-Aligned Household Consumption and Cropland Productivity Data | 2021 riels/person/year | seven-day food value multiplied by 52, deflated with interview-month food CPI, divided by household size | secondary stage-2 outcome; log used in regression | yes |
| Interview Calendar Year | Interview-Aligned Household Consumption and Cropland Productivity Data | year | actual household interview calendar year; the 2019 survey retains interviews conducted in both 2019 and 2020 | timing anchor | yes |
| Prior NPP Calendar Year | Interview-Aligned Household Consumption and Cropland Productivity Data | year | Interview Calendar Year minus one | exact annual NPP linkage key | yes |
| Prior-Year Strict-Cropland NPP | Interview-Aligned Household Consumption and Cropland Productivity Data | kg C/m2/year | Annual Strict-Cropland Mean NPP in Prior NPP Calendar Year within 5 km | focal stage-2 exposure; report per 0.1 kg C/m2 | yes |
| Prior-Year Inclusive-Agriculture NPP | Interview-Aligned Household Consumption and Cropland Productivity Data | kg C/m2/year | Annual Inclusive-Agriculture Mean NPP in Prior NPP Calendar Year within 5 km | stage-2 land-cover sensitivity | yes |
| Prior-Year Strict-Cropland NPP at 2 km | Interview-Aligned Household Consumption and Cropland Productivity Data | kg C/m2/year | strict-cropland NPP in Prior NPP Calendar Year within 2 km | narrow-buffer stage-2 sensitivity | yes |
| Prior-Year Inclusive-Agriculture NPP at 2 km | Interview-Aligned Household Consumption and Cropland Productivity Data | kg C/m2/year | inclusive-agriculture NPP in Prior NPP Calendar Year within 2 km | combined land-cover and narrow-buffer sensitivity | yes |
| Prior-Year Strict-Cropland NPP at 10 km | Interview-Aligned Household Consumption and Cropland Productivity Data | kg C/m2/year | strict-cropland NPP in Prior NPP Calendar Year within 10 km | wide-buffer stage-2 sensitivity | yes |
| Prior-Year Inclusive-Agriculture NPP at 10 km | Interview-Aligned Household Consumption and Cropland Productivity Data | kg C/m2/year | inclusive-agriculture NPP in Prior NPP Calendar Year within 10 km | combined land-cover and wide-buffer sensitivity | yes |
| Prior-Year NPP Pixel Support | Interview-Aligned Household Consumption and Cropland Productivity Data | count/share | candidate pixels, valid pixels, valid-pixel share, recoded QC percentage, and original above-100 QC pixel count for each radius and land-cover definition | stage-2 quality and support gate | yes |
| Household Composition Vector | Interview-Aligned Household Consumption and Cropland Productivity Data | natural shares/counts | household size, female share, mean age, child share, older-person share, and dependency ratio | prespecified demographic controls | yes |
| Household Head Ever Attended School | Interview-Aligned Household Consumption and Cropland Productivity Data | binary | education response for the unique household member coded as household head | household-head education control | yes |
| Socioeconomic Control Vector | Interview-Aligned Household Consumption and Cropland Productivity Data | natural units/categories | urban-rural status, agricultural participation, and Household Head Ever Attended School | prespecified sensitivity controls | yes |
| Household Survey Weight | Interview-Aligned Household Consumption and Cropland Productivity Data | survey weight | harmonised released household weight | primary stage-2 weighting | yes |
| Consumption Instrument Regime | Interview-Aligned Household Consumption and Cropland Productivity Data | category | 2007; 2009-2013; 2014-2017; and 2019-2021 questionnaire regimes | diagnostic and sensitivity interaction | yes |
| Stage-2 Complete-Case Flags | Interview-Aligned Household Consumption and Cropland Productivity Data | binary | positive survey weight, mapped 5 km strict-cropland prior-year NPP, required outcome, and complete prespecified control block | reproducible analytical-sample gate | yes |

Food is annualised from the prior seven days by multiplying by 52. Non-food components are
annualised using the item-specific 1-, 3-, 6-, or 12-month recall period. Housing services combine
actual or equivalent rent, utilities, and non-duplicated maintenance. Education is added separately
in 2019 and 2021 because it is absent from those waves' recall non-food list. No monetary outcome is
winsorised and no item nonresponse is imputed.

The interview-aligned household release contains 77,904 household-wave records. Among 62,526
households eligible for the real total-consumption outcome, strict-cropland NPP links to 42,886 at
2 km, 44,290 at 5 km, and 44,986 at 10 km. The primary 5 km model has 43,120 complete total-
consumption observations after applying the outcome, weight, NPP-linkage, and Household Composition
Vector gates; the corresponding food-consumption sample has 43,365 observations. Missing values
remain missing, and no regression variable is winsorised, clipped, or standardised.

### Spatial Regionalisation Variables

SKATER uses elevation, slope, long-run mean annual rainfall, long-run means of the four
climate-shock measures, cropland share, baseline population, and historical or predetermined road
accessibility. Features are standardised only for clustering. NPP, consumption, regression
residuals, coefficient signs, and p-values are excluded from region construction. REDCAP was
evaluated only as a pre-outcome candidate-selection diagnostic and is not a final analysis variable.

Candidate region counts were restricted to four through seven and evaluated before regional
outcome coefficients were inspected. One six-region SKATER partition is frozen. Every selected
region must contain at least 200 villages, an average of at least 15 complete stage-1 years per
village, and at least 3,000 stage-2 households.

For interpretation, the frozen codes have descriptive geographic labels. R1 is the northern-
northwestern interior and northern Tonle Sap arc, centred on Banteay Meanchey, Siem Reap, and
Kampong Thom and extending into parts of Battambang and Preah Vihear. R2 is the western Tonle Sap
agricultural belt, principally Battambang, Pursat, and Pailin. R3 is the eastern-northeastern Mekong
corridor and uplands, spanning Tboung Khmum, Kratie, Stung Treng, Ratanak Kiri, and Mondul Kiri. R4
is the lower Mekong and southeastern plains, principally Kandal, Prey Veng, Kampong Cham, and Svay
Rieng. R5 is the south-central interior plains, centred on Kampong Speu, Takeo, Kampong Chhnang, and
part of Kandal. R6 is the southern coastal belt, spanning Kampot, Preah Sihanouk, Koh Kong, Kep,
and southern Takeo. These labels summarize the geography of analytical clusters rather than define
administrative regions; cluster boundaries can cross province boundaries.

| Readable variable | Readable data name | Unit | Construction | Role | Final variable |
|---|---|---|---|---|---|
| CSES Public-Point Linkage | Interview-Aligned Household Consumption and Cropland Productivity Data | binary and identifier | deterministic link from a CSES village code to a unique national public village point | stage-2 NPP linkage and survey-support map | yes |
| Village Spatial Adjacency | Predetermined Village Spatial Context and Regionalisation Data | binary edge list | shared-neighbour graph among mapped village locations | spatial contiguity constraint | yes |
| SKATER Region ID | Predetermined Village Spatial Context and Regionalisation Data | category | frozen six-region minimum-spanning-tree partition of standardised outcome-blind clustering features | primary discrete spatial region | yes |

### Timing Alignment

- Stage 1 uses climate shocks and cropland NPP from the same calendar year \(t\).
- Stage 2 sets Prior NPP Calendar Year equal to Interview Calendar Year minus one and links all NPP
  definitions through that exact year. In the 2019 survey release, 5,034 households were interviewed
  in 2019 and 5,041 in 2020, so their prior NPP years are 2018 and 2019, respectively.
- No early-versus-late or other temporal sample split is part of the primary design.
- Survey-wave and, when available, interview-time fixed effects absorb common survey timing.

## 5. Identification Strategy

### Design Principle

The study estimates two linked conditional relationships at three spatial resolutions. Stage 1
uses within-village annual variation to quantify climate-NPP sensitivity. Stage 2 uses linked
repeated household cross-sections to quantify the household economic relevance of prior-year local
NPP. A national model provides the benchmark, the frozen outcome-blind regions identify coherent
departures from that benchmark, and continuous Stage-1 surfaces test whether the regional pattern
extends beyond imposed borders.

The primary synthesis compares regional ecological sensitivity with regional food-consumption
relevance. Each dimension retains its own estimate, uncertainty, and analytical support. This
side-by-side comparison produces a transparent regional prioritisation typology without converting
the two relationships into a product-of-coefficients estimate.

### Stage-1 Identification

Village fixed effects remove persistent location characteristics, including average terrain,
market access, and long-run agricultural suitability. Year fixed effects absorb national annual
shocks. Identification therefore comes from within-village changes in annual climate exposure and
NPP relative to common year conditions. Annual rainfall total separates rainfall quantity from
Rx5day and consecutive dry-day structure.

The primary 5 km model uses Annual Strict-Cropland Mean NPP and retains NPP Pixel Support as an
explicit quality gate. Annual Heat Days at or Above 35 C is the primary heat exposure. Annual Heat
Degree-Days Above 35 C replaces, rather than accompanies, the heat-day count in the alternative
heat-intensity model. The two heat definitions are compared on the same outcome support. Primary
inference clusters standard errors by village. Two-way village-and-year clustering, 0.5-degree
spatial-block clustering, and Conley spatial HAC are prespecified sensitivity estimators because
climate and ecological residuals may remain serially or spatially correlated. Conley inference
uses a Bartlett kernel, a 50 km focal cutoff, and 25 km and 100 km cutoff sensitivities; covariance
pairs are formed within calendar year after fixed-effect residualisation.

The revised rainfall and measurement audit adds four outcome-independent checks. First, Annual
Precipitation Total is omitted to test whether the dry-spell estimate depends on conditioning on a
highly correlated rainfall quantity. Second, Annual Precipitation Total is residualised against
Rx5day with village and year effects before entering the NPP model, so rainfall concentration and
residual annual quantity are reported separately. Third, Mean Annual Strict-Cropland NPP Filled-
Days Percentage enters as a continuous quality covariate. Fourth, the model is repeated in the
lower-P75 and lower-P50 portions of that quality measure, with thresholds fixed from the pooled
quality distribution before estimating the outcome model. These checks evaluate the dry-spell
coefficient; the conditional annual-rainfall coefficient is not treated as an agricultural-damage
estimand.

The resulting coefficients quantify within-village annual climate sensitivity of cropland carbon
productivity. National estimates populate National Climate-to-NPP Regression Results and National
Climate-to-NPP Responses; robustness evidence is separated into the corresponding appendix outputs.

### Stage-2 Identification

The household model links consumption to prior-year NPP, includes exact survey-time fixed effects,
uses released survey weights, and controls for household composition. Standard errors are clustered
by linked village because households in the same village and survey period share the NPP exposure.
Real 2021 Annual Total Consumption per Capita remains the originally prespecified benchmark.
Real 2021 Annual Food Consumption per Capita is the focal outcome for the revised household-
relevance claim because it retains direction and precision under the expanded socioeconomic control
block. Both enter in logarithms, while Prior-Year Strict-Cropland NPP remains in natural units.

Household Composition Vector is the original prespecified control block. Socioeconomic Control
Vector is required for the revised food-consumption headline and regional-priority estimates;
composition-only regional estimates remain visible as the original specification. Consumption
Instrument Regime, alternative location fixed effects, and alternative NPP definitions remain
sensitivities. A location fixed-effect sensitivity separates within-location temporal association
from persistent between-place differences but does not replace the repeated-cross-section estimand.

Stage 2 measures the household economic relevance of local cropland productivity and is not used as
a product-of-coefficients mediation model. Exact interview-year alignment is complete: Prior NPP
Calendar Year equals Interview Calendar Year minus one for every timed household record. National
NPP-to-Consumption Regression Results therefore uses the frozen interview-aligned stage-2 sample.

### Outcome-Blind Regionalisation

SKATER uses Village Spatial Adjacency and the standardised, outcome-blind feature set declared in
Section 4. NPP, consumption, residuals, coefficient signs, and p-values are not used to construct
regions. Four-through-seven-region SKATER and REDCAP candidates were evaluated before regional
outcome coefficients were inspected. The six-region SKATER solution was retained because it had
higher perturbation stability and lower mean within-region geographic distance while passing the
frozen support gate of at least 200 villages, 15 complete stage-1 years per village on average, and
3,000 stage-2 households in every region. REDCAP remains a candidate-selection audit only.

Zone-specific slopes are estimated through interactions in the common national sample rather than
through independently selected regional subsamples. A joint test of slope equality is evaluated
before individual regional intervals are interpreted. The frozen map and regional slopes populate
Outcome-Blind Regions and Zonal Climate-to-NPP Responses.

The reader-facing map identifies the frozen codes with the short labels north/northwest (R1),
western Tonle Sap (R2), east/northeast Mekong (R3), lower Mekong/southeast (R4), south-central
interior (R5), and southern coast (R6). The labels are descriptive aids and do not alter the
outcome-blind algorithm, frozen village assignments, or region-specific estimands.

### Regional Evidence Typology

Regional Dry-Spell Sensitivity and Food-Consumption Relevance and Regional Ecological-Economic
Priority Evidence compare the six regional coefficients without collapsing them into one index.
The ecological axis translates each regional dry-spell slope over the pooled P10-to-P90 dry-spell
contrast and expresses the result in kg C/ha and as a percentage of that region's mean annual NPP.
The household axis translates the expanded-control food-consumption slope for a 0.1 kg C/m2 NPP
increment. National estimates form the reference lines. A region is classified as jointly elevated
only when both point estimates exceed their corresponding national magnitude and both 95 percent
intervals exclude zero. Other regions are labelled ecological-only, household-relevance-only, or
mixed/discordant according to the two displayed dimensions. The rule is fixed before the expanded-
control regional estimates are inspected.

### Continuous Spatial Heterogeneity

Geographically weighted models apply the global model's timing controls and covariate structure
before local weighting. An adaptive nearest-neighbour bandwidth is selected without reference to
local coefficient significance. Local coefficients are accompanied by bandwidth, effective-sample,
and uncertainty diagnostics so that unsupported local extremes are not interpreted as geographic
findings. The continuous surfaces test whether discrete regional differences represent smooth and
reproducible spatial structure or sensitivity to selected boundaries. The main-text continuous
figure is restricted to the two climate-to-NPP surfaces, which have strong local information and
stable signs across adjacent bandwidths. The NPP-to-consumption surfaces are retained as an
appendix diagnostic because their substantially smaller effective local samples and mostly
zero-crossing local intervals do not support pointwise geographic claims.

### Evidence Classification

- H1 is supported when the dry-spell coefficient is negative, substantively interpretable in
  natural units, and stable under rainfall, NPP-quality, buffer, land-cover, and covariance checks.
- H2 is supported when the regional dry-spell equality test rejects, translated regional magnitudes
  differ materially, and the pattern is compatible with adequately supported continuous surfaces.
- H3 is supported when the expanded-control food-consumption coefficient is positive nationally
  and remains positive in the declared exposure sensitivities; total consumption is reported as the
  broader benchmark rather than a required confirmation.
- H4 is supported when at least one outcome-blind region meets the fixed jointly elevated rule under
  expanded household controls. A significant regional equality test without qualifying regional
  estimates is evidence of heterogeneity but not of a priority region.
- A stable direction with an interval spanning zero is described as imprecise rather than as
  evidence of no relationship. No arbitrary outcome-standard-deviation threshold is used.

The completed evidence supports all four hypotheses at their declared scope. For H1, the national
dry-spell coefficient is -0.002152 kg C m-2 per ten additional dry days and remains negative when
annual rainfall is omitted (-0.002304), residualised (-0.002152), continuously adjusted for NPP
quality (-0.002478), or restricted to the lower 75% (-0.002175) and lower 50% (-0.002244) of the
gap-filling distribution. H2 is supported by the rejected regional slope-equality test and natural-
unit differences: the pooled P10-to-P90 dry-spell contrast corresponds to a 1.46% national NPP
decline, compared with 2.64% in R4 and 3.76% in R6. H3 is supported by the positive expanded-control
food-consumption association nationally (2.13% per 0.1 kg C m-2 NPP); total consumption remains the
broader, more specification-sensitive benchmark. H4 is supported because R4 and R6 meet the fixed
jointly elevated rule, while R1 shows household relevance without above-national ecological
sensitivity and R5 provides a discordant comparison.

### Interpretation Boundaries

- Rx5day and consecutive dry days are meteorological exposures rather than observed disasters, and
  NPP is carbon productivity rather than crop yield or farm income.
- The two regional dimensions support prioritisation and additional field assessment; multiplying
  their coefficients or attributing intervention effects falls outside the design.
- Region boundaries remain frozen independently of outcomes, and isolated local coefficients are
  interpreted only with adequate support and compatibility with regional evidence.

## 6. Main Estimation Framework

### Notation and Interpretation Limits

All formulas below reuse the same symbols whenever the analytical unit and quantity are unchanged;
later formulas define only new symbols. Coefficients quantify the stated within-village panel or
survey-weighted repeated-cross-section relationships under their declared controls. The regional
synthesis keeps the ecological and household coefficients as separate evidence dimensions.

### Stage 1: Annual Climate Shocks and Cropland NPP

For public village point \(v\) and calendar year \(t\), the primary model is:

\[
NPP_{vt} = \alpha_v + \lambda_t + \beta_H Heat35_{vt}
+ \beta_R Rx5day_{vt} + \beta_D CDD_{vt}
+ \beta_P RainTotal_{vt} + \epsilon_{vt}.
\]

Here, \(NPP_{vt}\) is Annual Strict-Cropland Mean NPP; \(Heat35_{vt}\) is Annual Heat Days at or
Above 35 C; \(Rx5day_{vt}\) is Annual Maximum Consecutive Five-Day Precipitation Rx5day;
\(CDD_{vt}\) is Annual Maximum Consecutive Dry Days Below 1 mm; and \(RainTotal_{vt}\) is Annual
Precipitation Total. The terms \(\alpha_v\) and \(\lambda_t\) are village and calendar-year fixed
effects. The coefficients \(\beta_H\), \(\beta_R\), \(\beta_D\), and \(\beta_P\) are the conditional
climate-NPP slopes, and \(\epsilon_{vt}\) is the error term.

The alternative heat-intensity specification is:

\[
NPP_{vt} = \alpha_v + \lambda_t + \beta_{HDD} HDD35_{vt}
+ \beta_R Rx5day_{vt} + \beta_D CDD_{vt}
+ \beta_P RainTotal_{vt} + \epsilon_{vt}.
\]

The new term \(HDD35_{vt}\) is Annual Heat Degree-Days Above 35 C and \(\beta_{HDD}\) is its
conditional slope. Annual Heat Days at or Above 35 C and Annual Heat Degree-Days Above 35 C are
never entered together in a primary specification.

To test whether the Rx5day relationship is non-monotonic, the linear Rx5day term is replaced by a
restricted cubic spline whose knots are fixed at the pooled 10th, 50th, and 90th percentiles of
Annual Maximum Consecutive Five-Day Precipitation Rx5day before outcome estimation:

\[
NPP_{vt} = \alpha_v + \lambda_t + \beta_H Heat35_{vt}
+ \sum_{m=1}^{M} \rho_m B_m(Rx5day_{vt})
+ \beta_D CDD_{vt} + \beta_P RainTotal_{vt} + \epsilon_{vt}.
\]

The terms \(B_m(\cdot)\) are the restricted-cubic-spline basis functions, \(m\) indexes those
functions, \(M\) is the number of basis terms implied by the fixed knots, and \(\rho_m\) are the
corresponding coefficients. The plotted curve reports adjusted predicted NPP differences over
observed common support rather than extrapolating beyond the data.

National Climate-to-NPP Regression Results follows a fixed ladder: a heat-only fixed-effect model,
addition of Annual Precipitation Total, the full primary model above, and the full alternative
degree-day model. National Climate-to-NPP Responses reports the fully adjusted coefficients and the
flexible Rx5day curve. Coefficients are displayed for 10 heat days, 10 degree-days, 10 mm Rx5day,
10 consecutive dry days, and 100 mm annual precipitation; the estimation itself uses unstandardised
natural-unit variables.

Declared stage-1 sensitivities are Annual Inclusive-Agriculture Mean NPP, matched 2 km and 10 km
buffers, NPP Pixel Support restrictions, alternative heat degree-days, two-way clustering,
spatial-block clustering, Conley spatial HAC, and leave-one-year diagnostics. The Conley focal
cutoff is 50 km, with 25 km and 100 km sensitivities. The robustness plan assesses sign, magnitude,
interval overlap, and sample support and never selects an exposure definition by its p-value.

The rainfall-omission sensitivity removes Annual Precipitation Total while retaining the other
three primary climate regressors:

\[
NPP_{vt} = \alpha_v + \lambda_t + \beta_H^{(-P)} Heat35_{vt}
+ \beta_R^{(-P)} Rx5day_{vt} + \beta_D^{(-P)} CDD_{vt} + \epsilon_{vt}^{(-P)}.
\]

The superscript \((-P)\) denotes the specification without Annual Precipitation Total. The focal
comparison is between \(\beta_D^{(-P)}\) and the primary \(\beta_D\).

For the orthogonalised-rainfall sensitivity, Annual Precipitation Total is first decomposed as:

\[
RainTotal_{vt} = a_v + l_t + \pi Rx5day_{vt} + r_{vt}.
\]

Here, \(a_v\) and \(l_t\) are village and year effects in the auxiliary rainfall model, \(\pi\) is
the conditional Rx5day-rainfall slope, and \(r_{vt}\) is Residual Annual Precipitation Total. The
outcome model is then:

\[
NPP_{vt} = \alpha_v + \lambda_t + \beta_H^{(r)} Heat35_{vt}
+ \beta_R^{(r)} Rx5day_{vt} + \beta_D^{(r)} CDD_{vt}
+ \beta_P^{(r)} r_{vt} + \epsilon_{vt}^{(r)}.
\]

The superscript \((r)\) identifies the residual-rainfall specification. This decomposition makes
the Rx5day coefficient inclusive of the annual-rainfall component statistically associated with
Rx5day while \(\beta_P^{(r)}\) describes residual rainfall quantity.

The continuous NPP-quality adjustment adds Mean Annual Strict-Cropland NPP Filled-Days Percentage,
denoted \(QC_{vt}\):

\[
NPP_{vt} = \alpha_v + \lambda_t + \beta_H^{(Q)} Heat35_{vt}
+ \beta_R^{(Q)} Rx5day_{vt} + \beta_D^{(Q)} CDD_{vt}
+ \beta_P^{(Q)} RainTotal_{vt} + \kappa QC_{vt} + \epsilon_{vt}^{(Q)}.
\]

The new coefficient \(\kappa\) adjusts for the mean percentage of growing-season days whose annual
NPP inputs were gap filled. The primary model is also estimated after restricting \(QC_{vt}\) to
its pooled P75 and P50 thresholds, which are fixed before those outcome regressions are run.

For the within-transformed Stage-1 design matrix, Conley covariance at spatial cutoff \(c\) is:

\[
\widehat{V}_{Conley}(c) =
(\widetilde{X}'\widetilde{X})^{-1}
\left[
\sum_t \sum_{i \in t}\sum_{j \in t}
K\left(\frac{d_{ij}}{c}\right)
\widehat{u}_i\widehat{u}_j
\widetilde{x}_i\widetilde{x}_j'
\right]
(\widetilde{X}'\widetilde{X})^{-1}.
\]

Here, \(\widetilde{X}\) is the regressor matrix after village and calendar-year fixed-effect
residualisation; \(\widetilde{x}_i\) is its row for observation \(i\); \(\widehat{u}_i\) is the
corresponding residual; \(d_{ij}\) is the distance in kilometres between villages \(i\) and
\(j\); \(c\) is 25, 50, or 100 km; and
\(K(a)=\max(1-a,0)\) is the Bartlett kernel. The summation is restricted to observations in the
same calendar year. This sensitivity complements rather than replaces village-clustered primary
inference and two-way clustering.

### Stage 2: Cropland NPP and Household Consumption

For household \(h\), linked village \(v\), survey time \(s\), and outcome \(j\), the household
model is:

\[
\log(C_{hvs}^{j}) = \tau_s + \theta_j NPP_{v,s-1}
+ X_{hvs}'\gamma_j + \eta_{hvs}^{j}.
\]

Here, \(C_{hvs}^{j}\) is Real 2021 Annual Total Consumption per Capita when \(j=Total\) and Real
2021 Annual Food Consumption per Capita when \(j=Food\); \(NPP_{v,s-1}\) is Prior-Year
Strict-Cropland NPP; \(\tau_s\) is a survey-wave or exact survey-time fixed effect; \(X_{hvs}\) is
Household Composition Vector; \(\gamma_j\) is its coefficient vector; \(\theta_j\) is the
outcome-specific NPP slope; and \(\eta_{hvs}^{j}\) is the error term. Estimation uses Household
Survey Weight and village-clustered standard errors.

For a 0.1 kg C/m2 increase in Prior-Year Strict-Cropland NPP, the reported percentage difference
in consumption is:

\[
\Delta_j(0.1) = 100 \left[\exp(0.1\theta_j)-1\right].
\]

The new term \(\Delta_j(0.1)\) is the model-implied percentage difference in outcome \(j\) for the
declared NPP increment. National NPP-to-Consumption Regression Results retains the original fixed
ladder for both outcomes: survey-time controls only, addition of Household Composition Vector,
addition of Socioeconomic Control Vector, and a location-fixed-effect sensitivity. The original
composition-adjusted total-consumption model remains identified as prespecified; the expanded-
control food-consumption model supplies the revised household-relevance headline.

Declared stage-2 sensitivities interact the NPP slope with Consumption Instrument Regime, compare
Prior-Year Strict-Cropland NPP with Prior-Year Inclusive-Agriculture NPP, compare the 5 km primary
measure with Prior-Year Strict-Cropland NPP at 2 km and Prior-Year Strict-Cropland NPP at 10 km,
compare prior-year and contemporaneous timing, report the expanded-control sample change, and
repeat simpler models on the expanded-control common sample. These checks appear in
NPP-to-Consumption Robustness Results rather than changing the primary model.

### Regional Slope Models

For either stage, region-specific slopes are estimated in the common national sample. For
observation \(i\), focal exposure \(Z_i\), predetermined region \(k\), and \(K\) regions:

\[
Y_i = FE_i + W_i'\delta + \sum_{k=1}^{K}\mu_k R_{ik}
+ \sum_{k=1}^{K}\phi_k Z_i R_{ik} + u_i.
\]

Here, \(Y_i\) is the relevant NPP or log-consumption outcome; \(FE_i\) denotes the corresponding
global model's fixed effects; \(W_i\) contains the remaining approved covariates; \(\delta\) is
their coefficient vector; \(R_{ik}\) indicates membership in region \(k\); \(\mu_k\) is the region
main effect when it is not absorbed by fixed effects; \(\phi_k\) is the region-specific slope of
focal exposure \(Z_i\); and \(u_i\) is the error term. The same specification is run with SKATER
Region ID. The number of regions is frozen at \(K=6\) using the outcome-blind rules in Section 5.
Stage-1 regional estimates retain the common Annual Precipitation Total control. The revised
Stage-2 regional food model includes both Household Composition Vector and Socioeconomic Control
Vector; the original composition-only regional estimates remain reported as a comparison.

The prespecified joint test is:

\[
H_0: \phi_1 = \phi_2 = \cdots = \phi_K.
\]

The null hypothesis \(H_0\) states that the focal slope is equal across all regions. Regional
coefficients are interpreted as evidence of heterogeneity only after considering this joint test,
their confidence intervals, sample support, and compatibility with the continuous local surface.

Let \(q_{0.10}\) and \(q_{0.90}\) be the pooled P10 and P90 of Annual Maximum Consecutive Dry Days
Below 1 mm, let \(\phi_{Dk}\) be the Region \(k\) dry-spell coefficient reported per 10 days, and let
\(\overline{NPP}_k\) be that region's mean Annual Strict-Cropland Mean NPP. The translated regional
NPP change and proportional magnitude are:

\[
\Delta NPP_k^{10-90} = \phi_{Dk}\frac{q_{0.90}-q_{0.10}}{10},
\]

\[
M_k^{10-90} = 100\frac{\Delta NPP_k^{10-90}}{\overline{NPP}_k}.
\]

The first quantity is also multiplied by 10,000 to report kg C/ha. Let \(\theta_{Fk}^{(S)}\) be the
expanded-control regional slope for log Real 2021 Annual Food Consumption per Capita. Its separate
household translation is:

\[
\Delta Food_k(0.1) = 100\left[\exp\left(0.1\theta_{Fk}^{(S)}\right)-1\right].
\]

Regional Dry-Spell Sensitivity and Food-Consumption Relevance plots \(-M_k^{10-90}\) against
\(\Delta Food_k(0.1)\), with 95 percent intervals propagated from each coefficient and national
estimates shown as reference lines. Regional Ecological-Economic Priority Evidence reports the same
two dimensions, support counts, and the fixed classification rule from Section 5.

### Geographically Weighted Models

The continuous model residualises the outcome and focal regressors using the corresponding global
fixed effects and approved controls, then estimates a locally weighted relationship at target
location \(s_0\):

\[
\widehat{\psi}(s_0) = \arg\min_{\psi}
\sum_i q_i w\left[d(s_i,s_0);b(s_0)\right]
\left(\widetilde{Y}_i-\widetilde{W}_i'\psi\right)^2.
\]

Here, \(s_i\) is observation \(i\)'s mapped village location; \(d(s_i,s_0)\) is its distance from
target location \(s_0\); \(w[\cdot]\) is an adaptive nearest-neighbour kernel; \(b(s_0)\) is the
local bandwidth; \(\widetilde{Y}_i\) and \(\widetilde{W}_i\) are the outcome and regressors after
the corresponding global adjustment; \(q_i\) equals one in stage 1 and Household Survey Weight in
stage 2; \(\psi\) is the vector of local slopes; and \(\widehat{\psi}(s_0)\) is its estimate.

The neighbour count is selected once per model by corrected Akaike Information Criterion, with
cross-validation recorded as a sensitivity. Continuous Spatial Heterogeneity reports the two
climate-to-NPP surfaces. Continuous NPP-to-Consumption Spatial Diagnostics contains the two
household-outcome surfaces as appendix diagnostics. GWR Bandwidth and Effective-Sample Diagnostics
and GWR and Local-Regression Diagnostics report bandwidth, coefficient dispersion, uncertainty,
effective local sample size, and multiple-comparison support. GWR remains a spatial heterogeneity
diagnostic rather than a separate causal estimator.

## 7. Analytical Workflow

| analysis step | detailed question | variables used | formula/model used | data processing | generated figure/table title | claim evaluated | success condition | failure condition | support status |
|---|---|---|---|---|---|---|---|---|---|
| Stage-1 Sample and Linkage Audit | Supporting Point 1 | NPP Pixel Support; Annual Strict-Cropland Mean NPP; Annual Inclusive-Agriculture Mean NPP | linkage, duplicate-location, temporal-support, and attrition audit | collapse duplicate village codes that represent the same physical point; retain buffer, year, pixel-support, and quality counts | Research Design Data Linkage and Analytical Support; Analytical Samples and Variable Definitions | the national ecological benchmark rests on a traceable village-year sample | village locations, annual support, exclusions, and retained observations are reproducible | duplicate locations, temporal gaps, or exclusions cannot be reconciled into one analytical sample | supported; the stage-1 support and linkage audit is materialised |
| Climate and NPP Measurement Audit | Supporting Point 1 | Annual Heat Days at or Above 35 C; Annual Heat Degree-Days Above 35 C; Annual Maximum Consecutive Five-Day Precipitation Rx5day; Annual Maximum Consecutive Dry Days Below 1 mm; Annual Precipitation Total; Annual Strict-Cropland Mean NPP; Annual Inclusive-Agriculture Mean NPP; NPP Pixel Support; Mean Annual Strict-Cropland NPP Filled-Days Percentage | natural-unit distribution, dependence, geography, and measurement-quality audit | aggregate daily climate to village-buffer years; apply strict and inclusive land-cover masks; retain valid-pixel and filled-days support | National Geography of Climate Exposure and Cropland NPP; Climate-Shock Distributions and Correlations; Cropland Definition and NPP Quality Support; Climate-Shock Descriptive Statistics and Correlations; Annual Cropland-NPP Coverage and Quality | the focal exposures and ecological outcome have interpretable variation and adequate support | climate distributions, correlations, spatial coverage, and NPP quality are transparent before estimation | exposure overlap, missing support, or NPP quality cannot be characterised well enough to interpret the national model | supported for the current 5 km stage-1 panel |
| National Climate-to-NPP Estimation | Supporting Point 1 | Annual Heat Days at or Above 35 C; Annual Heat Degree-Days Above 35 C; Annual Maximum Consecutive Five-Day Precipitation Rx5day; Annual Maximum Consecutive Dry Days Below 1 mm; Annual Precipitation Total; Annual Strict-Cropland Mean NPP | two-way village and year fixed-effects models with primary heat-days, alternative heat degree-days, and flexible Rx5day specifications | use the complete national 5 km strict-cropland panel and translate coefficients into the declared natural-unit increments and observed P10-to-P90 dry-spell contrast | National Climate-to-NPP Responses; National Climate-to-NPP Regression Results | prolonged dry spells provide a reproducible national ecological constraint and benchmark magnitude | the dry-spell coefficient has a stable adverse direction and an interpretable natural-unit translation | the dry-spell direction is unstable, the interval is uninformative, or the observed-exposure translation cannot be supported | supported; the dry-spell coefficient is negative and precise while heat-intensity and Rx5day results are more conditional |
| Climate-to-NPP Robustness and Measurement Sensitivity | Supporting Point 5 | Annual Heat Days at or Above 35 C; Annual Heat Degree-Days Above 35 C; Annual Maximum Consecutive Five-Day Precipitation Rx5day; Annual Maximum Consecutive Dry Days Below 1 mm; Annual Precipitation Total; Annual Strict-Cropland Mean NPP; Annual Inclusive-Agriculture Mean NPP; NPP Pixel Support; Mean Annual Strict-Cropland NPP Filled-Days Percentage | rainfall omission and residualisation, continuous and restricted NPP-quality checks, alternative buffers and land cover, alternative inference, and leave-one-year sensitivity | hold the focal estimand fixed while changing one declared rainfall, quality, measurement, spatial-scale, or covariance choice at a time | Climate-to-NPP Robustness Coefficients; Climate-to-NPP Robustness Results | the dry-spell result is not an artefact of rainfall collinearity, wet-year retrieval conditions, or one measurement choice | the dry-spell coefficient remains adverse and comparable across the declared rainfall and NPP-quality treatments | the coefficient changes direction materially, loses interpretable support, or appears only under one rainfall or quality treatment | supported; the dry-spell coefficient remains negative and precise across the focal rainfall and quality checks |
| Outcome-Blind Spatial Regionalisation | Supporting Point 2 | Village Spatial Adjacency; SKATER Region ID | SKATER minimum-spanning-tree regionalisation with four-through-seven-region candidate diagnostics and one frozen six-region solution | standardise predetermined clustering features only; enforce spatial contiguity and minimum stage-1 and stage-2 support; exclude outcomes and fitted results | Outcome-Blind Regions and Zonal Climate-to-NPP Responses | regional boundaries are contiguous, adequately supported, and independent of outcomes | one contiguous solution passes all frozen support gates without using NPP, consumption, coefficients, or p-values | selected regions fail contiguity or support gates, or their construction uses outcome information | supported; one six-region SKATER partition is frozen |
| Regional Climate-to-NPP Heterogeneity Estimation | Supporting Point 2 | SKATER Region ID; Annual Heat Days at or Above 35 C; Annual Maximum Consecutive Five-Day Precipitation Rx5day; Annual Maximum Consecutive Dry Days Below 1 mm; Annual Precipitation Total; Annual Strict-Cropland Mean NPP | common-sample regional slope-interaction model, joint slope-equality test, and P10-to-P90 natural-unit translation | interact the focal climate exposures with the frozen regions in the common national sample and retain the national covariate set | Outcome-Blind Regions and Zonal Climate-to-NPP Responses; Regional Climate-to-NPP Regression Results | the national dry-spell benchmark conceals substantively important regional heterogeneity | the joint equality test rejects and translated regional dry-spell magnitudes differ materially with adequate support | regional slopes are compatible with one common slope or apparent differences are driven by unsupported regions | supported; the dry-spell equality test rejects and regional magnitudes differ materially |
| Continuous Climate-to-NPP Spatial Heterogeneity | Supporting Point 2 | Annual Heat Days at or Above 35 C; Annual Maximum Consecutive Dry Days Below 1 mm; Annual Precipitation Total; Annual Strict-Cropland Mean NPP | geographically weighted residual regression with adaptive bandwidth selection | residualise with the global fixed-effects specification and estimate local slopes using the selected neighbour bandwidth and adjacent-bandwidth checks | Continuous Spatial Heterogeneity; GWR Bandwidth and Effective-Sample Diagnostics; GWR and Local-Regression Diagnostics | the regional pattern reflects broader continuous ecological spatial structure rather than only one partition | local effective samples are adequate and the broad coefficient pattern remains stable across adjacent bandwidths | local support is inadequate or the sign pattern changes materially under adjacent bandwidths | supported for the dry-spell surface; adjacent-bandwidth signs are stable at 94 percent of villages |
| Household Consumption and Lagged-NPP Linkage Audit | Supporting Point 3 | Real 2021 Annual Total Consumption per Capita; Real 2021 Annual Food Consumption per Capita; Interview Calendar Year; Prior NPP Calendar Year; Prior-Year Strict-Cropland NPP; Prior-Year Inclusive-Agriculture NPP; Prior-Year Strict-Cropland NPP at 2 km; Prior-Year Strict-Cropland NPP at 10 km; Prior-Year NPP Pixel Support; Stage-2 Complete-Case Flags; Household Survey Weight; Consumption Instrument Regime; CSES Public-Point Linkage | consumption harmonisation, exact interview-year linkage, missingness, and attrition audit | annualise recall components, apply the appropriate CPI, compute per-capita outcomes, link interview year minus one to NPP, and retain wave-specific support flags | Research Design Data Linkage and Analytical Support; Analytical Samples and Variable Definitions; Household Linkage and Consumption Support by Wave; Consumption Harmonisation by Survey Wave; Household Linkage and Missingness by Survey Wave | stage-2 outcomes and prior-year NPP exposure are comparable and traceable across survey waves | timing, outcome construction, linkage, missingness, and analytical retention are reproducible by wave | survey instruments cannot be harmonised, interview timing is ambiguous, or linkage attrition cannot be characterised | supported; exact timing, three buffer scales, two land-cover definitions, and complete-case flags are materialised |
| National NPP-to-Consumption Estimation | Supporting Point 3 | Prior-Year Strict-Cropland NPP; Stage-2 Complete-Case Flags; Real 2021 Annual Total Consumption per Capita; Real 2021 Annual Food Consumption per Capita; Household Composition Vector; Socioeconomic Control Vector; Household Survey Weight | survey-weighted repeated-cross-section regression ladder with survey-time effects and village-clustered uncertainty | estimate total- and food-consumption models with sequential household-composition and socioeconomic controls and translate a 0.1 kg C/m2 NPP contrast into percentage differences | Cropland NPP and Household Consumption; National NPP-to-Consumption Regression Results | prior-year cropland productivity has household economic relevance, with food consumption as the focal robust outcome | the food-consumption estimate remains positive and precise under the expanded control block | the food-consumption estimate loses direction or precision after the declared controls | supported for food consumption; total consumption remains the more specification-sensitive benchmark |
| NPP-to-Consumption Robustness | Supporting Point 3 | Prior-Year Strict-Cropland NPP; Prior-Year Inclusive-Agriculture NPP; Prior-Year Strict-Cropland NPP at 2 km; Prior-Year Strict-Cropland NPP at 10 km; Prior-Year NPP Pixel Support; Stage-2 Complete-Case Flags; Real 2021 Annual Total Consumption per Capita; Real 2021 Annual Food Consumption per Capita; Household Composition Vector; Socioeconomic Control Vector; Household Head Ever Attended School; Household Survey Weight; Consumption Instrument Regime | controls, timing, instrument-regime, location-effect, land-cover, buffer-scale, and common-sample sensitivity ladder | vary one declared exposure, timing, control, instrument, location-effect, or sample rule at a time while retaining survey weights and village-clustered uncertainty | NPP-to-Consumption Robustness Results | the focal food-consumption relationship is not generated by one questionnaire regime, timing rule, NPP definition, buffer, or changing sample | the food estimate retains its positive direction across the core declared alternatives and its limitations are transparent | the food result exists only for one timing, land-cover, buffer, instrument regime, or selected sample | partially supported; food estimates are more stable than total-consumption estimates, but some specifications lose precision |
| Regional NPP-to-Consumption Heterogeneity Estimation | Supporting Point 4 | Prior-Year Strict-Cropland NPP; Real 2021 Annual Total Consumption per Capita; Real 2021 Annual Food Consumption per Capita; Household Composition Vector; Socioeconomic Control Vector; Household Survey Weight; SKATER Region ID | expanded-control survey-weighted regional slope-interaction models with joint equality tests | interact prior-year NPP with the frozen regions in the linked household sample and retain household-composition and socioeconomic controls | Cropland NPP and Household Consumption; Regional NPP-to-Consumption Regression Results | household economic relevance varies across the same outcome-blind regions used for ecological heterogeneity | the food-consumption slope equality test rejects and region-specific estimates retain adequate household support | regional food slopes are compatible with one common slope or are driven by sparse regional samples | supported for food consumption; the expanded-control regional equality test rejects |
| Regional Ecological-Economic Priority Synthesis | Supporting Point 4 | SKATER Region ID; Annual Maximum Consecutive Dry Days Below 1 mm; Annual Strict-Cropland Mean NPP; Prior-Year Strict-Cropland NPP; Real 2021 Annual Food Consumption per Capita; Household Composition Vector; Socioeconomic Control Vector; Household Survey Weight | separate P10-to-P90 regional NPP translation, expanded-control food-consumption translation, national-reference comparison, and fixed two-dimensional typology | calculate ecological and household translations separately, retain their confidence intervals and sample support, and classify regions without multiplying coefficients | Regional Dry-Spell Sensitivity and Food-Consumption Relevance; Regional Ecological-Economic Priority Evidence | some regions jointly concentrate ecological dry-spell sensitivity and household food-consumption relevance | at least one adequately supported region lies above both fixed national-reference dimensions under expanded household controls | apparent priority regions disappear under expanded controls, require an arbitrary composite score, or lack sample support | supported; R4 and R6 meet the fixed jointly elevated rule while discordant regions remain visible |
| Continuous NPP-to-Consumption Spatial Diagnostics | Supporting Point 4 | Prior-Year Strict-Cropland NPP; Real 2021 Annual Total Consumption per Capita; Real 2021 Annual Food Consumption per Capita; Household Composition Vector; Household Survey Weight | geographically weighted residual regression for the two household outcomes | residualise with the corresponding national household model and report adaptive bandwidth, effective local sample, coefficient dispersion, and interval support | Continuous NPP-to-Consumption Spatial Diagnostics; GWR Bandwidth and Effective-Sample Diagnostics; GWR and Local-Regression Diagnostics | continuous local estimates diagnose whether the regional household pattern is spatially diffuse or concentrated | local effective samples and adjacent-bandwidth behaviour are sufficient for broad descriptive interpretation | local samples are sparse or intervals are too unstable for pointwise interpretation | diagnostic only; local effective samples and interval support are insufficient for a pointwise priority claim |

The analysis proceeds in this order. Spatial regions, bandwidth rules, sample gates, QC thresholds,
and regional typology rules are frozen before the new coefficients are inspected. The national
model remains the reference even when regional magnitudes are stronger. Null, imprecise, or
discordant results remain reportable and are not replaced by post hoc variables, boundaries,
outcomes, or temporal partitions.

Interpretation limits are applied consistently across the workflow: Stage 1 quantifies annual
cropland-carbon sensitivity, Stage 2 quantifies household economic relevance, and the regional
synthesis compares the two estimates without multiplying them into a mediated loss.

### Pre-Estimation Gate

Exact interview-year linkage, 2 km and 10 km NPP, and Prior-Year Inclusive-Agriculture NPP are
complete. The spatial gate is also complete: one six-region SKATER assignment was frozen before
regional outcome coefficients were inspected. Each selected region is contiguous and passes the
approved minimum of 200 villages, 15 complete stage-1 years per village on average, and 3,000
stage-2 households. The NPP-quality P75 and P50 thresholds and the national-reference regional
typology rule were frozen before the new estimates. Rainfall/QC sensitivity estimation, the
expanded-control regional food model, and the fixed regional classification are now complete.

## 8. Figure and Table Plan

The main-text sequence moves from analytical support and national estimates to discrete spatial
heterogeneity, household relevance, and a regional comparison of ecological sensitivity with food-
consumption relevance. Continuous surfaces, measurement, linkage, regionalisation, robustness, and
local-model diagnostics are reserved for the appendix. The old dry-spell-EVI and direct heat-food
outputs remain superseded and are not publication inputs for this plan.

### Figures

| title | what it expresses | figure type | subpanels | key variables | placement | status |
|---|---|---|---:|---|---|---|
| Research Design Data Linkage and Analytical Support | documents national village linkage, temporal support, NPP linkage across buffer scales and land-cover definitions, and sample attrition before any coefficient is interpreted | map and bar | 4 | CSES Public-Point Linkage; NPP Pixel Support; Interview Calendar Year; Prior NPP Calendar Year; Prior-Year Strict-Cropland NPP; Prior-Year Inclusive-Agriculture NPP; Prior-Year Strict-Cropland NPP at 2 km; Prior-Year Inclusive-Agriculture NPP at 2 km; Prior-Year Strict-Cropland NPP at 10 km; Prior-Year Inclusive-Agriculture NPP at 10 km; Prior-Year NPP Pixel Support; Stage-2 Complete-Case Flags; Real 2021 Annual Total Consumption per Capita; Real 2021 Annual Food Consumption per Capita | main | done |
| National Geography of Climate Exposure and Cropland NPP | shows where heat, extreme rainfall, dry spells, and cropland productivity are concentrated across Cambodia | map | 4 | Annual Heat Days at or Above 35 C; Annual Maximum Consecutive Five-Day Precipitation Rx5day; Annual Maximum Consecutive Dry Days Below 1 mm; Annual Strict-Cropland Mean NPP | main | done |
| National Climate-to-NPP Responses | presents the national fixed-effect estimates, the alternative heat-intensity specification, and the flexible Rx5day response in natural units | forest and line | 3 | Annual Heat Days at or Above 35 C; Annual Heat Degree-Days Above 35 C; Annual Maximum Consecutive Five-Day Precipitation Rx5day; Annual Maximum Consecutive Dry Days Below 1 mm; Annual Precipitation Total; Annual Strict-Cropland Mean NPP | main | done |
| Outcome-Blind Regions and Zonal Climate-to-NPP Responses | maps the one frozen six-region SKATER partition and tests whether climate-NPP slopes differ across its contiguous regions | map and forest | 2 | Village Spatial Adjacency; SKATER Region ID; Annual Heat Days at or Above 35 C; Annual Maximum Consecutive Five-Day Precipitation Rx5day; Annual Maximum Consecutive Dry Days Below 1 mm; Annual Strict-Cropland Mean NPP | main | done |
| Cropland NPP and Household Consumption | compares the national regression ladder for total consumption, the food-consumption confirmation, and expanded-control frozen-SKATER regional slopes | forest | 4 | Prior-Year Strict-Cropland NPP; Real 2021 Annual Total Consumption per Capita; Real 2021 Annual Food Consumption per Capita; Household Composition Vector; Socioeconomic Control Vector; Household Survey Weight; SKATER Region ID | main | done |
| Regional Dry-Spell Sensitivity and Food-Consumption Relevance | compares the six regions on natural-unit dry-spell sensitivity and expanded-control food-consumption relevance, with uncertainty and analytical support shown separately | scatter | 1 | SKATER Region ID; Annual Maximum Consecutive Dry Days Below 1 mm; Annual Strict-Cropland Mean NPP; Prior-Year Strict-Cropland NPP; Real 2021 Annual Food Consumption per Capita; Household Composition Vector; Socioeconomic Control Vector; Household Survey Weight | main | done |
| Continuous Spatial Heterogeneity | maps the continuously varying local heat-day and dry-spell relationships with cropland NPP | coefficient map | 2 | Annual Heat Days at or Above 35 C; Annual Maximum Consecutive Dry Days Below 1 mm; Annual Strict-Cropland Mean NPP | appendix | done |
| Climate-Shock Distributions and Correlations | reports natural-unit distributions and dependence among the four shocks, especially overlap between the two heat measures | histogram and heatmap | 5 | Annual Heat Days at or Above 35 C; Annual Heat Degree-Days Above 35 C; Annual Maximum Consecutive Five-Day Precipitation Rx5day; Annual Maximum Consecutive Dry Days Below 1 mm; Annual Precipitation Total | appendix | done |
| Cropland Definition and NPP Quality Support | compares strict and inclusive agriculture coverage and documents pixel support for the ecological outcome | map, histogram, and scatter | 3 | Annual Strict-Cropland Mean NPP; Annual Inclusive-Agriculture Mean NPP; NPP Pixel Support | appendix | done |
| Climate-to-NPP Robustness Coefficients | shows whether national stage-1 coefficients remain stable across rainfall handling, continuous NPP-quality adjustment, quality-support restriction, buffer, NPP definition, functional form, and uncertainty choices | forest | 4 | Annual Heat Days at or Above 35 C; Annual Heat Degree-Days Above 35 C; Annual Maximum Consecutive Five-Day Precipitation Rx5day; Annual Maximum Consecutive Dry Days Below 1 mm; Annual Precipitation Total; Annual Strict-Cropland Mean NPP; Annual Inclusive-Agriculture Mean NPP; NPP Pixel Support; Mean Annual Strict-Cropland NPP Filled-Days Percentage | appendix | done |
| Household Linkage and Consumption Support by Wave | displays exact interview timing, linked and unlinked household support, consumption coverage, and survey-regime changes across waves | bar and line | 4 | CSES Public-Point Linkage; Interview Calendar Year; Prior NPP Calendar Year; Prior-Year Strict-Cropland NPP; Prior-Year Strict-Cropland NPP at 2 km; Prior-Year Strict-Cropland NPP at 10 km; Stage-2 Complete-Case Flags; Real 2021 Annual Total Consumption per Capita; Real 2021 Annual Food Consumption per Capita; Consumption Instrument Regime; Household Survey Weight | appendix | done |
| GWR Bandwidth and Effective-Sample Diagnostics | evaluates whether local coefficient surfaces are supported by adequate adaptive bandwidths and local information | histogram and map | 4 | Annual Strict-Cropland Mean NPP; Prior-Year Strict-Cropland NPP; Real 2021 Annual Total Consumption per Capita; Real 2021 Annual Food Consumption per Capita | appendix | done |
| Continuous NPP-to-Consumption Spatial Diagnostics | reports total- and food-consumption local coefficient surfaces without treating locally imprecise estimates as pointwise findings | coefficient map | 2 | Prior-Year Strict-Cropland NPP; Real 2021 Annual Total Consumption per Capita; Real 2021 Annual Food Consumption per Capita; Household Composition Vector; Household Survey Weight | appendix | done |

### Tables

| title | what it expresses | rows | columns | row meaning | column meaning | placement | status |
|---|---|---:|---:|---|---|---|---|
| Analytical Samples and Variable Definitions | consolidates variable definitions, units, timing, sample support, missingness, and linkage for both stages | about 24 | about 8 | final variable or analytical sample | definition, role, period, unit, observations, locations, missingness, and linkage status | main | done |
| National Climate-to-NPP Regression Results | presents Stargazer-style national fixed-effect results for the primary heat-day and alternative degree-day specifications | about 10 | about 8 | climate coefficient, fixed effect, sample statistic, or fit statistic | sequential specification and alternative heat model | main | done |
| Regional Climate-to-NPP Regression Results | reports frozen-SKATER regional climate slopes and joint slope-equality tests in a common national sample | about 24 | about 9 | region-specific climate coefficient or model statistic | region and focal climate specification | main | done |
| National NPP-to-Consumption Regression Results | presents the survey-weighted regression ladder for total consumption and the food-consumption confirmation | about 14 | about 10 | NPP coefficient, control block, fixed effect, sample statistic, or fit statistic | outcome and sequential household specification | main | done |
| Regional Ecological-Economic Priority Evidence | reports regional dry-spell coefficients, P10-to-P90 NPP translations, expanded-control food-consumption associations, support, and a transparent evidence typology | about 10 | about 9 | region or model-support statistic | dry-spell sensitivity, natural-unit translation, food-consumption relevance, analytical support, and evidence type | main | done |
| Regional NPP-to-Consumption Regression Results | reports expanded-control frozen-SKATER regional NPP-consumption slopes and joint slope-equality tests | about 10 | about 9 | region-specific NPP coefficient or model statistic | region and consumption outcome | appendix | done |
| Consumption Harmonisation by Survey Wave | documents food, non-food, housing, education, CPI, and per-capita construction choices by survey wave | about 10 | about 10 | survey wave or questionnaire regime | source component, recall period, annualisation, deflator, housing, education, eligibility, and observations | appendix | done |
| Climate-Shock Descriptive Statistics and Correlations | reports natural-unit distributions and correlations for the stage-1 exposure family | about 12 | about 10 | climate variable or variable pair | distribution statistic, correlation, period, and support | appendix | done |
| Annual Cropland-NPP Coverage and Quality | documents annual strict and inclusive cropland support and NPP pixel-quality gates | about 21 | about 9 | calendar year | village count, candidate pixels, valid pixels, valid share, strict NPP, inclusive NPP, and exclusions | appendix | done |
| Climate-to-NPP Robustness Results | reports buffer, NPP-definition, support-gate, functional-form, uncertainty, and leave-one-year sensitivities | about 24 | about 9 | robustness specification and focal climate exposure | estimate, standard error, interval, sample, fixed effects, uncertainty method, and stability assessment | appendix | done |
| Household Linkage and Missingness by Survey Wave | documents exact interview-year linkage, lagged NPP availability, outcome missingness, and analytical sample retention | about 10 | about 10 | survey wave or questionnaire regime | released households, outcome-eligible households, mapped villages, NPP-linked households, missingness, and weighted support | appendix | done |
| NPP-to-Consumption Robustness Results | reports alternative outcomes, controls, timing, 2 km and 10 km scales, land-cover definitions, instrument regimes, location effects, and matched-sample checks | about 24 | about 10 | robustness specification and outcome | NPP estimate, standard error, interval, sample, weighting, controls, fixed effects, and stability assessment | appendix | done |
| GWR and Local-Regression Diagnostics | reports adaptive bandwidth selection, effective local sample size, local fit, coefficient dispersion, and multiple-comparison diagnostics | about 20 | about 10 | stage, exposure-outcome pair, or diagnostic | bandwidth, neighbour count, effective sample, local-fit summary, coefficient quantiles, uncertainty, and support flag | appendix | done |

The intended architecture remains six main-text figures and five main-text tables. The two completed
priority outputs replace the continuous Stage-1 surface and the composition-only regional
consumption table in the main sequence; seven figures and eight tables remain in the appendix.
Main tables retain conventional regression-table structure; figures communicate geography,
coefficient patterns, and model support rather than duplicating the same numeric results.

### Variable-Coverage Note

All variables referenced above appear in Section 4 and are marked for final analysis. Exact
interview-year alignment and 2, 5, and 10 km NPP construction are complete. One six-region SKATER
assignment is frozen and the declared regional pre-estimation gate is complete. Thirteen figures and
thirteen tables are complete, including the regional-priority figure and table.
Publication readiness is evaluated by `critique-research-outputs` rather than inferred from
completion status alone.
