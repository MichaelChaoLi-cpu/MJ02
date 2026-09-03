# Human review register: Cambodia–Thailand Gate C design

Status: **PARTIALLY RESOLVED — NPP-only scope and the first main model were approved and estimated; remaining inference and promotion criteria still require review.**

## 1. What the user has actually approved

The explicit human decisions currently establish the following scientific direction:

1. replace historical Khmer Rouge exposure with the dated and geocoded Cambodia–Thailand
   conflict;
2. ask whether conflict exposure amplified later natural-hazard impacts, rather than making
   the direct 2011 activity decline the final claim;
3. organize the national data and examine drought first, extreme rainfall second;
4. retain economic-activity and ecological-productivity outcome domains; and
5. use a prospective gate before opening the Gate C target coefficient.
6. use natural-unit NPP as the primary Gate C outcome, standardized NPP as a scale check,
   and remove LongNTL from the Gate C analytical target; and
7. estimate the first NPP model with grid and climate-cell × year fixed effects, the
   2001–2007 versus 2012–2024 comparison, and the continuous-dose × post × drought target.

These decisions do **not** imply human approval of every numerical threshold or operational
model choice embedded by the analyst.

## 2. Effect-size and power crosswalk

The Gate C interaction is interpreted as the post-conflict change in the drought-response
slope at the event-location-versus-zero-conflict-dose contrast. One unit of dry-rainfall
intensity corresponds to a one-standard-deviation seasonal precipitation shortfall when
rainfall is below its climate normal; non-dry years have zero dry intensity.

The table uses the currently simulated rho = 0.5 error scenario. It is a review aid, not an
approved decision rule.

| Candidate absolute effect | LongNTL-scale change | LongNTL power | NPP anomaly-Z change | NPP power |
|---:|---:|---:|---:|---:|
| 0.10 outcome SD | 0.00499 asinh units | 12.0% | 0.0869 | 13.5% |
| 0.15 outcome SD | 0.00748 asinh units | 21.2% | 0.1304 | 24.4% |
| 0.20 outcome SD | 0.00997 asinh units | 33.8% | 0.1739 | 39.3% |
| 0.25 outcome SD | 0.01247 asinh units | 48.7% | 0.2174 | 55.9% |
| 0.30 outcome SD | 0.01496 asinh units | 63.8% | 0.2608 | 71.6% |
| 0.35 outcome SD | 0.01746 asinh units | 77.0% | 0.3043 | 84.0% |
| 0.40 outcome SD | 0.01995 asinh units | 86.9% | 0.3478 | 92.1% |

The LongNTL translation requires special caution. During 2000–2007, only 0.55% of frontier
grid-cell-years have positive LongNTL; 99.45% are zero. Consequently, an asinh-unit change
cannot be converted into one uniform percentage decline and the full-sample SD is driven by
a very small lit subset. NPP is easier to interpret because its source variable is already
an anomaly Z score, although its empirical pre-conflict SD in this sample is 0.869 rather
than exactly one.

## 3. Consequential choices requiring review

| Design choice | Current operational value | Provenance | Why it matters | Analyst recommendation for review |
|---|---|---|---|---|
| Smallest effect of substantive interest | 0.20 outcome SD | Analyst convention; not separately approved | Determines whether the design is called adequately powered | Do not approve in SD alone; first decide a meaningful NPP loss and a defensible LongNTL estimand |
| Required power | 80% | Conventional analyst default | Determines the allowed false-negative risk | Retain 80% as a planning convention unless there is a substantive reason to accept greater risk |
| Test size | Two-sided 5% | Conventional analyst default | Controls false positives and affects MDE | Retain for the main test; do not switch to one-sided merely to gain power |
| Primary economic outcome | Continuous 1 km LongNTL | Outcome domain approved, exact form not separately reviewed | 99.45% pre-period zeros make the continuous grid outcome hard to interpret | Reconsider as primary; test aggregation to settlements/communes or a predeclared hurdle outcome |
| Primary ecological outcome | Annual NPP anomaly Z | Outcome domain approved | NPP measures ecological production, not crop yield or welfare | Retain as primary ecological outcome with restrained claims |
| Primary drought measure | Positive hinge of May–October rainfall anomaly Z | Drought-first approved; exact formula selected by analyst | Captures meteorological rainfall deficit but not soil moisture, SPEI, or observed crop drought | Review against dry-spell duration; consider a two-measure confirmation rule |
| Conflict year | 2011 | UCDP activation and verified chronology | Defines treatment timing | Retain 2011, with 2008–2010 explicitly treated as escalation rather than clean pre-period |
| Pre/post windows | 2000–2007 versus 2012–2024 | Analyst-selected | Controls identifying variation and excludes ambiguous years | Retain as primary; review shorter balanced windows as declared sensitivity only |
| Spatial sample | Cells within 60 km of the Cambodia–Thailand frontier | Gate A support choice | Wider support improves comparison but may mix unrelated locations | Retain as a support domain, not as a claim that conflict reached 60 km |
| Conflict dose | Linear decline from one at an event to zero at 60 km | Analyst-selected approximation | Imposes the spatial shape of treatment | Compare prospectively with fixed rings and a monotone nonlinear dose; do not select by significance |
| Spatial covariance unit | 10 km blocks | Analyst-selected screening scale | Drives effective sample size and uncertainty | Estimate empirical spatial correlation and review 5, 10, and 20 km blocks before freezing inference |
| Power error persistence | AR(1) rho = 0.5 primary; 0 and 0.8 sensitivity | Analyst-selected simulation values | Changes simulated power materially | Replace arbitrary primary rho with an estimate/range from pre-conflict residuals |
| Fixed effects in power audit | 10 km block and border-sector × year | Analyst-selected approximation | Determines which variation identifies the interaction | Align the power design exactly with the eventual regression before using it as a stop rule |
| Analysis weights | Frozen Gate B overlap weights | Inherited analyst design | Balance for direct effects may not optimize hazard-slope comparisons | Re-audit overlap for Gate C rather than automatically inheriting Gate B weights |
| Co-hazard controls | Extreme rainfall and heat | Analyst-selected | May reduce omitted-variable bias but increase collinearity | Declare a minimal primary model and a co-hazard-adjusted robustness model |
| Hazard common support | Exposed pre-period Q01–Q99 | Analyst-selected | Removes extrapolation but the 1% cutoffs are arbitrary | Show Q02.5–Q97.5 and Q05–Q95 sensitivity without selecting a favorable range |
| Direct-effect prerequisite | Gate B must be credible before Gate C promotion | Analyst-proposed sequential logic, broadly accepted through workflow | Conflict may alter vulnerability even if annual lights do not show a clean direct level loss | Reconsider whether a direct satellite level effect is logically necessary or only supporting evidence |
| Cross-sector requirement | Evidence should not depend only on Preah Vihear | Analyst promotion rule | Protects against a single-site story but may be unrealistic with heterogeneous conflicts | Retain the leave-one-sector diagnostic; define in advance what heterogeneity would be acceptable |
| Spatial placebo rule | Eligible unaffected border placements | Analyst inference design | Only 11 support-passing placebos limit attainable p-values | Review whether the placebo universe is substantively exchangeable; do not use cell p-values as replacement |
| Multiple outcomes and hazards | LongNTL/NPP × drought/extreme rain, with additional secondary hazards | Scope partly approved, correction rule absent | Creates multiple-testing discretion | Freeze one primary outcome–hazard pair or an explicit hierarchical testing sequence |
| Stop before target inspection | Yes | Explicit workflow approval | Prevents thresholds from being changed in response to the result | Retain until this review is complete |

## 4. Data facts that are not human choices

- The annual Gate C panel has 927,600 cell-years and is unique by grid cell and year.
- LongNTL covers 2000–2024; NPP is missing in 2000 and covers 2001–2024.
- The main pre/post sample has 779,184 cell-years.
- Drought has usable pre- and post-conflict temporal support.
- Treatment variation originates from two conflict sectors, and the direct LongNTL signal is
  concentrated in one sector.
- The current triple-interaction regressor retains 19.8% of its variance after lower-order
  terms and fixed effects are removed (VIF 5.05).

## 5. Recommended order of human decisions

1. Decide whether NPP, LongNTL, or a revised economic-activity outcome carries the primary
   scientific claim.
2. Define the smallest substantively important change in that outcome's native units.
3. Confirm the drought definition and whether one or two drought indicators are required.
4. Confirm conflict-dose shape, time windows, and spatial support.
5. Confirm the exact regression, inference unit, alpha, power target, and multiple-testing
   sequence.
6. Rerun the prospective power audit using only the approved choices.
7. Open the Gate C coefficient only if the approved audit passes.

The machine-readable effect crosswalk and outcome-scale audit are stored with the Gate C
experimental artifacts.
