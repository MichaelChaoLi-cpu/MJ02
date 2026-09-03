# Gate C prospective power decision: conflict exposure and drought sensitivity

Status: **PAUSED FOR HUMAN REVIEW — the provisional analyst-selected 0.20 SD / 80% screening rule is not met. Those thresholds were not separately approved by the human researcher, and post-conflict outcome coefficients remain unopened.**

## Question tested by this gate

The proposed Gate C experiment asks whether exposure to the 2011 Cambodia–Thailand
border conflict changed the marginal effect of later drought on local outcomes. The target
coefficient is the interaction between continuous conflict dose, the post-2011 period, and
drought intensity. A negative coefficient would mean that the same drought shock was more
damaging after the conflict in more-exposed locations. This is distinct from testing whether
exposed locations are poorer on average.

## Frozen analysis support

- Outcomes: annual LongNTL and annual MODIS NPP anomaly.
- Primary hazard: May–October dry-rainfall intensity.
- Pre-conflict period: 2000–2007.
- Excluded escalation/conflict period: 2008–2011.
- Post-conflict period: 2012–2024.
- Exposure: continuous 2011 conflict dose, equal to one at an event location and declining
  linearly to zero at 60 km.
- Estimation level for the power audit: 10 km spatial-block × border-sector × year.
- Fixed effects: spatial block and border-sector × year.
- Lower-order terms: drought, dose × post, post × drought, and dose × drought.
- Co-hazard controls: extreme-rainfall intensity and heat intensity.
- Provisional analyst-selected smallest effect: 0.20 pre-conflict outcome SD, pending human review.
- Provisional analyst-selected required power: 80% at a two-sided 5% test, pending human review.

The analysis-ready source panel contains 927,600 grid-cell-years (37,104 cells over 25
years). The primary pre/post sample contains 779,184 grid-cell-years. Aggregation produces
10,080 block-sector-year observations over 455 spatial blocks and 21 included years.

## Prospective result

Outcome variance was calibrated only from the pre-conflict period. No post-conflict outcome
or target-interaction coefficient was inspected. Under the primary AR(1) calibration of
rho = 0.5:

| Outcome | Null SE (outcome SD) | Power at 0.20 SD | Effect needed for 80% power |
|---|---:|---:|---:|
| LongNTL | 0.130 | 33.8% | 0.363 SD |
| Annual NPP | 0.119 | 39.3% | 0.332 SD |

Neither outcome reaches the frozen 80% threshold. Across the full rho = 0.0, 0.5, and 0.8
simulation set, power at 0.20 SD ranges from 27.9% to 61.4%; no scenario passes. Positive
serial correlation raises power in this particular within-block change design because
persistent errors partly cancel under temporal differencing; the rho scenarios should
therefore be read as sensitivity cases, not as an ordered conservative-to-optimistic scale.

The target interaction is estimable but weakly separated from its lower-order components.
After fixed-effect absorption and nuisance-term residualization, only 19.8% of its variance
remains, corresponding to a variance-inflation factor of 5.05.

## Interpretation

The limitation is not the number of raster rows or the length of the satellite panel. It is
the amount of independent treatment variation. Conflict dose originates from two localized
border sectors, while the validated direct LongNTL signal is concentrated in Preah Vihear.
Repeated years and thousands of nearby pixels do not create additional independent conflict
assignments. The block-year simulation is already more favorable than inference based on
the small number of conflict sectors or eligible placebo sectors, so failure here is a firm
stop signal.

Opening the post-conflict regression now could produce a coefficient and a conventional
p-value, but the design could not reliably distinguish the agreed 0.20 SD amplification
effect from noise. A non-significant estimate would not establish resilience or no effect;
an isolated significant estimate would be vulnerable to spatial pseudo-replication.

## Decision and next admissible paths

Per the approved sequential protocol, the current experiment pauses before estimating the
actual Gate C target coefficient. The numerical screening rule itself is now explicitly
under human review; see `docs/proposals/cambodia_thailand_gate_c_human_review.md`. The
direction is not promoted into `docs/AnaSOP.md` and should not be presented as a Nature
Communications-ready causal result with current data.

Credible ways to reopen Gate C are:

1. add independent, dated conflict episodes or comparable cross-border conflict settings
   and rerun a multi-event power audit;
2. obtain an independent treatment measure with meaningful within-sector variation, such
   as verified settlement damage, displacement, or access disruption;
3. propose and obtain human approval for a larger smallest effect of substantive interest
   before inspecting the target coefficient, recognizing that this narrows the scientific
   claim to large amplification effects only; or
4. develop a higher-frequency vegetation design and pass a new power audit that explicitly
   accounts for temporal and spatial dependence.

Simply reducing fixed effects, clustering at pixels, trying many bandwidths, or selecting
only the sector with the strongest Gate B result would not solve the identification problem.

## Reproducible artifacts

- Analysis-ready panel: `data/processed/cambodia_thailand_gate_c_annual_panel_preprocessed.parquet`
- Panel builder: `src/preprocessing/preprocess_cambodia_thailand_gate_c_panel.py`
- Feasibility audit: `src/analyses/audit_cambodia_thailand_gate_c_feasibility.py`
- Power audit: `src/analyses/audit_gate_c_drought_power.py`
- Power results: `data/exp/experiment-design/cambodia-thailand-gate-c/gate_c_drought_prospective_power.csv`
- Design-rank results: `data/exp/experiment-design/cambodia-thailand-gate-c/gate_c_drought_design_rank.csv`
- Power figure: `data/exp/experiment-design/cambodia-thailand-gate-c/gate_c_drought_power_curve.png`
