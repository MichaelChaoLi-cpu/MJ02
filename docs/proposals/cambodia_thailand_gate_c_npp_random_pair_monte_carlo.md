# Gate C NPP validation 1B: 1,000-draw random two-sector Monte Carlo

Status: **passes the human-approved directional spatial rule (one-sided p = 0.0340); equal-tail two-sided p = 0.0679. This is a post-result spatial validation, not a preregistered causal test.**

Human protocol records: `MILI-D-20260822-016` and `MILI-D-20260822-017`

## Frozen redesigned protocol

Before any simulated NPP coefficient was estimated, the analysis generated and froze 1,000
accepted random two-sector assignments using seed 2011. The rules were:

- draw both sector anchors uniformly by distance along the Cambodia–Thailand border;
- require at least 30 km between anchors and impose no maximum separation;
- require each anchor to be at least 120 km along the border from both actual sectors;
- construct a linear conflict dose that reaches zero at 60 km;
- independently recompute outcome-free overlap weights for every assignment;
- accept maximum absolute weighted SMD no greater than 0.10; and
- accept treated 10 km block ESS of at least 10.

The algorithm needed 1,882 candidate draws to freeze 1,000 accepted assignments. The
freeze metadata records that no NPP outcome was loaded during assignment screening. Because
the 120 km exclusion zones around the two actual sectors overlap, the eligible anchor domain
is the approximately 0–346 km southwestern portion of the 774 km shared border.

After freezing, every accepted assignment was estimated with the same natural-unit NPP,
2001–2007 versus 2012–2024 periods, grid fixed effects, climate-cell × year fixed effects,
and 10 km block-clustered model as the actual conflict placement.

## Main Monte Carlo result

- Actual target estimate: -0.043132 kg C/m2.
- Accepted simulations: 1,000.
- Simulated estimates at least as negative as actual: 33.
- Approved add-one one-sided p-value: (1 + 33) / (1 + 1,000) = 0.033966.
- Monte Carlo standard error of the one-sided p-value: 0.005728.
- Binomial 95% interval for the underlying lower-tail frequency: approximately 0.0228–0.0460.
- Equal-tail two-sided rank p-value: 0.067932.
- Zero-centered absolute-value diagnostic: 0.444555; this is not the appropriate primary
  two-sided rank test because the simulated distribution is not centered at zero.

The actual estimate lies at approximately the 3.3rd percentile of the random-pair
distribution. It therefore passes the human-approved directional 5% spatial rule.

## Reference-distribution diagnostics

The simulated target coefficients have mean 0.03093, median 0.03346, and SD 0.03914 kg
C/m2. Their 5th percentile is -0.03778, which is less negative than the actual estimate.

Simulation geometry varies as approved:

- sector separation ranges from 30.0 to 343.7 km, with median 161.1 km;
- the actual 208.1 km separation is at the 70.9th percentile;
- simulated treated cells range from 6,502 to 13,005, with median 10,096.5;
- the actual 10,605 treated cells are at the 66.4th percentile.

The simulated coefficient correlates -0.376 with sector separation, -0.351 with treated
cell count, and 0.236 with mean exposed dose. The actual geometry is not at the edge of
these distributions, but the correlations show why varying sector geometry changes the
reference distribution. These diagnostics must accompany the p-value.

## Interpretation

Under the redesigned directional rule, the actual conflict geography produces a more
negative post-conflict NPP drought-slope change than approximately 96.6% of admissible
random two-sector assignments. This restores spatial support that was unresolved in the
coarse 11-placement test.

It does not turn the conflict placement into a random treatment. The 1,000-draw design and
one-sided 5% rule were adopted after the actual main coefficient was known. Accordingly,
this is a transparent post-result validation. It supports spatial unusualness under the
newly frozen random-pair universe, while the broader causal claim still requires the
unopened sector, distance-gradient, covariance-scale, and temporal-dynamic diagnostics.

By human decision `MILI-D-20260822-018`, this is the sole current spatial-validation result.
The earlier 11-placement exercise is methodologically superseded, excluded from the active
evidence chain, and retained only as an audit artifact.

## Reproducible artifacts

- Script: `src/analyses/run_cambodia_thailand_gate_c_npp_random_pair_monte_carlo.py`
- Frozen assignments: `data/exp/experiments/cambodia-thailand-gate-c-npp/random-pair-monte-carlo/gate_c_npp_random_pair_accepted_assignments.csv`
- Assignment-freeze metadata: `data/exp/experiments/cambodia-thailand-gate-c-npp/random-pair-monte-carlo/gate_c_npp_random_pair_freeze_metadata.json`
- Complete estimates: `data/exp/experiments/cambodia-thailand-gate-c-npp/random-pair-monte-carlo/gate_c_npp_random_pair_estimates.csv`
- Inference summary: `data/exp/experiments/cambodia-thailand-gate-c-npp/random-pair-monte-carlo/gate_c_npp_random_pair_inference.csv`
- Review workbook: `data/exp/experiments/cambodia-thailand-gate-c-npp/random-pair-monte-carlo/gate_c_npp_random_pair_summary.xlsx`
- Distribution figure: `data/exp/experiments/cambodia-thailand-gate-c-npp/random-pair-monte-carlo/gate_c_npp_random_pair_distribution.png`
