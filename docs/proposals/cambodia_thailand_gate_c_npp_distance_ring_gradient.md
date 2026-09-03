# Gate C NPP validation 3: fixed distance-ring gradient

Status: **fails the predeclared negative distance-decay pattern. All four fixed-ring
coefficients are small, positive, and individually imprecise; they are jointly compatible
with zero (Wald p = 0.7249). The inner 0–20 km and outer 20–60 km coefficients are nearly
identical (difference 0.00015 kg C/m2, p = 0.9807).**

Human protocol record: `MILI-D-20260822-021`

## Approved question

The experiment relaxes the main model's assumption that post-conflict drought sensitivity
changes linearly with the constructed conflict dose. It estimates separate coefficients for
cells 0–10, 10–20, 20–40, and 40–60 km from a verified 2011 conflict coordinate, relative
to frontier controls more than 60 km from every 2008–2011 candidate event. Every ring is
reported; no ring, bandwidth, or sign was selected after viewing the outcome.

## Frozen implementation

- Outcome: annual land NPP anomaly in kg C/m2.
- Hazard: positive May–October rainfall-deficit intensity.
- Target: distance-ring exposure × post-2011 × drought intensity.
- Years: 2001–2007 versus 2012–2024; 2008–2011 remains excluded.
- Fixed effects: 1 km grid cell and climate-cell × year.
- Weights: frozen Gate B binary-support overlap weights used by the opened continuous-dose
  model.
- Inference: debiased standard errors clustered by the frozen 10 km spatial block.
- Additional fixed comparison: inner 0–20 km versus outer 20–60 km.

The script re-estimates the continuous-dose model on the same sample as a QA check and
exactly reproduces the opened coefficient of -0.043132 kg C/m2.

## Results

| Distance specification | Target estimate | Clustered SE | 95% CI | p-value |
|---|---:|---:|---:|---:|
| 0–10 km | 0.00564 | 0.01014 | [-0.01423, 0.02551] | 0.5780 |
| 10–20 km | 0.00797 | 0.00958 | [-0.01080, 0.02674] | 0.4053 |
| 20–40 km | 0.00782 | 0.00722 | [-0.00633, 0.02196] | 0.2788 |
| 40–60 km | 0.00305 | 0.00296 | [-0.00276, 0.00885] | 0.3041 |
| Inner 0–20 km | 0.00320 | 0.00696 | [-0.01043, 0.01683] | 0.6457 |
| Outer 20–60 km | 0.00305 | 0.00296 | [-0.00276, 0.00885] | 0.3041 |
| Inner minus outer | 0.00015 | 0.00629 | [-0.01218, 0.01248] | 0.9807 |

The four fixed-ring target coefficients have a joint Wald statistic of 2.059 and p-value
of 0.7249. None is negative, and their point estimates do not become less negative toward
60 km because there is no negative ring coefficient to attenuate. The adjacent-ring
contrasts are also individually compatible with zero.

## Support limitation

The frozen weights achieve balance for all cells within 60 km collectively versus frontier
controls; they were not designed to make every narrow ring an independently balanced
treatment. The ring-specific 10 km block effective sample sizes are 4.1, 7.8, 19.8, and
51.7 from the nearest to the outermost ring. Consequently, especially the two innermost
coefficients are functional-form diagnostics rather than separately powered causal effects.

This limitation explains reduced ring-level precision, but it does not create the expected
negative ordering: all four point estimates are on the opposite side of zero, while the
predeclared inner-minus-outer contrast is close to exactly zero.

## Interpretation

The fixed-ring result does not support the substantive claim that places closer to the
2011 conflict sites developed greater post-conflict NPP sensitivity to drought. Because the
same script exactly reproduces the negative continuous-dose coefficient before replacing
the dose with ring indicators, the discrepancy is a functional-form problem rather than a
sample or coding mismatch. The continuous model extracts a negative linear slope from
within the 0–60 km support, but that slope does not translate into negative level contrasts
for predetermined distance bands.

The accepted 1,000-draw Monte Carlo and the sector decomposition both use the same continuous
dose. They show that the continuous coefficient is spatially unusual and not sign-reversed
across the two sectors, but they do not validate its assumed linear distance form. Therefore,
the combined evidence no longer supports promoting the continuous estimate as a robust
conflict-induced drought-amplification effect. It remains a model-dependent finding requiring
explicit explanation or a prospectively justified alternative exposure function.

## Consequence for the evidence chain

This is a substantive failure of the planned spatial-mechanism diagnostic. Covariance-scale
and temporal-dynamic checks may still determine whether the continuous association is
statistically and temporally stable, but they cannot by themselves establish the missing
distance-decay pattern. No alternative ring boundaries or weights should be searched after
seeing this result.

## Reproducible artifacts

- Script: `src/analyses/test_cambodia_thailand_gate_c_npp_distance_ring_gradient.py`
- Tidy coefficients and contrasts: `data/exp/experiments/cambodia-thailand-gate-c-npp/distance-ring-gradient/gate_c_npp_distance_ring_gradient_tidy.csv`
- One-sheet review workbook: `data/exp/experiments/cambodia-thailand-gate-c-npp/distance-ring-gradient/gate_c_npp_distance_ring_gradient_summary.xlsx`
- Protocol, support warning, and QA metadata: `data/exp/experiments/cambodia-thailand-gate-c-npp/distance-ring-gradient/gate_c_npp_distance_ring_gradient_metadata.json`
