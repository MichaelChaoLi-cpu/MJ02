# Gate C NPP validation 2: decomposition across the two 2011 conflict sectors

Status: **both sector estimates are negative, but neither sector-only estimate is
individually precise at the 5% level. In the joint common-sample model, Preah Vihear is
negative and significant, Ta Moan/Ta Krabey is negative but imprecise, and the difference
between the two sector coefficients is not significant. The Ta Moan/Ta Krabey standalone
comparison fails the 0.10 balance benchmark and must be treated as diagnostic.**

Human protocol record: `MILI-D-20260822-019`

## Approved question

The experiment asks whether the pooled negative NPP drought-sensitivity estimate is a
directionally coherent feature of both 2011 conflict sectors or an artifact of combining
one negative sector with an opposing sector. It does not require two separately significant
sector estimates, because splitting two treatment locations necessarily reduces precision.

## Frozen implementation

- Outcome: annual land NPP anomaly in kg C/m2.
- Hazard: positive May–October rainfall-deficit intensity.
- Target: continuous conflict dose × post-2011 × drought intensity.
- Years: 2001–2007 versus 2012–2024; 2008–2011 remains excluded.
- Fixed effects: 1 km grid cell and climate-cell × year.
- Inference: debiased standard errors clustered by frozen 10 km spatial block.
- Sector-only samples: cells within 60 km of the target sector plus controls more than
  60 km from every 2008–2011 event. Cells exposed only to the other sector are excluded.
- Sector-only weights: outcome-independent overlap weights re-estimated within each sample.
- Joint model: both sectors' dose × post, dose × drought, and dose × post × drought terms
  enter simultaneously on the original pooled common sample and use the frozen pooled
  overlap weights.

## Results

| Specification | Target estimate | Clustered SE | 95% CI | p-value | Treated cells | Balance |
|---|---:|---:|---:|---:|---:|---:|
| Pooled two-sector baseline | -0.04313 | 0.02096 | [-0.08422, -0.00205] | 0.0396 | 10,605 | pooled frozen weights |
| Preah Vihear only | -0.04228 | 0.03232 | [-0.10563, 0.02108] | 0.1909 | 5,276 | max absolute SMD 0.055 |
| Ta Moan/Ta Krabey only | -0.08906 | 0.06891 | [-0.22413, 0.04600] | 0.1962 | 5,329 | max absolute SMD 0.227 |
| Joint model: Preah Vihear | -0.06170 | 0.02618 | [-0.11301, -0.01039] | 0.0184 | 5,276 | pooled common sample |
| Joint model: Ta Moan/Ta Krabey | -0.04396 | 0.04294 | [-0.12812, 0.04020] | 0.3060 | 5,329 | pooled common sample |
| Joint difference: Preah minus Ta Moan | -0.01774 | 0.05029 | [-0.11631, 0.08083] | 0.7242 | — | Wald contrast |

The sector-only models contain 31,775 and 31,828 grid cells respectively, observed over 20
included years. Preah Vihear retains 37.4 treated 10 km block-equivalents and achieves good
weighted balance. Ta Moan/Ta Krabey retains 21.0 treated block-equivalents, but its latitude
imbalance remains 0.227 SD after weighting. This exceeds the established 0.10 diagnostic
benchmark, so its sector-only coefficient cannot carry a standalone causal interpretation.

## What the experiment establishes

The pooled result is not created by averaging a negative coefficient in one sector with a
positive coefficient in the other: every opened sector-specific target estimate is
negative. In the joint common-sample model, the two point estimates are also close enough
that their difference is highly compatible with zero (p = 0.724). Thus, there is no evidence
that the drought-sensitivity change has opposite signs across the two conflict sectors or
that the pooled coefficient is statistically different across sectors.

The experiment does **not** establish two independent replications. Once the sample is split,
both sector-only confidence intervals include zero. The joint model identifies the Preah
Vihear coefficient more precisely, but the nonsignificant sector difference means the data
do not support claiming that the effect exists only in Preah Vihear. Ta Moan/Ta Krabey also
has weak standalone common support. The defensible reading is therefore directional
coherence with limited sector-level precision, not two separately proven local effects.

## Consequence for the evidence chain

This diagnostic modestly strengthens the pooled finding because it rules out an obvious
sign-reversal failure and finds no detectable cross-sector coefficient difference. It does
not by itself complete causal validation. The remaining sequential diagnostics are the
declared distance-ring gradient, covariance sensitivity at 5, 10, and 20 km, and temporal
decomposition into early, medium, and late post-conflict periods.

## Reproducible artifacts

- Script: `src/analyses/test_cambodia_thailand_gate_c_npp_sector_decomposition.py`
- Tidy coefficients: `data/exp/experiments/cambodia-thailand-gate-c-npp/sector-decomposition/gate_c_npp_sector_decomposition_tidy.csv`
- One-sheet review workbook: `data/exp/experiments/cambodia-thailand-gate-c-npp/sector-decomposition/gate_c_npp_sector_decomposition_summary.xlsx`
- Protocol and support metadata: `data/exp/experiments/cambodia-thailand-gate-c-npp/sector-decomposition/gate_c_npp_sector_decomposition_metadata.json`
