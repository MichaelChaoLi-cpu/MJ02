# Gate C NPP validation 1: spatial-placebo inference

Status: **SUPERSEDED — the human researcher judged 11 placements inadequate for formal inference. This file is retained only as an audit artifact and is excluded from the active evidence chain and manuscript-facing results. The accepted spatial test uses 1,000 random two-sector assignments.**

Human sequencing record: `MILI-D-20260822-015`

## Frozen test

The two-sector conflict template was translated to the 11 eligible unaffected locations in
the frozen Gate B placebo universe. Every placement preserves the actual distance between
the two conflict sectors. Cells within 60 km of every actual 2008–2011 event were excluded
from placebo estimation. For each placebo placement, the experiment independently rebuilt:

1. distance to the two placebo sectors;
2. the continuous linear conflict dose;
3. outcome-independent binary overlap weights; and
4. the approved natural-unit NPP × dry-rainfall Gate C regression with grid and
   climate-cell × year fixed effects.

The placebo target is exactly the conflict-dose × post-conflict × drought interaction used
for the actual conflict placement. The frozen support rule requires maximum absolute
weighted SMD no greater than 0.10 and treated 10 km block ESS of at least 10.

## Result

- Actual conflict-placement estimate: -0.04313 kg C/m2.
- Eligible placebo placements: 11.
- Support-passing placebo placements: 11.
- Placebo estimates at least as negative as actual: 2.
- One-sided negative-tail rank p-value with add-one correction: 0.25.
- Placebo estimates at least as large in absolute value as actual: 5.
- Two-sided absolute rank p-value with add-one correction: 0.50.

The two more-negative placebo estimates are P010 (-0.05275) and P011 (-0.04634). P009 is
also close to the actual result at -0.04215. Therefore the actual placement is third-most
negative among the 12 actual-or-placebo assignments, rather than an exceptional frontier
location.

With only 11 placebos, the smallest attainable add-one one-sided p-value is 1/12 = 0.0833.
Consequently this frozen placebo universe could never establish a 5% randomization p-value,
even if the actual estimate were most negative. This design limitation was not made
sufficiently explicit in the earlier promotion language. It does not change the present
conclusion because two placebo estimates are more negative than the actual estimate and the
observed rank p-value is 0.25.

## Interpretation

The ordinary 10 km block-clustered main-model p-value of 0.0396 does not survive the more
credible spatial assignment comparison. The negative coefficient may reflect a broader
west-to-east or border-sector-specific change in NPP drought sensitivity rather than the
causal effect of the 2011 conflict sites.

This test does not prove that the conflict had no ecological effect. It shows that the
actual conflict geography is not sufficiently unusual relative to other balanced frontier
placements to support the causal interpretation by itself. The result must remain
"model-based association, spatially unvalidated."

After human review, the spatial reference distribution was redesigned rather than proceeding
immediately to the sector diagnostic. The redesigned result is reported in
`docs/proposals/cambodia_thailand_gate_c_npp_random_pair_monte_carlo.md`. The next unopened
experiment remains the separate-sector and leave-one-sector diagnostic.

## Reproducible artifacts

- Script: `src/analyses/test_cambodia_thailand_gate_c_npp_placebo_sites.py`
- Placement estimates: `data/exp/experiments/cambodia-thailand-gate-c-npp/spatial-placebo/gate_c_npp_spatial_placebo_estimates.csv`
- Inference summary: `data/exp/experiments/cambodia-thailand-gate-c-npp/spatial-placebo/gate_c_npp_spatial_placebo_inference.csv`
- Review workbook: `data/exp/experiments/cambodia-thailand-gate-c-npp/spatial-placebo/gate_c_npp_spatial_placebo_summary.xlsx`
- Distribution figure: `data/exp/experiments/cambodia-thailand-gate-c-npp/spatial-placebo/gate_c_npp_spatial_placebo_distribution.png`
