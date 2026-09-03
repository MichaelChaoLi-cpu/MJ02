# Observed-flood confound test for the Cambodia–Thailand Gate B signal

Status: **2011 inundation does not explain the candidate light decline; causal promotion threshold still not met**  
Analysis date: **2026-08-22**  
Flood source: **Global Flood Database v1.4, events 3850 and 3853**

## 1. Question

Cambodia experienced severe flooding in 2011. The Gate B nighttime-light decline
could therefore be a flood effect spatially coincident with the conflict sites
rather than a conflict effect. This experiment asks three progressively stricter
questions:

1. Does conflict-distance exposure predict satellite-observed inundation?
2. Does the 2011 light coefficient survive direct adjustment for inundated area
   and satellite observation quality?
3. Does it survive when the sample is restricted to well-observed cells with no,
   or less than 10%, detected inundation?

## 2. Data construction and design

The flooded, duration, clear-view, and permanent-water bands for GFD events 3850
and 3853 were aggregated to all 179,072 cells of the stable Cambodia 1 km grid.
Flooded area excludes JRC permanent water. The maximum inundated share across
the overlapping events is primary so that a single flood episode is not counted
twice. Cells without a clear satellite observation remain missing rather than
being classified as dry.

Both LongNTL and CCNL are estimated over 2010–2013, with 2010 as the reference
year. Every specification retains grid-cell fixed effects, border-sector-by-year
fixed effects, the four frozen weather controls, and 10 km spatial-block clustered
covariance. Outcome-independent overlap weights are re-estimated after each flood
restriction.

The five specifications are:

- the original full sample;
- the subset with a clear GFD observation;
- that observed subset with flood share and clear-observation share interacted
  separately with 2011, 2012, and 2013;
- well-observed cells with no detected flood; and
- well-observed cells with less than 10% flooded area.

## 3. Results

### 2011 light estimates relative to 2010

| Scenario | Product | Full sample | Flood adjusted | Observed dry cells | Below 10% flooded |
|---|---|---:|---:|---:|---:|
| All 2011 event sectors | LongNTL | -0.641 [-1.272, -0.010] | -0.854 [-1.611, -0.097] | -0.545 [-1.148, 0.058] | -0.560 [-1.176, 0.056] |
| All 2011 event sectors | CCNL | -0.290 [-0.637, 0.058] | -0.513 [-1.003, -0.022] | -0.398 [-0.833, 0.036] | -0.410 [-0.851, 0.031] |
| Preah Vihear only | LongNTL | -0.595 [-1.091, -0.099] | -0.818 [-1.439, -0.197] | -0.600 [-1.120, -0.079] | -0.598 [-1.117, -0.080] |
| Preah Vihear only | CCNL | -0.294 [-0.687, 0.098] | -0.578 [-1.088, -0.069] | -0.527 [-1.010, -0.044] | -0.530 [-1.015, -0.045] |

Entries are standardized coefficients with 95% confidence intervals in brackets.
Magnitudes across restricted samples should not be read as a formal amplification
test because the support and overlap weights change. The relevant fact is that
the coefficient does not move toward zero when observed flooding is controlled
or excluded.

### Spatial overlap with inundation

In the Preah Vihear comparison, 3.14% of clearly observed treated-support cells
and 1.98% of clearly observed frontier-control cells registered any flood. Mean
maximum inundated shares were only 0.0062 and 0.0045, respectively. After
border-sector adjustment and overlap weighting, the conflict-distance dose did
not predict maximum inundated share (coefficient -0.0011, SE 0.0037, p=0.767) or
the probability of any detected flood (coefficient -0.0034, SE 0.0210, p=0.871).

The dry-cell Preah analysis retains 19,338 cells, including 4,800 treated-support
cells in 66 spatial blocks. Its maximum weighted covariate difference is 0.081
SD, below the frozen 0.10 balance threshold.

## 4. Interpretation

The test rejects a specific competing explanation: the Preah Vihear light decline
is not created by the mapped 2011 inundation footprint. The negative 2011 signal
appears in two independently processed DMSP-derived light series, remains after
direct flood adjustment, and remains among satellite-observed dry cells.

This is a useful strengthening, but it is not a causal promotion. The signal is
still concentrated in one conflict sector; only eleven support-passing spatial
placebos are available; CCNL is sparse and smaller in the unrestricted sample;
and the data do not directly observe displacement, road closure, damaged
settlements, or another unmeasured local 2011 shock. The appropriate claim is
therefore “not explained by observed inundation,” not “proved to be caused by
conflict.”

## 5. Reproducible artifacts

- Flood preprocessing: `src/preprocessing/preprocess_gfd_2011_national_grid.py`
- Flood-confound estimator: `src/analyses/test_gate_b_2011_flood_confound.py`
- Processed exposure: `data/processed/cambodia_national_2011_gfd_flood_exposure_preprocessed.parquet`
- Coefficients: `data/exp/experiment-design/cambodia-thailand-gate-b/gate_b_2011_flood_confound_coefficients.csv`
- Flood overlap diagnostics: `data/exp/experiment-design/cambodia-thailand-gate-b/gate_b_2011_flood_overlap_diagnostics.csv`
- Support diagnostics: `data/exp/experiment-design/cambodia-thailand-gate-b/gate_b_2011_flood_confound_support.csv`
- Figure: `data/exp/experiment-design/cambodia-thailand-gate-b/gate_b_2011_flood_confound.png`
