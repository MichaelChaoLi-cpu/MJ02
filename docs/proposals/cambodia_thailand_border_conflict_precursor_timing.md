# Precursor-timing test for the Cambodia–Thailand Gate B signal

Status: **timing evidence strengthened; causal promotion threshold still not met**  
Analysis date: **2026-08-22**  
Outcome: **annual asinh LongNTL, 2000–2014**

## 1. Question

The Preah Vihear sector had verified Cambodia–Thailand clashes in 2008, 2009,
and 2011. The 2008 and 2009 events each recorded two UCDP best deaths, while
the 2011 sector events recorded eleven. This provides a within-location timing
diagnostic: did the nighttime-light gap already appear during the two small
precursor clashes, or was it concentrated in the more intense 2011 episode?

## 2. Design

The test reuses the frozen outcome-independent Gate B samples and overlap
weights. It estimates a separate conflict-distance-dose interaction for every
year from 2000 through 2014, relative to the pooled 2005–2007 reference period.
The model includes national 1 km grid-cell fixed effects, one-degree border-
sector-by-year fixed effects, the four frozen weather controls, and 10 km
spatial-block clustered covariance.

Two samples are shown: all 2011 event sectors and Preah Vihear alone. The latter
is the informative comparison because the 2008 and 2009 events occurred in that
same sector.

## 3. Results

### Preah Vihear annual coefficients

| Year | UCDP best deaths in sector | Estimate (pre-period SD) | 95% CI | p-value |
|---:|---:|---:|---:|---:|
| 2008 | 2 | -0.000 | [-0.021, 0.020] | 0.986 |
| 2009 | 2 | 0.003 | [-0.011, 0.017] | 0.669 |
| 2010 | 0 recorded lethal clash | -0.068 | [-0.145, 0.008] | 0.079 |
| 2011 | 11 | -0.570 | [-1.000, -0.139] | 0.009 |
| 2012 | 0 | -0.207 | [-0.428, 0.014] | 0.066 |
| 2013 | 0 | -0.026 | [-0.071, 0.020] | 0.267 |

The 2008 and 2009 coefficients are jointly compatible with zero (p=0.819). The
2011 coefficient differs from their mean by -0.570 SD (p=0.0096). The pooled
all-sector contrast gives the same ordering, with a 2011-versus-precursor
difference of -0.649 SD (p=0.034).

The full annual path is flat through 2009, turns modestly negative in 2010,
reaches its minimum in 2011, attenuates in 2012, and is close to zero by 2013.
All years through 2012 belong to the same reconstructed LongNTL source stage,
so the 2011 minimum is not the 2012/2013 reconstructed-to-observed transition.

## 4. Interpretation

This experiment strengthens temporal specificity. The candidate decline is not
a generic feature of every year with a border incident at Preah Vihear: the two
small 2008–2009 clashes leave no detectable annual light response, whereas the
larger 2011 fighting coincides with a sharp, temporary decline.

It does not prove an intensity dose-response. There are only three clash years,
annual lights may not detect short low-intensity events, and 2010 already shows
a modest negative estimate despite no recorded lethal UCDP event. That 2010
movement could reflect pre-conflict tension, noise, or another local process.
The defensible conclusion is therefore that the 2011 signal has stronger timing
alignment than the precursor years—not that UCDP deaths causally determine its
magnitude.

## 5. Reproducible artifacts

- Estimator: `src/analyses/test_gate_b_precursor_timing.py`
- Annual coefficients: `data/exp/experiment-design/cambodia-thailand-gate-b/gate_b_precursor_timing_coefficients.csv`
- Timing contrasts: `data/exp/experiment-design/cambodia-thailand-gate-b/gate_b_precursor_timing_tests.csv`
- Figure: `data/exp/experiment-design/cambodia-thailand-gate-b/gate_b_precursor_timing.png`

