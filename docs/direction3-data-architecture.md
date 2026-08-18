# Direction 3 Data Architecture

## Objective

Organize an analysis-ready system for studying whether persistent mine/ERW contamination
amplifies the effects of rainfall shocks on agriculture, household welfare, food security, and
education in Cambodia.

This is a repeated cross-sectional design. The main identifiers are survey year, PSU, household,
person, province, district, commune, and village. Mine exposure is measured at village level;
climate shocks are initially measured at commune-month and commune-year levels.

## Planned Data Layers

| layer | unit | current status | intended role |
|---|---|---|---|
| Mine baseline records | contamination point | processed | Preserve source-level exposure evidence and categories |
| Mine exposure | village | processed | Main explanatory variable and heterogeneity measures |
| CSES geography | PSU-village-year | processed; exact linkage for 2007-2021 | Connect households to village exposure and commune climate |
| Climate | commune/district/province-month | processed | Survey-aligned rainfall history and explicit spatial fallbacks |
| Climate | commune/district/province-year | processed | Annual and May-October rainfall anomalies at recorded resolution |
| CPI | component-month and component-year | processed | Convert monetary outcomes to 2021 constant-price riels |
| CSES source modules | original module grain | 27 harmonized modules processed | Preserve outcomes, controls, weights, and mechanisms before aggregation |
| PSU analysis skeleton | PSU-survey wave | processed | Exposure, climate, and module-coverage entry point |
| Household analysis spine | household-survey wave | processed | Household linkage entry point for outcome construction |
| Household core outcomes | household-survey wave | processed | Comparable agriculture, food, food-security, and household context variables |
| Education core outcomes | person-survey wave | processed | Comparable attendance, attainment, expenditure, weights, and household context |
| Final estimation files | household/person-survey wave | pending sample rules | Apply estimation samples and model-ready transformations |

## Current Sample Recommendation

- Main linked sample: CSES 2007-2021.
- Preserve 2004 during harmonization, but exclude it from the initial mine-linked analysis
  because an administrative village-code crosswalk has not been located.
- Use the full Cambodia sample for linkage diagnostics, then define the estimation comparison
  set within mine-affected regions after common-support checks.
- Interpret unmatched villages as having **no recorded point in the public baseline**, not as
  verified mine-free villages.

## Candidate Mine Exposure Variables

- Indicator for any recorded mine/ERW baseline point in the village.
- Number of contamination records and its log transformation.
- Counts by fear level: high, medium, low, none/missing.
- Counts by proximity: very near, near, far, very far, missing.
- Counts by reported land-class code.
- First and last survey dates, operator count, and contamination-point mean coordinates.

The source contains 16,224 points across 2,278 villages. It does not contain contaminated-area
polygons, so record count must not be described as contaminated area or physical mine density.

## Candidate Climate Variables

- Monthly rainfall in millimetres.
- Commune-by-calendar-month standardized rainfall anomaly.
- Annual rainfall and standardized annual anomaly.
- May-October rainfall and standardized May-October anomaly.
- Bottom-decile dry-shock and top-decile wet-shock indicators.
- Grid-cell count and extraction method as climate measurement-quality fields.

CHIRPS v2 covers every month from January 2004 through December 2021. Of 1,633 commune
polygons, 308 are smaller than or poorly aligned with the 0.05-degree grid and currently use the
grid cell nearest to an interior representative point. This measurement distinction will be
retained for robustness checks.

The strict historical commune-code link is preserved as a separate diagnostic. The enhanced
link first accepts exact codes, unique hierarchical name matches, or high-confidence fuzzy
commune matches (minimum similarity 0.85 and minimum winning margin 0.08). Remaining locations
receive explicitly flagged district or province rainfall rather than silent imputation. At the
PSU-wave level, 5,574 of 5,617 links remain at commune resolution, 39 use district rainfall, and
4 use province rainfall. All 5,617 have observed enhanced climate values. At the household-wave
level, this corresponds to 62,444 commune, 436 district, and 40 province observations in the
62,920-household main sample.

## Candidate Outcome Priority

1. Primary outcomes: cultivated land, crop production/sales, agricultural inputs, and food
   consumption or food insecurity.
2. Secondary outcomes: total consumption/welfare, livestock and nonfarm diversification,
   migration and borrowing, and school enrollment/attendance.
3. Mechanism and heterogeneity variables: market and road access, household assets, agricultural
   dependence, land class, proximity, fear level, and urban/rural status.

## Confirmed Preprocessing Decisions

1. Use 2007-2021 as the main linked sample and retain 2004 as an unlinked archive.
2. Prioritize agriculture and food security, treat broader consumption and education as secondary,
   and retain the proposed mechanism families.
3. Retain both continuous rainfall anomalies and bottom/top decile shock indicators.
4. Correct the single survey date from 1913 to 2013 while retaining the original date and a
   correction flag.
5. Do not impute or winsorize in the first harmonized release. Preserve nominal monetary values
   alongside the confirmed 2021 constant-price counterparts.
6. Treat the original CSES ZIP files as the canonical raw sources and read Stata modules directly
   from those archives. Do not retain duplicate expanded survey directories. Two small survey-
   instrument files that differed from their archived versions are preserved separately under
   `data/raw/CSE/source_supplements/`; neither is an analytical data input.

## Constructed Core Outcomes

The concept-level release contains 77,904 unique household-wave rows and 343,204 unique
person-wave education rows. CSES 2004 remains in both files for archival comparability, but only
the 62,920 households from 2007-2021 are marked as the main linked sample.

- Agriculture: parcel area, irrigability, cultivated and harvested area, crop quantity, crop
  diversity, yield, post-harvest loss, nominal production value, and nominal input costs.
- Food: reported total, purchased, and own-produced item values; per-member value; direct severe
  food-insecurity experiences where the question and coding are comparable.
- Education: ever attended, current attendance, years attended, expenditure, ages 6-17
  eligibility, sex, and person survey weights.
- Values remain missing when the source module/question is unavailable. No missing value is
  imputed and no outlier is winsorized. Every monetary outcome retains its nominal value and now
  has a separate 2021 constant-price counterpart.

## Monetary Harmonization

The IMF Consumer Price Index dataflow provides complete monthly Cambodia series for all items,
food and non-alcoholic beverages, and education from 2007 through 2021. The reported series are
sourced from national authorities and use October-December 2006 as their original reference
period. Each component is rebased so its 2021 annual mean equals 100.

- Food values use interview-month food CPI. The 10,075 households in the 2019 wave lack an
  interview-month field and use the explicitly flagged 2019 annual mean food CPI.
- Education expenditure uses annual mean education CPI because its reporting period spans the
  school year rather than only the interview month.
- Agricultural input costs and constructed crop production value use annual mean all-items CPI;
  physical area, production, and yield remain the preferred primary agricultural outcomes.
- The conversion is `real value = nominal value * 2021 component mean / period component CPI`.
  Nominal values, CPI values, deflation factors, and method flags all remain in the output.

## Processed Release Validation

- The complete 2011-12 PSU listing is used. A separate edited-name file contains only 224 of 360
  PSUs and is not used as the linkage base.
- The PSU-wave skeleton contains 5,617 unique rows with no duplicate keys. All have audited
  village geography; 736 have a recorded public mine-baseline point.
- The original strict commune-code CHIRPS linkage remains 5,175 of 5,617 PSU-wave rows. The
  enhanced, fully flagged linkage covers all 5,617 rows; 99.2% remain at commune resolution.
- The household-wave spine contains 77,904 unique rows: 62,920 in the 2007-2021 main sample and
  14,984 archived 2004 households. All main-sample households link to audited geography and the
  enhanced climate layer; the original strict link remains available for sensitivity analysis.
- Detailed CSES modules preserve their original item, parcel, person, household, or village grain.
  They have standardized identifiers and English-readable ASCII column names but are not yet
  aggregated into final conceptual outcomes.

## Reproducible Artifacts

- `src/preprocessing/inventory_direction3_sources.py`
- `src/preprocessing/audit_direction3_spatial_linkage.py`
- `src/preprocessing/prepare_direction3_external_candidates.py`
- `src/preprocessing/preprocess_direction3_external.py`
- `src/preprocessing/preprocess_direction3_cses.py`
- `src/preprocessing/build_direction3_analysis_skeleton.py`
- `src/preprocessing/repair_direction3_climate_linkage.py`
- `src/preprocessing/construct_direction3_core_outcomes.py`
- `src/preprocessing/acquire_cambodia_cpi.py`
- `src/preprocessing/write_direction3_decisions.py`
- `data/exp/data-preprocessing/direction3_dataset_registry.csv`
- `data/exp/data-preprocessing/direction3_candidate_variables.csv`
- `data/exp/data-preprocessing/direction3_cses_source_manifest.csv`
- `data/exp/data-preprocessing/direction3_cses_variable_dictionary.csv`
- `data/exp/data-preprocessing/decisions.json`
- `data/exp/data-preprocessing/direction3_processed_validation.csv`
- `data/exp/data-preprocessing/direction3_processed_release_manifest.csv`
- `data/exp/data-preprocessing/direction3_public_geography_sources.csv`
- `data/exp/data-preprocessing/direction3_climate_linkage_repair_validation.csv`
- `data/exp/data-preprocessing/direction3_core_outcome_dictionary.csv`
- `data/exp/data-preprocessing/direction3_core_outcome_validation.csv`
- `data/exp/data-preprocessing/direction3_cpi_source_manifest.csv`
- `data/exp/data-preprocessing/direction3_cpi_validation.csv`
- `data/exp/feasibility-check/direction3_spatial_linkage_by_wave.csv`
- `data/processed/mine_exposure_village_preprocessed.parquet`
- `data/processed/climate_commune_month_preprocessed.parquet`
- `data/processed/climate_commune_year_preprocessed.parquet`
- `data/processed/direction3_psu_year_analysis_skeleton_preprocessed.parquet`
- `data/processed/direction3_household_year_spine_preprocessed.parquet`
- `data/processed/direction3_psu_year_climate_enhanced_preprocessed.parquet`
- `data/processed/direction3_household_core_outcomes_preprocessed.parquet`
- `data/processed/direction3_education_core_outcomes_preprocessed.parquet`
- `data/processed/cambodia_cpi_monthly_preprocessed.parquet`
- `data/processed/cambodia_cpi_annual_preprocessed.parquet`
