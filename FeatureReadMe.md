# Feature Reference: EAR Threshold Refinement

This document summarizes the EAR features, landmarks, and interpolated threshold crossings that are exported by the threshold-refinement workflow. Unless noted otherwise, times are measured in **seconds** and sample indices are integer positions in the signal. Time values derive from samples as:

```
time_seconds = sample_index / sfreq
```

where `sfreq` is the sampling frequency in Hertz.

## Threshold and closure metrics

- **classification_threshold** (EAR units): Threshold used to classify the blink.
- **threshold_value** (EAR units): Candidate EAR threshold being evaluated.
- **closed_duration_seconds** (seconds): Total time the EAR stayed below the threshold.
- **closed_fraction** (unitless fraction): Portion of the blink window under the threshold.
- **time_under_threshold_seconds** (seconds): Synonym for `closed_duration_seconds`.
- **time_under_threshold_fraction** (unitless): Synonym for `closed_fraction`.
- **auc_below_threshold** (EAR-seconds): Integral of `(threshold - EAR)` while the EAR is below the threshold, representing blink depth over time.

## Classification outputs

- **blink_classification** (string): Final label for the blink, either taken from `blink_type` overrides or the computed value.
- **blink_classification_computed** (string): Computed label describing the blink shape (e.g., `"full"` or `"partial"`).
- **blink_type_original** (string): Original annotation label if present.

## Distribution and descriptive statistics (EAR units unless noted)

- **ear_mean**, **ear_median**, **ear_std**, **ear_var**: Mean, median, standard deviation, and variance of EAR within the blink.
- **ear_mad**, **ear_iqr**: Median absolute deviation and interquartile range of EAR.
- **ear_skewness**, **ear_kurtosis**: Skewness and kurtosis of EAR distribution.
- **ear_min**, **ear_max**: Minimum and maximum EAR values.
- **ear_time_of_min** (seconds): Time of the minimum EAR value.
- **ear_baseline**: Baseline EAR estimate around the blink.
- **ear_blink_depth**: Blink depth, computed as `ear_baseline - ear_min` (EAR units).

## Dynamics (slopes, speeds, acceleration)

All slope/acceleration features use EAR units; time-based results use seconds. Negative slopes indicate closing (EAR decreasing), and positive slopes indicate reopening (EAR increasing).

- **max_closing_speed**, **max_opening_speed**: Peak negative/positive slopes during closing and reopening phases.
- **max_negative_slope**, **max_positive_slope**: Maximum absolute negative/positive slopes across the blink.
- **mean_closing_slope**, **mean_reopening_slope**: Average slopes during closing and reopening.
- **max_negative_acceleration**, **max_positive_acceleration**: Peak negative/positive acceleration of the EAR signal.
- **time_to_close**, **time_to_reopen** (seconds): Duration of the closing and reopening phases.
- **refined_closing_slope**, **refined_opening_slope**: Slopes computed using refined start/end landmarks for improved robustness.
- **interpolated_closing_slope**, **interpolated_opening_slope**: Slopes derived from interpolated threshold crossings to emphasize threshold-aligned timing.

## Candidate and refinement metadata

- **candidate_id**: Unique identifier for the blink candidate.
- **blink_type**: Blink label from annotations when provided.
- **coarse_onset_time**, **coarse_offset_time** (seconds): Original annotated onset/offset times.
- **coarse_duration** (seconds): Duration from the coarse annotations.
- **refined_onset_time**, **refined_offset_time** (seconds): Refined onset/offset after threshold refinement.
- **refined_duration** (seconds): Refined offset minus refined onset.
- **onset_offset_seconds**, **offset_offset_seconds** (seconds): Differences between refined and coarse onset/offset.
- **coarse_start_sample**, **coarse_end_sample**: Sample indices for the coarse window.
- **refined_start_sample**, **refined_end_sample**: Sample indices for the refined window.
- **refined_lowest_point_sample**: Sample index of the lowest EAR within the refined window.
- **threshold_crossing_found** (boolean): True when a threshold crossing triplet was found for the candidate.

## Interpolated threshold crossing landmarks

- **left_interpolated_threshold**, **right_interpolated_threshold** (seconds): Downward and upward threshold crossings computed via linear interpolation between adjacent samples surrounding the crossing points.
- **left_interpolated_threshold_sample**, **right_interpolated_threshold_sample**: Integer sample indices nearest to the interpolated crossings (used for plotting/reference; may differ slightly from the exact crossing times).
- **left_interpolated_threshold_found**, **right_interpolated_threshold_found** (boolean): Flags indicating whether each interpolated crossing was successfully located.
- **interpolated_thresholds_found** (boolean): True when both interpolated crossings were found.

Interpolation details:

- Crossings are searched within a padded window around the refined start/end samples.
- The crossing time is linearly interpolated using the EAR values immediately before and after the threshold sign change.
- Sample-index versions are derived for convenience and should be interpreted as the nearest plotted sample, not the exact crossing time.

## Landmark glossary and visualization notes

- **Coarse start/end**: Initial annotation window boundaries.
- **Refined start/end**: Start/end derived from threshold refinement. In plots these appear as **diamond markers** aligned to their (time, EAR) coordinates.
- **Refined lowest point**: Sample of the minimum EAR within the refined window; also shown as a diamond marker.
- **Threshold crossings (left/right)**: Downward and upward crossings relative to the threshold. Interpolated versions may be drawn with triangular or distinct markers to differentiate them from refined landmarks.

All landmarks are plotted at `time = sample / sfreq` with the corresponding EAR value to ensure consistent alignment across the report figures.
