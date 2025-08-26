### Table 1: Features That **CAN** Be Calculated

These features are robust because they depend on fundamental blink landmarks and raw signal properties, which are available even for blinks that could not be fully fitted with the 'tent' model.

| Feature Category | Feature | Depends On (Input Columns) | Function | Remark |
| :--- | :--- | :--- | :--- | :--- |
| **Base and Zero-Crossing Durations** | `duration_base` | `right_base`, `left_base` | `set_blink_duration` | Relies on fundamental base landmarks, which are assumed to be present. |
| | `duration_zero` | `right_zero`, `left_zero` | `set_blink_duration` | Relies on fundamental zero-crossing landmarks, which are assumed to be present. |
| **Base and Zero-Crossing A/V Ratios** | `pos_amp_vel_ratio_zero` | `left_zero`, `max_blink` | `set_blink_amp_velocity_ratio_zero_to_max`| Uses **raw signal velocity** between landmarks, not fitted line slopes. |
| | `neg_amp_vel_ratio_zero` | `max_blink`, `right_zero` | `set_blink_amp_velocity_ratio_zero_to_max`| Uses **raw signal velocity** between landmarks, not fitted line slopes. |
| | `pos_amp_vel_ratio_base` | `left_base`, `max_blink` | `amplitude_velocity_ratio_base` | Uses **raw signal velocity** between landmarks, not fitted line slopes. |
| | `neg_amp_vel_ratio_base` | `max_blink`, `right_base` | `amplitude_velocity_ratio_base` | Uses **raw signal velocity** between landmarks, not fitted line slopes. |
| **Base and Zero-Crossing Shut Times** | `closing_time_zero` | `max_blink`, `left_zero` | `time_zero_shut` | Calculation is based on the timing of fundamental landmarks. |
| | `reopening_time_zero`| `right_zero`, `max_blink` | `time_zero_shut` | Calculation is based on the timing of fundamental landmarks. |
| | `time_shut_base` | `left_base`, `right_base`, `max_value` | `time_zero_shut` & `time_base_shut` | Based on fundamental landmarks and raw signal amplitude. (Note: This is calculated redundantly). |
| **Blink Peak Properties (from raw signal)**| `peak_max_blink` | `max_value` | `extract_other_times` | Derived directly from the detected maximum peak of the **raw signal**. |
| | `peak_time_blink` | `max_blink` | `extract_other_times` | Derived directly from the detected maximum peak of the **raw signal**. |
| **Inter-Blink Intervals**| `inter_blink_max_amp`| `max_blink` | `extract_other_times` | Calculated from the timing of the raw signal's maximum amplitude peaks. |
| | `inter_blink_max_vel_base` | `peaks_pos_vel_base` | `extract_other_times` | Calculated from the timing of maximum velocity peaks found using fundamental landmarks. |
| | `inter_blink_max_vel_zero` | `peaks_pos_vel_zero` | `extract_other_times` | Calculated from the timing of maximum velocity peaks found using fundamental landmarks. |

---

### Table 2: Features That **CANNOT** Be Calculated

These features will fail because their calculation depends on columns from `cols_fit_range`, `cols_lines_intesection`, or `cols_half_height`, which are missing for blinks dropped by the upstream `dropna()` call.

| Feature Category | Feature | Depends On (Problematic Input Columns) | Function | Remark |
| :--- | :--- | :--- | :--- | :--- |
| **Tent-Based Durations** | `duration_tent` | `right_x_intercept`, `left_x_intercept` | `set_blink_duration` | Relies on x-intercepts calculated from the linear 'tent' fit, which are unavailable for dropped blinks. |
| **Half-Height Durations**| `duration_half_base` | `right_base_half_height`, `left_base_half_height` | `set_blink_duration` | Relies on **half-height crossing points** (`cols_half_height`), which are unavailable for dropped blinks. |
| | `duration_half_zero` | `right_zero_half_height`, `left_zero_half_height` | `set_blink_duration` | Relies on **half-height crossing points** (`cols_half_height`), which are unavailable for dropped blinks. |
| **Tent-Based A/V Ratios** | `neg_amp_vel_ratio_tent`| `aver_right_velocity` | `amplitude_velocity_ratio_tent` | Relies on the **average velocities (slopes)** derived from the 'tent' line fits. |
| | `pos_amp_vel_ratio_tent`| `aver_left_velocity` | `amplitude_velocity_ratio_tent` | Relies on the **average velocities (slopes)** derived from the 'tent' line fits. |
| **Tent-Based Shut Times** | `closing_time_tent` | `x_intersect`, `left_x_intercept` | `time_base_shut` | Relies on 'tent' fit landmarks (intercepts and intersection point) which are unavailable for dropped blinks. |
| | `reopening_time_tent` | `right_x_intercept`, `x_intersect` | `time_base_shut` | Relies on 'tent' fit landmarks (intercepts and intersection point) which are unavailable for dropped blinks. |
| | `time_shut_tent` | `left_x_intercept`, `right_x_intercept` | `time_base_shut` | Relies on 'tent' fit landmarks (intercepts and intersection point) which are unavailable for dropped blinks. |
| **Tent-Based Peak Properties**| `peak_max_tent` | `y_intersect` | `extract_other_times` | Relies on the calculated **intersection point (x, y)** of the 'tent' fit lines, not the raw signal peak. |
| | `peak_time_tent` | `x_intersect` | `extract_other_times` | Relies on the calculated **intersection point (x, y)** of the 'tent' fit lines, not the raw signal peak. |
---

### Mitigating NaNs with `run_fit`

`compute_segment_blink_properties` now accepts a `run_fit` flag. When disabled (the default), the fitting stage is skipped to preserve all raw-signal features in Table 1. Enabling it reintroduces tent-based metrics from Table 2 but may drop blinks due to NaNs in the fit range. This approach maintains backward compatibility with the legacy blinker code while allowing advanced fits when desired.

### Legacy Compatibility Notes

The original MATLAB-based **blinker** toolbox always executed the fitting stage
to obtain tent-derived metrics.  ``pyear`` preserves those column names and
calculation semantics.  When ``run_fit=True`` all columns (including
``duration_tent`` and ``peak_time_tent``) are emitted so that existing analysis
scripts continue to operate without modification.  Leaving ``run_fit`` disabled
avoids NaN-related blink drops but still computes robust features, enabling a
gradual migration path from the legacy code.
