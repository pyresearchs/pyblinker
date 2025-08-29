# Blink Properties Unit Tests

This folder contains unit tests for verifying the correctness and robustness of blink feature extraction in EEG signals, inspired by functionality from the **`pyblinker`** package.

## 📌 Purpose

The tests here validate the extraction of physiologically meaningful blink properties from candidate EEG signals and associated blink metadata.

These tests ensure that the `BlinkProperties` class:

- Correctly computes durations, amplitude ratios, and shut times.
- Produces outputs in the expected structure.
- Enforces logical and physical constraints (e.g., no NaNs, positive durations, valid velocity estimates).

---

## 📂 Test Files

| File                             | Description |
|----------------------------------|-------------|
| `test_blink_features.py`         | Tests low-level feature calculations such as zero crossings, fit range detection, and slope intersection for blinks. |
| `test_blink_properties.py`       | Tests the full `BlinkProperties` pipeline: duration calculation, amplitude-velocity ratios, shut durations, and blink timing features. |

---

## 🧪 Blink Properties Tested

The tests cover the following blink-related metrics:

- **Blink Durations**:
    - `duration_base`, `duration_zero`, `duration_tent`
    - `duration_half_base`, `duration_half_zero`

- **Amplitude-Velocity Ratios (AVR)**:
    - `pos_amp_vel_ratio_zero`, `neg_amp_vel_ratio_zero`
    - `pos_amp_vel_ratio_base`, `neg_amp_vel_ratio_base`
    - `pos_amp_vel_ratio_tent`, `neg_amp_vel_ratio_tent`

- **Shut Time Durations**:
    - `time_shut_base`, `time_shut_zero`, `time_shut_tent`

- **Opening and Closing Times**:
    - `closing_time_zero`, `reopening_time_zero`
    - `closing_time_tent`, `reopening_time_tent`

- **Blink Peaks**:
    - `peak_max_blink`, `peak_max_tent`
    - `peak_time_blink`, `peak_time_tent`

- **Inter-blink Features**:
    - `inter_blink_max_amp`
    - `inter_blink_max_vel_base`, `inter_blink_max_vel_zero`

---

## 📁 Test Inputs

The tests rely on fixtures located in `test/test_files`:

- `S1_candidate_signal.npy` — 1D EEG signal for blink detection
- `blink_properties_fits.pkl` — Precomputed blink metadata, including landmarks and linear fits

---

## 🔄 Variable and Function Name Changes

| Old Name | New Name | Affected File | Function/Context |
| -------- | -------- | ------------- | ---------------- |
| `fitBlink` | `fit_blink` | `fit_blink.py` | Function name |
| `baseLeft` | `base_left` | `base_left_right.py` | Variable name |
| `startBlinks` | `start_blink` | `fit_blink.py` | DataFrame column |
| `endBlinks` | `end_blink` | `fit_blink.py` | DataFrame column |
| `maxValue` | `max_value` | `fit_blink.py` | DataFrame column |
| `maxFrame` | `max_blink` | `fit_blink.py` | DataFrame column |
| `outerStarts` | `outer_start` | `fit_blink.py` | DataFrame column |
| `outerEnds` | `outer_end` | `fit_blink.py` | DataFrame column |
| `leftZero` | `left_zero` | `fit_blink.py` | DataFrame column |
| `rightZero` | `right_zero` | `fit_blink.py` | DataFrame column |
| `leftBase` | `left_base` | `base_left_right.py` | DataFrame column |
| `rightBase` | `right_base` | `base_left_right.py` | DataFrame column |
| `maxPosVelFrame` | `max_pos_vel_frame` | `base_left_right.py` | Column name |
| `maxNegVelFrame` | `max_neg_vel_frame` | `base_left_right.py` | Column name |
| `leftBaseHalfHeight` | `left_base_half_height` | `fit_blink.py` | Column name |
| `rightBaseHalfHeight` | `right_base_half_height` | `fit_blink.py` | Column name |
| `leftZeroHalfHeight` | `left_zero_half_height` | `fit_blink.py` | Column name |
| `rightZeroHalfHeight` | `right_zero_half_height` | `fit_blink.py` | Column name |
| `averLeftVelocity` | `aver_left_velocity` | `fit_blink.py` | Column name |
| `averRightVelocity` | `aver_right_velocity` | `fit_blink.py` | Column name |
| `leftXIntercept` | `left_x_intercept` | `fit_blink.py` | Column name |
| `rightXIntercept` | `right_x_intercept` | `fit_blink.py` | Column name |
| `leftXIntercept_int` | `left_x_intercept_int` | `fit_blink.py` | Column name |
| `rightXIntercept_int` | `right_x_intercept_int` | `fit_blink.py` | Column name |
| `xIntersect` | `x_intersect` | `fit_blink.py` | Column name |
| `yIntersect` | `y_intersect` | `fit_blink.py` | Column name |
| `peaksPosVelBase` | `peaks_pos_vel_base` | `extract_blink_properties.py` | Column name |
| `peaksPosVelZero` | `peaks_pos_vel_zero` | `extract_blink_properties.py` | Column name |
| `extractBlinkProps` | `extract_blink_properties` | `extract_blink_properties.py` | Function name |
| `durationBase` | `duration_base` | `extract_blink_properties.py` | Column name |
| `durationZero` | `duration_zero` | `extract_blink_properties.py` | Column name |
| `durationTent` | `duration_tent` | `extract_blink_properties.py` | Column name |
| `durationHalfBase` | `duration_half_base` | `extract_blink_properties.py` | Column name |
| `durationHalfZero` | `duration_half_zero` | `extract_blink_properties.py` | Column name |
| `posAmpVelRatioZero` | `pos_amp_vel_ratio_zero` | `extract_blink_properties.py` | Column name |
| `negAmpVelRatioZero` | `neg_amp_vel_ratio_zero` | `extract_blink_properties.py` | Column name |
| `posAmpVelRatioBase` | `pos_amp_vel_ratio_base` | `extract_blink_properties.py` | Column name |
| `negAmpVelRatioBase` | `neg_amp_vel_ratio_base` | `extract_blink_properties.py` | Column name |
| `posAmpVelRatioTent` | `pos_amp_vel_ratio_tent` | `extract_blink_properties.py` | Column name |
| `negAmpVelRatioTent` | `neg_amp_vel_ratio_tent` | `extract_blink_properties.py` | Column name |
| `timeShutBase` | `time_shut_base` | `extract_blink_properties.py` | Column name |
| `timeShutZero` | `time_shut_zero` | `extract_blink_properties.py` | Column name |
| `timeShutTent` | `time_shut_tent` | `extract_blink_properties.py` | Column name |
| `closingTimeZero` | `closing_time_zero` | `extract_blink_properties.py` | Column name |
| `reopeningTimeZero` | `reopening_time_zero` | `extract_blink_properties.py` | Column name |
| `closingTimeTent` | `closing_time_tent` | `extract_blink_properties.py` | Column name |
| `reopeningTimeTent` | `reopening_time_tent` | `extract_blink_properties.py` | Column name |
| `peakMaxBlink` | `peak_max_blink` | `extract_blink_properties.py` | Column name |
| `peakMaxTent` | `peak_max_tent` | `extract_blink_properties.py` | Column name |
| `peakTimeBlink` | `peak_time_blink` | `extract_blink_properties.py` | Column name |
| `peakTimeTent` | `peak_time_tent` | `extract_blink_properties.py` | Column name |
| `interBlinkMaxAmp` | `inter_blink_max_amp` | `extract_blink_properties.py` | Column name |
| `interBlinkMaxVelBase` | `inter_blink_max_vel_base` | `extract_blink_properties.py` | Column name |
| `interBlinkMaxVelZero` | `inter_blink_max_vel_zero` | `extract_blink_properties.py` | Column name |
| `_max_pos_vel_frame` | `max_pos_vel_frame` | `zero_crossing.py` | Function name |
| `_get_left_base` | `get_left_base` | `zero_crossing.py` | Function name |
| `_get_right_base` | `get_right_base` | `zero_crossing.py` | Function name |
| `_get_half_height` | `get_half_height` | `zero_crossing.py` | Function name |

Column names now use singular nouns such as `start_blink` or `end_blink` to emphasize a single blink event rather than multiple events.

The test data provided in `*.pkl` files may contain the old variable names. Use the utility script in `utils/update_pkl_variables.py` to remap them before running tests:

```bash
python utils/update_pkl_variables.py
```

This script updates both `blink_properties_fits.pkl` and `file_test_blink_position.pkl` in place.

---

## ✅ Running the Tests

From the `test.features/pyblinker/` directory, run:

```bash
python run_selected_tests.py
```

