# Feature Reference

This document provides a comprehensive reference for all blink features available in the `pyblinker/blink_features/` package. Features are categorized by their applicability: generic features shared across modalities, features specific to Eye Aspect Ratio (EAR) from video, and features specific to electrophysiological signals (EEG/EOG).

Unless noted otherwise:
- **Time** is measured in **seconds**.
- **Amplitude** is measured in the native units of the input signal (e.g., µV for EEG, unitless ratio for EAR).
- **Sampling Frequency (sfreq)** is in **Hertz (Hz)**.

Formulas use standard mathematical notation where $x[n]$ represents the signal at sample $n$, and $\Delta t = 1/sfreq$.

---

## 1. Shared Features (EAR, EEG, EOG)

These features can be computed for any detected blink, regardless of the recording modality. They describe the timing, shape, and dynamics of the blink event.

### A. Blink Event Statistics

These features quantify the occurrence and frequency of blinks over an epoch (a defined time window).

**Source:** `pyblinker/blink_features/blink_events/`

| Feature Name | Definition / Formula | Significance |
| :--- | :--- | :--- |
| **blink_total** | Count of valid blinks within the epoch. | Basic measure of blink activity. |
| **blink_rate** | $\frac{\text{blink\_total}}{\text{epoch\_duration\_seconds}} \times 60$ | Standard measure of fatigue; blink rate often increases with time-on-task but can decrease in high visual load. |
| **ibi_mean** | $\frac{1}{N-1} \sum_{i=1}^{N-1} (t_{start}[i+1] - t_{end}[i])$ | Mean time between consecutive blinks. Correlates with alertness. |
| **ibi_std** | Standard deviation of Inter-Blink Intervals (IBI). | Variability in blink timing; higher variability often indicates fatigue. |
| **ibi_cv** | $\frac{\text{ibi\_std}}{\text{ibi\_mean}}$ | Coefficient of variation of IBI. |
| **ibi_rmssd** | $\sqrt{\frac{1}{N-1}\sum (IBI_{i+1} - IBI_i)^2}$ | Root mean square of successive differences. Measures short-term variability (similar to HRV). |
| **poincare_sd1/sd2** | SD1 (width) and SD2 (length) of the Poincaré plot of IBIs. | Nonlinear measures of short-term vs. long-term variability. |
| **ibi_permutation_entropy** | Information theoretic measure of order in IBI sequences. | Detects predictable patterns in blinking behavior. |

### B. Kinematic & Morphological Features (Core)

These features describe the shape of a single blink waveform. They are computed relative to specific landmarks (e.g., base, zero-crossing, tent-fit).

**Source:** `pyblinker/blink_features/_core_blink.py`

**Note:** Feature names often include a suffix indicating the segmentation method (e.g., `_base`, `_zero`, `_tent`).
- **EAR**: Uses `_base`, `_tent` (relative to open-eye baseline).
- **EEG/EOG**: Uses `_base`, `_zero` (zero-crossing), `_tent`, `_half_*`.

#### 1. Duration and Timing
| Feature Name | Definition | Variables | Significance |
| :--- | :--- | :--- | :--- |
| **rise_time_peak** | $t_{peak} - t_{start}$ | Time from onset to peak closure/amplitude. | Slower closing phase can indicate drowsiness. |
| **fall_time_peak** | $t_{end} - t_{peak}$ | Time from peak to full reopening. | Reopening phase is often more affected by fatigue than closing. |
| **rise_time_10_90** | Time from 10% to 90% amplitude during closing. | Robust measure of closing speed, ignoring onset tails. |
| **fall_time_10_90** | Time from 90% to 10% amplitude during reopening. | Robust reopening duration. |
| **half_width** | Duration where amplitude $\ge 0.5 \times \text{max\_amp}$. | "FWHM" (Full Width at Half Maximum). Describes blink broadness. |

#### 2. Velocity and Acceleration
| Feature Name | Definition | Significance |
| :--- | :--- | :--- |
| **vel_peak_abs** | $\max(|v[n]|)$ where $v[n] = \frac{x[n] - x[n-1]}{\Delta t}$ | Maximum speed of the eyelid. Saccadic velocities decrease with fatigue. |
| **vel_mean_abs** | $\frac{1}{N} \sum |v[n]|$ | Average speed throughout the blink. |
| **slope_rise_pos** | $\max(v[n])$ (positive slope) | Maximum closing velocity (for EEG/EOG) or opening (for EAR depending on polarity). |
| **slope_fall_neg** | $\min(v[n])$ (negative slope) | Maximum reopening velocity (for EEG/EOG) or closing (for EAR). |
| **acc_peak_abs** | $\max(|a[n]|)$ where $a[n] = \frac{v[n] - v[n-1]}{\Delta t}$ | Peak acceleration. Reflects neuromuscular force. |

#### 3. Area and Symmetry
| Feature Name | Definition | Significance |
| :--- | :--- | :--- |
| **area_abs_total_trapz** | $\int |x(t)| dt$ (via trapezoidal rule) | Total "strength" or magnitude of the blink event. |
| **symmetry_trapz** | $\frac{\text{Area}_{left} - \text{Area}_{right}}{\text{Area}_{total}}$ | Asymmetry between closing and reopening phases. Fatigue often prolongs the reopening phase, altering symmetry. |

#### 4. Amplitude
| Feature Name | Definition | Significance |
| :--- | :--- | :--- |
| **amp_peak_abs** | $|x_{peak}|$ | Maximum excursion from baseline. |
| **amp_peak_to_trough** | $x_{max} - x_{min}$ | Total dynamic range of the blink. |

---

## 2. EAR-specific Features (Video/Camera)

Features derived from the Eye Aspect Ratio (EAR) signal. EAR is a unitless ratio of eye height to eye width. Blinks appear as downward dips in the signal.

### A. Open-Eye State Features
Features computed from the periods *between* blinks, reflecting the baseline state of the eye.

**Source:** `pyblinker/blink_features/open_eye/features/`

| Feature Name | Definition | Variables | Significance |
| :--- | :--- | :--- | :--- |
| **perclos** | $\frac{\text{Samples} \le \text{Threshold}}{\text{Total Samples}} \times 100$ | `Threshold`: typically 80% of baseline. | **Gold standard** for drowsy driving detection. Measures "slow eye closures". |
| **baseline_drift** | Slope of linear regression on open-eye EAR samples. | Negative slope indicates gradual eyelid drooping (ptosis) associated with drowsiness. |
| **micropause_count** | Count of transient drops ($0.1s < duration < 0.3s$) that are not full blinks. | Detects brief lapses in attention or "microsleep" precursors. |
| **zero_crossing_rate** | Rate of velocity sign changes during open eyes. | High ZCR implies jittery/unstable gaze; low ZCR implies fixated/staring state. |
| **eye_opening_rms** | $\sqrt{\frac{1}{N} \sum (x[n] - \mu)^2}$ | RMS of open-eye signal. | Measures stability of the eye opening. |

### B. EAR Blink Metrics
Features describing the specific shape of EAR "dips".

**Source:** `pyblinker/blink_features/ear_metrics/feature_extraction.py`

| Feature Name | Definition / Calculation | Significance |
| :--- | :--- | :--- |
| **ear_blink_depth** | $\text{baseline} - \text{min\_ear}$ | How fully the eye closed. Partial blinks are common in distraction/fatigue. |
| **closed_duration_seconds** | Time where $\text{EAR} < \text{Threshold}$. | Duration of functional blindness during a blink. |
| **auc_below_threshold** | $\sum (\text{Threshold} - \text{EAR}[n]) \times \Delta t$ | "Area Under Curve" of the closure. Combines duration and depth. |
| **refined_closing_slope** | Slope of line from start to lowest point. | Speed of eyelid descent. |
| **refined_opening_slope** | Slope of line from lowest point to end. | Speed of eyelid ascent. |
| **interpolated_closing_slope** | Slope using interpolated threshold crossing times. | More precise slope measurement independent of sampling rate quantization. |
| **ear_kurtosis/skewness** | Statistical shape descriptors of the EAR distribution within the blink. | Shape deviations from a "normal" bell-curve like blink. |

---

## 3. EEG/EOG-specific Features (Electrophysiology)

Features derived from voltage signals (EOG, EEG). Blinks appear as high-amplitude, positive (or negative depending on montage) bell-shaped curves.

### A. Energy & Complexity
These features capture the signal power and complexity, useful for artifact rejection and discriminating blinks from other artifacts.

**Source:** `pyblinker/blink_features/energy/`

| Feature Name | Definition | Significance |
| :--- | :--- | :--- |
| **blink_signal_energy** | $\sum x[n]^2$ | Total energy. Distinguishes large blinks from small saccades. |
| **teager_kaiser_energy** | $\sum |x[n]^2 - x[n-1]x[n+1]|$ | TKEO emphasizes instantaneous changes in energy/frequency. Good for onset detection. |
| **blink_line_length** | $\sum |x[n] - x[n-1]|$ | "Coastline" of the signal. Measures waveform complexity/fractal dimension. |
| **blink_velocity_integral** | $\int |v(t)| dt$ | Total path length of the velocity profile. |

### B. Frequency Domain (Wavelets)
Decomposition of the blink waveform into frequency bands using Discrete Wavelet Transform (DWT).

**Source:** `pyblinker/blink_features/frequency_domain/features.py`

| Feature Name | Definition | Variables | Significance |
| :--- | :--- | :--- |
| **wavelet_energy_d1..d4** | Sum of squared coefficients for detail levels D1-D4 (using `db4` wavelet). | Bands: D1 (High freq) $\to$ D4 (Low freq). | Blinks are low-frequency events (typically D3-D4). High energy in D1/D2 indicates muscle noise (EMG) or artifacts superimposed on the blink. |

### C. Zero-Crossing Features (Legacy/EOG)
Specific time-domain features reliant on the signal crossing zero (common in AC-coupled EOG).

**Source:** `pyblinker/blink_features/waveform_features/extract_blink_properties.py`

| Feature Name | Definition | Significance |
| :--- | :--- | :--- |
| **duration_zero** | Time between upward and downward zero crossings. | Classic definition of blink duration in EOG studies. |
| **pos_amp_vel_ratio_zero** | Ratio of Amplitude to Velocity during opening phase (zero-based). | Used to distinguish blinks (bell-shape) from saccades (step-shape). |
| **time_shut_zero** | Duration signal stays above a specific amplitude threshold (e.g., 50% max). | Measures the "dwell time" of the eye in the closed state. |