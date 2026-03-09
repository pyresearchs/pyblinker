# Feature Naming Convention

This project uses a **strict, parseable naming convention** so agents (and code) can generate, validate, and decode feature names consistently.

---

## 1) Canonical format

### Feature-level signals

```
<modality>__<style>__<feature>__<metric>_<stat>__<channel>
```

* `__` (double underscore) separates the **major fields**
* `_` (single underscore) separates `<metric>` from `<stat>`
* `<stat>` is always the **last token before** the final `__<channel>`

✅ Example (from morphology/zero):

```
eeg__zero__morphology__duration_zero_mean__EEG-E8
```

---

## 2) Allowed values

### 2.1 `<stat>`

One of:

* `mean`
* `std`
* `cv`

---

### 2.2 `<modality>`

One of:

* `eeg`
* `eog`
* `ear`

---

### 2.3 `<style>`

**For `eeg` and `eog`:**

* `zero`
* `base`
* `tent`
* `half`
* `peak`

> Note: morphology examples also include `inter_blink` as a style; treat it as a valid style **for morphology** when applicable.

**For `ear`:**

* `th_point`
* `th_interpolation`

---

### 2.4 `<feature>`

Supported feature groups:

* `Blink_events`
* `energy`
* `kinematics`
* `morphology`
* `open_eye`

> Keep feature strings consistent with what your pipeline expects (case-sensitive if your downstream code is case-sensitive).

---

## 3) `<metric>` by feature

### 3.1 Feature = `energy`

Allowed metrics:

* `wavelet_energy_d1`
* `wavelet_energy_d2`
* `wavelet_energy_d3`
* `wavelet_energy_d4`
* `blink_signal_energy`
* `teager_kaiser_energy`
* `blink_velocity_integral`

**Naming example**

```
eog__base__energy__wavelet_energy_d2_std__EOG-<CH>
```

---

### 3.2 Feature = `kinematics`

Allowed metrics:

* `amp_vel_ratio_base`
* `amp_vel_ratio_tent`
* `amp_vel_ratio_zero_to_max`
* `blink_velocity`
* `inter_blink_max_vel`
* `inter_blink_max_vel_base`
* `inter_blink_max_vel_zero`
* `aver_left_velocity`
* `aver_right_velocity`
* `neg_amp_vel_ratio_base`
* `pos_amp_vel_ratio_base`
* `neg_amp_vel_ratio_zero`
* `pos_amp_vel_ratio_zero`
* `neg_amp_vel_ratio_tent`
* `pos_amp_vel_ratio_tent`

**Naming example**

```
eeg__tent__kinematics__blink_velocity_cv__EEG-E8
```

---

## 4) Morphology rules (most complex)

Morphology metrics fall into **two buckets**:

### 4.1 Style-dependent morphology metrics (EEG/EOG only)

For `eeg` and `eog`, some morphology metrics explicitly depend on the style and **embed the style into the metric name** (e.g., `duration_zero`, `closing_time_tent`, etc.).

#### Style = `zero`

Metrics:

* `duration_zero`
* `closing_time_zero`
* `reopening_time_zero`
* `time_shut_zero`

Examples:

```
eeg__zero__morphology__duration_zero_mean__EEG-E8
eeg__zero__morphology__closing_time_zero_std__EEG-E8
```

#### Style = `base`

Metrics:

* `duration_base`
* `time_shut_base`

Examples:

```
eeg__base__morphology__duration_base_cv__EEG-E8
```

#### Style = `tent`

Metrics:

* `duration_tent`
* `closing_time_tent`
* `reopening_time_tent`
* `time_shut_tent`

#### Style = `half`

Metrics:

* `duration_half_base`
* `duration_half_zero`

#### Style = `peak`

Metrics:

* `peak_time_blink`
* `peak_time_tent`
* `peak_max_blink`
* `peak_max_tent`

#### Style = `inter_blink`

Metrics:

* `inter_blink_max_amp`

---

### 4.2 General morphology metrics (EEG/EOG/EAR)

In addition to the style-dependent set above, morphology also includes other metrics such as:

* `amp_peak_abs_base`
* `amp_peak_signed_base`
* `amp_peak_to_trough_base`
* `amp_trough_signed_base`
* `area_abs_total_rect_base`
* `area_abs_total_trapz_base`
* `duration`
* `fall_time_10_90_base`
* `fall_time_peak_base`
* `half_width_base`
* `rise_time_10_90_base`
* `rise_time_peak_base`
* `symmetry_rect_base`
* `symmetry_trapz_base`

**Naming example**

```
eog__base__morphology__half_width_base_mean__EOG-<CH>
```

---

### 4.3 Morphology for `ear`

For `ear`, morphology includes (at least) these metrics:

* `duration_zero`
* `closing_time_zero`
* `reopening_time_zero`
* (…extend this list as your extractor defines additional ear morphology metrics)

**Naming example**

```
ear__th_point__morphology__duration_zero_mean__EAR-<CH>
```

---

## 5) `<channel>`

`<channel>` is the physical/logical channel identifier and is appended at the end:

Examples:

* `EEG-E8` (from your examples)
* `EOG-...`
* `EAR-...`

**Rule of thumb:** channel strings are typically **not normalized** (may include uppercase and hyphens). Keep them exactly as provided by the data source.

---

## 6) Segmentation naming (special case)

Segmentation signals use a different convention:

```
start__<style>__ear
end__<style>__ear
```

Examples:

```
start__th_point__ear
end__th_point__ear
```

> Segmentation names do **not** include `<stat>` or `<channel>` in the pattern you provided.

---

## 7) Validation / parsing (recommended)

A simple way to parse feature names is:

1. Split by `__` into 5 parts:

    * modality, style, feature, metric+stat, channel
2. Split `metric+stat` by the **last** `_` into:

    * metric, stat

This ensures metrics that contain underscores (like `wavelet_energy_d3`) still work.

---

## 8) Quick examples

* Energy:

  ```
  eeg__base__energy__wavelet_energy_d1_mean__EEG-E8
  ```

* Kinematics:

  ```
  eog__zero__kinematics__inter_blink_max_vel_std__EOG-<CH>
  ```

* Morphology (style-dependent):

  ```
  eeg__tent__morphology__time_shut_tent_cv__EEG-E8
  ```

* Morphology (general):

  ```
  eeg__base__morphology__symmetry_trapz_base_mean__EEG-E8
  ```

* Segmentation:

  ```
  start__th_interpolation__ear
  ```

---
