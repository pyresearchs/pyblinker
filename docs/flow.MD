```md
START: [ MNE Epochs (30s @ 30Hz) + epoch.metadata ]

|
|---------------------------------------------------------------------|
|  A) INPUT VALIDATION + INDEXING                                      |
|---------------------------------------------------------------------|
V
+-----------------------------------------------------------------------+
| A1. Verify Epoch structure                                             |
|   - n_times ≈ 3000 samples                                              |
|   - modalities available: EAR, EEG, EOG                                |
|   - confirm metadata contains blink fields                             |
+-----------------------------------------------------------------------+
|
V
+-----------------------------------------------------------------------+
| A2. Read metadata boundary keys                                         |
|   - onset__<style>__<modality>                                    |
|   - duration__<style>__<modality>                                       |
|   styles: zero, base, interpolated, ...                                 |
|   modalities: ear, eeg, eog                                             |
+-----------------------------------------------------------------------+
|
V
+-----------------------------------------------------------------------+
| A3. Construct BlinkBoundaryTable (ONE TIME)                             |
|   For each epoch_id:                                                   |
|      For each modality in {ear, eeg, eog}:                             |
|     	For each style in {zero, base, interpolated, th_interpolation,...}:                |
|         - onset = metadata[onset__style__modality]            |
|         - dur   = metadata[duration__style__modality]               |
|         - offset_sec = onset_sec + dur_sec                              |
|         - onset_samp, offset_samp = sec→sample (30Hz)                   |
|         - add validity flags (in range, dur>min, not NaN, etc.)         |
+-----------------------------------------------------------------------+
|
V
+-----------------------------------------------------------------------+
| A4. Boundary QA / Normalization                                         |
|   - clip to [0, n_times-1]                                              |
|   - drop/flag invalid windows                                           |
|   - optional: enforce min window length (e.g., >= 3 samples at 30Hz)    |
|   OUTPUT: Clean BlinkBoundaryTable                                      |
+-----------------------------------------------------------------------+

|
|---------------------------------------------------------------------|
|  B) PREPROCESS ONCE PER EPOCH × MODALITY                              |
|---------------------------------------------------------------------|
V
+-----------------------------------------------------------------------+
| B1. For each epoch_id and modality in {EAR, EEG, EOG}:                  |
|   Load raw trace: x_raw                                                 |
+-----------------------------------------------------------------------+
|
V
+-----------------------------------------------------------------------+
| B2. Shared preprocessing (cached)                                       |
|   - baseline correction / detrend                                       |
|   - normalization (z-score or robust scaling)                           |
|   - smoothing (important at 30Hz)                                       |
|   OUTPUT: x (normalized), x_smooth                                      |
+-----------------------------------------------------------------------+
|
V
+-----------------------------------------------------------------------+
| B3. Derived traces (cached, reused by ALL styles)                       |
|   - dx = derivative(x_smooth)                                           |
|   - d2x = second derivative (optional)                                  |
|   - envelope / moving RMS (optional)                                    |
|   OUTPUT: x, x_smooth, dx, d2x, envelope                                |
+-----------------------------------------------------------------------+
|
|---------------------------------------------------------------------|
|  C) OPTIONAL: EXPENSIVE TRANSFORMS ONCE PER EPOCH × MODALITY          |
|---------------------------------------------------------------------|
V
+-----------------------------------------------------------------------+
| C1. Frequency strategy selection (choose one)                           |
|   [Preferred at 30Hz]  Per-window Welch/FFT (fast)                      |
|   [If time-freq needed] Epoch-level STFT (compute once, then slice)     |
|   [Avoid by default]  Wavelet per blink per style (slow & low-Fs)       |
+-----------------------------------------------------------------------+
|
+--------------------------+---------------------------+
|                          |                           |
V                          V                           V
+----------------------+  +---------------------------+  +------------------------------+
| C2a. No epoch TF map |  | C2b. STFT once per epoch   |  | C2c. (Not recommended)       |
| (do Welch per window)|  | - compute TF map for x     |  | wavelet per window per style |
+----------------------+  | - cache TF map             |  +------------------------------+
+---------------------------+

|
|---------------------------------------------------------------------|
|  D) SEGMENTATION (SLICE-MANY)                                         |
|---------------------------------------------------------------------|
V
+-----------------------------------------------------------------------+
| D1. Canonical window per blink (optional but recommended)              |
|   - define padded window indices per blink (e.g., onset-0.5s to off+0.5s)|
|   - ensures consistent feature context                                 |
+-----------------------------------------------------------------------+
|
V
+-----------------------------------------------------------------------+
| D2. For each row in BlinkBoundaryTable (epoch_id, style, modality):     |
|   - fetch cached traces for that epoch_id+modality                      |
|   - slice style-specific window [onset_samp : offset_samp]              |
|   - (optional) slice canonical padded segment and then sub-slice        |
+-----------------------------------------------------------------------+

|
|---------------------------------------------------------------------|
|  E) FEATURE EXTRACTION (WINDOW REDUCTIONS; STYLE IS JUST A LABEL)     |
|---------------------------------------------------------------------|
V
+-----------------------------------------------------------------------+
| E0. Input to feature modules                                            |
|   - time-domain slices: x, x_smooth                                     |
|   - kinematics: dx (or dx_up)                                           |
|   - optional: TF map slices (STFT bins) or Welch PSD per window         |
+-----------------------------------------------------------------------+
|
+---------------------------+---------------------------+---------------------------+
|                           |                           |                           |
V                           V                           V                           V
+---------------------+   +------------------------+  +------------------------+  +------------------------+
| E1. Morphology       |  | E2. Kinematics         |  | E3. Energy             |  | E4. Frequency Domain   |
| (x / x_smooth)       |  | (dx or dx_up)          |  | (x, envelope, TKEO)    |  | (Welch PSD or STFT)    |
| - amplitude p2p      |  | - max closing slope     |  | - avg energy           |  | - bandpower 0-2/2-5/...|
| - AUC                |  | - max opening slope     |  | - Teager energy        |  | - spectral entropy     |
| - FWHM (better up)   |  | - slope timing          |  | - wavelet energy*      |  | - peak frequency       |
+---------------------+   +------------------------+  +------------------------+  +------------------------+
*only if you kept wavelet; otherwise drop/replace with RMS energy

|
V
+-----------------------------------------------------------------------+
| E5. Open-eye metrics (computed efficiently)                             |
|   Strategy: compute ONCE per epoch per modality, not per style           |
|   - define "open" segments using a reference style (e.g., base) OR      |
|     union of all blink windows across styles (conservative)              |
|   Metrics: inter-blink interval, openness consistency, baseline drift    |
+-----------------------------------------------------------------------+

|
|---------------------------------------------------------------------|
|  F) STYLE COMPARISON (EXPERIMENTATION OUTPUTS)                        |
|---------------------------------------------------------------------|
V
+-----------------------------------------------------------------------+
| F1. Style-delta features (cheap)                                        |
|   - onset_shift(style vs base)                                          |
|   - duration_diff(style vs base)                                        |
|   - feature_ratio(style vs base)                                        |
|   These directly quantify boundary effects across styles                 |
+-----------------------------------------------------------------------+

|
|---------------------------------------------------------------------|
|  G) OUTPUT ASSEMBLY                                                   |
|---------------------------------------------------------------------|
V
+-----------------------------------------------------------------------+
| G1. Feature naming / indexing                                           |
|     <modality>__<style>__<category>__<metric>                            |
+-----------------------------------------------------------------------+
|
V
+-----------------------------------------------------------------------+
| G2. Export / Save                                                       |
|   - dataframe / parquet / csv                                           |
|   - plus QA report: invalid boundaries, window length stats, etc.        |
+-----------------------------------------------------------------------+

END

```
