# Reporting and Quality Control

## Purpose
Automated algorithms are never perfect. `pyblinker` integrates a strong "Quality Control (QC)" philosophy, enabling users to visually verify every stage of processing. HTML reports are the primary artifact for this verification.

## Report Content
A typical `pyblinker` report (MNE-based) includes:

*   **Signal Overlays**: The raw signal (EEG/EOG/EAR) with detected events marked.
*   **Blink Markers**: Vertical lines indicating the refined start (green), peak (red), and end (blue) of each blink.
*   **Epoch Summaries**: Views grouped by epoch, allowing users to quickly scan for missed blinks or false positives.
*   **Feature Distributions**: (Optional) Histograms or scatter plots of computed features.

### EAR refined report field alignment
* **Feature/Change**: The refined blink report now derives threshold crossings and trough markers from the updated EAR timing fields (`onset__refine__ear`, `duration__refine__ear`, `onset__th_interpolation__ear`, `duration__th_interpolation__ear`, `trough__th_point__ear`) instead of legacy sample columns. If those fields are missing, the report helper derives them from refined sample indices when possible.
* **Related Code**: `pyblinker/outside_annotation/reporting_flow.py` now also derives interpolated threshold timing from `left_interpolated_threshold_sample`/`right_interpolated_threshold_sample` when needed.
* **Related Code**: `pyblinker/outside_annotation/reporting_flow.py`, `pyblinker/viz/ear_report.py`.
* **Verification (Tutorials & Tests)**: `tutorial/03c_ear_threshold_multi_candidate_refinement.py`, `tutorial/06_refined_blink_report_tutorial.py`, `test/epoch_blink_finder/test_blink_report.py`.
* **Feature/Change**: Reports can now accept an `mne.Epochs` object directly and will use epoch metadata plus per-epoch signals to place landmarks.
* **Related Code**: `pyblinker/outside_annotation/reporting_flow.py`.
* **Verification (Tutorials & Tests)**: `tutorial/03c_ear_threshold_multi_candidate_refinement.py`, `test/epoch_blink_finder/test_blink_report.py`.
* **Feature/Change**: Refined blink reports now consistently use `mne.Epochs` metadata/signals, require `onset__refine__ear` and `duration__refine__ear`, and gate interpolated threshold markers behind the `mark_threshold_crossings` flag to avoid unexpected overlays. Missing interpolated sample columns are called out in figure captions.
* **Related Code**: `pyblinker/outside_annotation/reporting_flow.py`.
* **Verification (Tutorials & Tests)**: `tutorial/03a_ear_threshold_blink_refinement.py`, `tutorial/03c_ear_threshold_multi_candidate_refinement.py`, `tutorial/06_refined_blink_report_tutorial.py`.

### EEG epoch landmark plotting
* **Feature/Change**: The EEG epoch landmark tutorial now renders EEG traces as low-opacity lines with scatter points, includes a horizontal zero-reference line, pads each blink window to include the last available EEG landmark before plotting, and annotates the caption when any landmark values are missing. Landmarks still come from exploded `epochs.metadata` fields like `start__left_base__eeg` and `x_intersect__eeg`.
* **Related Code**: `tutorial/03d_eeg_plotting_all_landmark_epochs.py`.
* **Verification (Tutorials & Tests)**: `tutorial/03d_eeg_plotting_all_landmark_epochs.py`, `test/segmentation/test_eeg_refinement_outputs.py`.
* **Feature/Change**: EEG epoch landmark plots now draw landmark markers behind the EEG scatter points so the EEG-E8 dots sit on top without changing marker sizes.
* **Related Code**: `tutorial/03d_eeg_plotting_all_landmark_epochs.py`.
* **Verification (Tutorials & Tests)**: `tutorial/03d_eeg_plotting_all_landmark_epochs.py`, `test/segmentation/test_eeg_refinement_outputs.py`.

### MATLAB FitBlinks landmark reports
* **Feature/Change**: Added a dedicated HTML report generator that renders per-blink FitBlinks landmarks from MATLAB output alongside Python-derived outputs using the same MATLAB input signal and blink-selection logic as the FitBlinks comparison script. Each blink is rendered with a two-column layout (plot + legend) and a caption that enumerates every FitBlinks landmark value.
* **Related Code**: `pyblinker/outside_annotation/matlab_fitblink_report.py`.
* **Verification (Tutorials & Tests)**: `test/outside_annotation/test_matlab_fitblink_report.py`.

## When to Generate Reports
Reports can be generated at multiple stages:

1.  **After Detection**: To verify that candidate regions cover the visible blinks.
2.  **After Refinement**: To check if the start/end points land on the correct signal landmarks (e.g., zero-crossings).
3.  **After Epoching**: To review which epochs contain blinks and which were rejected.

## Reproducibility
Reports are saved as standalone HTML files. These files serve as a permanent record of the preprocessing state, ensuring that the analysis can be audited later.

### Flowchart

```mermaid
graph TD
    A[Processed Epochs / Blinks] --> B[Report Generator]
    B --> C[Generate Signal Plots]
    B --> D[Overlay Blink Markers]
    C --> E[Assemble HTML Report]
    D --> E
    E --> F[Save to Disk (.html)]
    F --> G[User Visual Inspection]
```

## Related Code

*   **`pyblinker/viz/blink_report.py`**: Main utility for generating MNE reports for blink detection.
*   **`pyblinker/viz/ear_report.py`**: Specialized reporting for EAR signals.
*   **`pyblinker/viz/report_utils.py`**: Helper functions for plot generation.

## Tutorials

*   **`tutorial/06_refined_blink_report_tutorial.py`**:
    The main guide for QC. It shows how to generate an HTML report that overlays the *refined* blink boundaries (start/peak/end) on top of the raw signal. This allows the user to visually confirm if the refinement logic is snapping to the correct signal features.

## Unit Tests

*   **`test/epoch_blink_finder/test_blink_report.py`**:
    Verifies the report generation pipeline. It checks that:
    1.  The `MNE.Report` object is created successfully.
    2.  Figures (matplotlib plots) are correctly added to the report.
    3.  The final HTML file can be saved to disk without errors.
    This ensures that the visualization tools remain compatible with the installed version of `matplotlib` and `mne`.

## Refactor update: deprecated outside-annotation reporting package removal

### The Feature/Change
- Removed the legacy `pyblinker/outside_annotation` package from core source to reduce quarantined/unused maintenance surface as part of major-structure cleanup.

### Related Code
- Removed:
  - `pyblinker/outside_annotation/__init__.py`
  - `pyblinker/outside_annotation/cli.py`
  - `pyblinker/outside_annotation/ear_energy_report.py`
  - `pyblinker/outside_annotation/matlab_fitblink_report.py`
  - `pyblinker/outside_annotation/refined_blink_flow.py`
  - `pyblinker/outside_annotation/reporting_flow.py`

### Verification (Tutorials & Tests)
- Unit tests:
  - `test/run_all_tests.py` (full suite)
