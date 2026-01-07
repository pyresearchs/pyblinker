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
