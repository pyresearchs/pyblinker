# Reporting and Quality Control

## Purpose
Automated algorithms are never perfect. `pyblinker` integrates a strong "Quality Control (QC)" philosophy, enabling users to visually verify every stage of processing. HTML reports are the primary artifact for this verification.

## Report Content
A typical `pyblinker` report (MNE-based) includes:

*   **Signal Overlays**: The raw signal (EEG/EOG/EAR) with detected events marked.
*   **Blink Markers**: Vertical lines indicating the refined start (green), peak (red), and end (blue) of each blink.
*   **Epoch Summaries**: Views grouped by epoch, allowing users to quickly scan for missed blinks or false positives.
*   **Feature Distributions**: (Optional) Histograms or scatter plots of computed features.

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
