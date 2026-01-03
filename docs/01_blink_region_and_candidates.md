# Blink Region Identification and Candidates

## Purpose
Before precise blink metrics can be calculated, the system must identify "candidate regions" where a blink likely occurred. This step focuses on finding the rough temporal window of a blink event.

## Candidate Signals
Blinks can be detected from various signal modalities:

*   **EOG (Electrooculography)**: The most direct measure of eye movements. Vertical EOG (VEOG) typically shows large, clear deflections during blinks.
*   **EEG (Electroencephalography)**: Frontal channels (e.g., Fp1, Fp2) pick up strong blink artifacts.
*   **EAR (Eye Aspect Ratio)**: Derived from video/camera frames. EAR drops significantly when the eye closes.
*   **ICA Components**: Independent Component Analysis can isolate blink artifacts into a single component, which serves as a clean candidate signal for detection.

## Detection Methods

### 1. Automated Thresholding
The primary automated method in `pyblinker` follows the legacy BLINKER approach:
1.  Compute the **Robust Standard Deviation** (using Median Absolute Deviation - MAD) of the candidate signal.
2.  Set a threshold (typically `mean + N * robust_std`).
3.  Identify regions where the signal exceeds this threshold for a minimum duration.
4.  Merge events that are too close together.

### 2. Manual Annotation
Users can manually annotate blinks using:
*   **MNE Visualization Tools**: `raw.plot()` allows interactive annotation.
*   **Custom Viewers**: Tools that display aligned video (EAR) and signal traces.

**Required Annotation Fields:**
If importing external annotations, `pyblinker` expects MNE-compatible annotations with:
*   **Onset**: Start time of the blink event (seconds).
*   **Duration**: Duration of the blink event (seconds).
*   **Description**: A label (e.g., `"blink"`, `"BAD_blink"`).

### Flowchart

```mermaid
graph TD
    A[Candidate Signal] --> B{Detection Method?}
    B -- Automated --> C[Calculate Robust Std Dev]
    C --> D[Apply Thresholding]
    D --> E[Merge Proximal Events]
    B -- Manual --> F[User Annotation (MNE/CSV)]
    E --> G[Candidate Regions]
    F --> G
```

## Related Code

*   **`pyblinker/blinker/get_blink_positions.py`**: Implements the automated detection logic (MAD calculation, thresholding, and merging).
*   **`pyblinker/blinker/fit_blink.py`**: Handles the fitting of blink shapes to the detected candidate regions.
*   **`pyblinker/utils/annotation_utils.py`**: Utilities for converting between different annotation formats and MNE structures.
