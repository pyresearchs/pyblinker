# Overview

## Purpose
`pyblinker` is a Python package designed for the automated detection, characterization, and analysis of eye-blink artifacts in diverse biosignals. It supports Electroencephalography (EEG), Electrooculography (EOG), and video-based Eye Aspect Ratio (EAR).

The library is built to replace and extend legacy MATLAB workflows (specifically the BLINKER toolbox) with a modern, MNE-Python-compatible ecosystem. It emphasizes reproducibility, modularity, and an "epoch-based" workflow that aligns with standard event-related potential (ERP) analyses.

## End-to-End Pipeline

The typical workflow in `pyblinker` consists of the following stages:

1.  **Preprocessing**: Load data (MNE raw objects), apply filters, and select candidate channels.
2.  **Blink Detection**: Identify "candidate blink regions" using automated thresholding or import existing annotations.
3.  **Refinement**: Precisely define blink start, peak, and end points using signal-specific strategies (e.g., zero-crossing, threshold interpolation).
4.  **Epoching**: Slice the continuous data into fixed-length epochs (e.g., 30s) or event-locked windows, aligning refined blink metadata with these epochs.
5.  **Feature Extraction**: Compute a rich set of metrics (amplitude, velocity, duration, EAR-specific features) for every blink and aggregate them per epoch.
6.  **Reporting**: Generate HTML reports for visual quality control (QC) of detected blinks and computed features.

### Pipeline Flowchart

```mermaid
graph TD
    A[Raw Data (EEG/EOG/EAR)] --> B[Preprocessing & Candidate Selection]
    B --> C[Blink Region Detection]
    C --> D[Refinement (Start/End Definition)]
    D --> E[Epoching & Metadata Alignment]
    E --> F[Feature Extraction]
    F --> G[Reporting & QC]
```

## Design Goals

*   **MNE Compatibility**: Seamlessly integrates with MNE-Python `Raw` and `Epochs` objects.
*   **Modularity**: Feature extractors are decoupled from detection logic, allowing users to add custom metrics easily.
*   **Reproducibility**: Explicit configuration of segmentation strategies (e.g., "zero-crossing" vs. "threshold") ensures results are consistent.
*   **QC-First**: Every step supports visual reporting to verify the algorithm's performance on individual subjects.

## Related Code

*   **`pyblinker/pipeline.py`**: The main entry point for the feature extraction pipeline. It orchestrates the aggregation of features from different sub-modules.
*   **`pyblinker/pipeline_steps.py`**: Defines high-level steps often used in scripts.
*   **`pyblinker/blinker/pyblinker.py`**: Contains the core logic for the legacy BLINKER-style detection.
