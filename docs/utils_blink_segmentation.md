# Blink Segmentation Strategies

This document explains the different ways to define blink start and end points, and how this affects blink metric calculation in `pyblinker`.

## Overview

Accurate definition of blink onset (start) and offset (end) is crucial for calculating consistent and meaningful blink features. Different modalities (EEG, EOG, EAR) and different research questions may require different segmentation strategies.

In `pyblinker`, feature calculation is generally based on the `blink_onset` and `blink_duration` provided in the metadata. This gives users the flexibility to define these parameters according to their specific needs.

## Default Assumptions

*   **EEG and EOG Data:** Typically, the blink start and end are defined by the **interpolated zero-crossing points** (or points where the signal crosses a baseline/threshold associated with the blink event).
*   **EAR (Eye Aspect Ratio) Data:** The blink start and end are typically defined by the **threshold crossing points** (where the EAR drops below a certain open-eye threshold).

However, these are just conventions. You might want to apply the "threshold crossing" logic to EEG/EOG or the "zero crossing" logic to EAR, depending on your preprocessing and signal characteristics.

## Segmentation Strategies

`pyblinker` supports several strategies for defining the blink interval. These can be specified when refining annotations or processing epochs.

*   **`"base"`**: Uses the "outer" or widest possible definition of the blink base. This might correspond to the deviation from the baseline before the main rise/fall.
*   **`"zero"`**: Uses the zero-crossing (or threshold-crossing) points. This is the most common definition for blink duration (half-amplitude duration is another, but zero-crossing is standard for the full event duration in many contexts).
*   **`"tent"`**: (Description of tent strategy - typically involving peak and slopes).
*   **`"half_base"`**: Uses the points at half the height of the blink amplitude relative to the base.
*   **`"half_zero"`**: Uses the points at half the height of the blink amplitude relative to the zero-crossing/threshold.

* However, in some case, we may start with only one definition (e.g., only zero-crossing points), and perhaps, along the way in the pipeline
    we want to compute other definitions (e.g., base points) for more comprehensive analysis.
## Impact on Logic Implementation

The function `pyblinker.utils.refinement_utils.slice_raw_into_mne_epochs_refine_annot` is responsible for slicing raw data into epochs and refining the blink metadata.

To support flexible definitions:
1.  Users can specify a `segmentation_strategy` (e.g., `"zero"`, `"base"`).
2.  The function computes the appropriate `blink_onset` and `blink_duration` based on this strategy.
3.  Downstream feature calculation functions use these computed onsets/durations, remaining agnostic to the specific definition used.

This ensures that whether you are analyzing EEG, EOG, or video-based EAR signals, you can align the feature extraction window precisely with your event definition.
