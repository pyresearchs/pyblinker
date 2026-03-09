# Test Data Inventory and Usage

This document maps the files located in `test/test_files/` to the Python scripts, tutorials, and unit tests that consume them. It serves as a guide for understanding the purpose of each test artifact.

## Recording Files

### `ear_eog_raw.fif`
*   **Description**: A short raw recording (MNE FIF format) containing both EAR (Eye Aspect Ratio) and EEG/EOG channels, with manual blink annotations.
*   **Primary Use**: The main dataset for tutorials and feature extraction tests. Tests that require an *unannotated* recording clear annotations programmatically to mirror the previous `ear_eog_without_annotation_raw.fif` fixture.
*   **Used By**:
    *   `test/epoch_blink_finder/test_blink_finder.py`
    *   `test/epoch_blink_finder/test_blink_finder_drop.py`
    *   `test/epoch_blink_finder/test_blink_report.py`
    *   `tutorial/05b_eeg_feature_extraction_tutorial.py`
    *   `tutorial/05c_minimal_blink_feature_tutorial.py`
    *   `tutorial/04_epoching_and_blink_validation_report.py`
    *   `test/blink_features/*` (Most feature aggregation tests)
    *   `test/utils/test_slice_raw_into_mne_epochs.py`

### `mne_sample_audvis_raw.edf`
*   **Description**: An EDF file exported from the standard MNE sample dataset.
*   **Primary Use**: Cross-validation with MATLAB (which natively handles EDF) and testing format compatibility.
*   **Used By**:
    *   `tutorial/03d_understand_diff_in_blink_position.py`
    *   `test/migration_files/matlab_code/` (MATLAB processing scripts)
    *   `test/data_setup.py` (Generates this file)

## Metadata and Ground Truth

### `ear_eog_blink_count_epoch.csv`
*   **Description**: A CSV file listing the expected number of blinks per epoch for the `ear_eog_raw.fif` recording.
*   **Primary Use**: Ground-truth assertion for testing the accuracy of blink mapping to epochs.
*   **Used By**:
    *   `test/blink_features/blink_events/test_blink_count.py`
    *   `test/blink_features/blink_events/test_aggregate_event_features.py`
    *   `test/epoch_blink_finder/test_blink_finder.py`
    *   `tutorial/05b_eeg_feature_extraction_tutorial.py`

### `ear_eog.csv`
*   **Description**: Manual blink annotations (onset, duration, description) aligned to `ear_eog_raw.fif`.
*   **Primary Use**: Attaching coarse annotations before refinement and feature extraction.
*   **Used By**:
    *   `test/blink_feature_ear/energy/test_energy_features.py`
    *   `test/segmentation/test_ear_refinement_outputs.py`
    *   `test/test_ear_threshold_refinement.py`
    *   `test/test_refined_blink_flow.py`
    *   `tutorial/03a_ear_threshold_blink_refinement.py`
    *   `tutorial/03c_ear_threshold_multi_candidate_refinement.py`
    *   `tutorial/05a_ear_energy_feature_tutorial.py`
    *   `tutorial/06_refined_blink_report_tutorial.py`

### `ear_metadata_threshold_interpolation.fif`
*   **Description**: Reference epochs produced from `ear_eog_raw.fif` using threshold interpolation; includes metadata for regression checks.
*   **Primary Use**: Ensuring the segmentation pipeline yields stable epoch structures and metadata.
*   **Used By**:
    *   `test/segmentation/test_ear_refinement_outputs.py`

### `ear_multi_threshold_refined_blinks.csv`
*   **Description**: Reference table of refined EAR blinks across multiple candidate thresholds.
*   **Primary Use**: Validating EAR threshold sweeps and downstream feature calculations.
*   **Used By**:
    *   `test/segmentation/test_ear_refinement_outputs.py`
    *   `tutorial/03c_ear_threshold_multi_candidate_refinement.py`

## Intermediate Data (Pickles & Numpy)

### `S1_candidate_signal.npy`
*   **Description**: A 1D NumPy array representing a pre-extracted "candidate signal" (e.g., component activity) for blink detection.
*   **Primary Use**: Unit testing the low-level detection and fitting logic without loading a full MNE Raw object.
*   **Used By**:
    *   `test/blink_features/pyblinker/test_blink_features.py`
    *   `test/blinker_migration/test_blink_features.py`

### `blink_properties_fits.pkl`
*   **Description**: A Pickle file containing pre-computed blink metadata, including landmarks and linear fits.
*   **Primary Use**: Regression testing for the `blink_features` module, ensuring that property calculations remain stable.
*   **Used By**:
    *   `test/blink_features/pyblinker/test_blink_properties.py`
    *   `test/blinker_migration/test_blink_properties.py`

### `file_test_blink_position.pkl`
*   **Description**: Pickle file storing expected blink positions for regression testing.
*   **Used By**:
    *   `test/blink_features/pyblinker/utils/update_pkl_variables.py`

### `file_test_epoch_full_pipeline.pkl`
*   **Description**: Pickle file storing state for full pipeline epoch tests.
*   **Used By**:
    *   `test/data_setup.py` (Registry)

### `data_for_selecting_best_channels.pkl`
*   **Description**: Data used to validate the automatic channel selection logic.
*   **Used By**:
    *   `test/data_setup.py` (Registry)

## Relocation and cleanup

*   **Change**: Consolidated all EAR/EOG sample assets under `test/test_files/` and removed duplicate FIF recordings.
*   **Related Code**:
    *   `test/data_setup.py` (test asset registry)
    *   `pyblinker/outside_annotation/cli.py` (sample CLI defaults)
    *   `tutorial/03a_ear_threshold_blink_refinement.py`, `tutorial/03c_ear_threshold_multi_candidate_refinement.py`, `tutorial/05a_ear_energy_feature_tutorial.py`, `tutorial/06_refined_blink_report_tutorial.py`
*   **Verification**:
    *   Unit tests: `test/epoch_blink_finder/test_blink_finder.py`, `test/segmentation/test_ear_refinement_outputs.py`, `test/test_ear_threshold_refinement.py`
    *   Tutorials: EAR refinement tutorials listed above load assets from `test/test_files/`
