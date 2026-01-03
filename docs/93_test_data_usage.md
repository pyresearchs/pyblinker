# Test Data Inventory and Usage

This document maps the files located in `test/test_files/` to the Python scripts, tutorials, and unit tests that consume them. It serves as a guide for understanding the purpose of each test artifact.

## Recording Files

### `ear_eog_raw.fif`
*   **Description**: A short raw recording (MNE FIF format) containing both EAR (Eye Aspect Ratio) and EEG/EOG channels, with manual blink annotations.
*   **Primary Use**: The main dataset for tutorials and feature extraction tests.
*   **Used By**:
    *   `tutorial/05b_eeg_feature_extraction_tutorial.py`
    *   `tutorial/05c_minimal_blink_feature_tutorial.py`
    *   `tutorial/04_epoching_and_blink_validation_report.py`
    *   `test/blink_features/*` (Most feature aggregation tests)
    *   `test/utils/test_slice_raw_into_mne_epochs.py`

### `ear_eog_without_annotation_raw.fif`
*   **Description**: Same as `ear_eog_raw.fif` but stripped of annotations.
*   **Primary Use**: Validating *automated* blink detection workflows where the system must find blinks from scratch without existing ground truth.
*   **Used By**:
    *   `test/epoch_blink_finder/test_blink_finder.py`
    *   `test/epoch_blink_finder/test_blink_finder_drop.py`
    *   `test/epoch_blink_finder/test_blink_report.py`

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
