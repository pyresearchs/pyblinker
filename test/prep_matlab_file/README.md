# MATLAB File Preparation for Testing

This directory contains utilities for generating intermediate MATLAB (`.mat`) files used to cross-verify the Python implementation against legacy MATLAB results.

## Scripts

### `convert_input_fif_to_edf.py`

**Note:** Despite the historical filename, this script is now a **utility module** (library) and does not run as a standalone command-line tool.

*   **Purpose:** Converts a raw FIF file into a specific single-channel MATLAB `.mat` format.
*   **Usage:** It exports the function `convert_fif_to_mat`, which is imported and used by setup scripts.
*   **Output:** A `.mat` file containing the signal data (`blinkComp`), sampling rate, and metadata.

### `setup_resampled_files.py`

*   **Purpose:** The main entry point for generating required resampled test artifacts.
*   **Location:** `test/prep_matlab_file/setup_resampled_files.py`

## Test Data Generation Workflow

To ensure reproducible tests, we generate resampled (20 Hz) versions of the raw test data in FIF and MAT formats.

Running the setup script ensures the following chain of artifacts exists in `test/test_files/`:

1.  **Source:** `ear_eog_raw.fif` (Original)
2.  **Resampled FIF:** `ear_eog_resamp-20_raw.fif` (Derived from Source, resampled to 20Hz)
3.  **Intermediate MAT:** `ear_eog_resamp-20_raw.mat` (Derived from Resampled FIF using `convert_fif_to_mat`)

### How to Generate Files

If the test files are missing, run the following command from the project root:

```bash
python -m test.prep_matlab_file.setup_resampled_files
```

This command is idempotent: it checks for the existence of files and only generates them if they are missing.
