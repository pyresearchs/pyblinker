"""
Setup utility for generating resampled test artifacts (FIF, MAT).

This script is responsible for preparing specific test data files required for
validating the blinker pipeline, specifically regarding migration and
replication of MATLAB-based results.

## Overview
The testing suite requires input data to be available in multiple formats (FIF, MAT)
and specifically sampled at 20 Hz to match legacy processing pipelines. This script
automates the creation of these derived files from the source raw data.

## Workflow
The script performs the following idempotent checks and operations:

1.  **Resampled FIF Generation (20 Hz)**
    -   **Check:** Does `test/test_files/ear_eog_resamp-20_raw.fif` exist?
    -   **Action:** If missing, loads the source file `test/test_files/ear_eog_raw.fif`,
        resamples it to 20 Hz, and saves the new FIF file.

2.  **Intermediate MATLAB (.mat) Generation**
    -   **Check:** Does `test/test_files/ear_eog_resamp-20_raw.mat` exist?
    -   **Action:** If missing, uses the `convert_fif_to_mat` utility (from sibling module
        `convert_input_fif_to_edf`) to convert the *resampled FIF* into a MATLAB file
        containing the single channel 'EEG-E8'.

## Directory Structure Assumptions
-   **Source Data:** `test/test_files/ear_eog_raw.fif`
-   **Output Directory:** `test/test_files/`
-   **Script Location:** `test/prep_matlab_file/`

## Usage
Run this script via the command line from the project root:
    
    python -m test.prep_matlab_file.setup_resampled_files

This ensures that imports are resolved correctly within the package structure.
"""

import logging
import sys
from pathlib import Path

import mne

# Import the sibling utility function for FIF -> MAT conversion.
# We attempt an absolute import first (standard for 'python -m ...'),
# falling back to a path modification if run directly as a script.
try:
    from test.prep_matlab_file.convert_input_fif_to_edf import convert_fif_to_mat
except ImportError:
    # Fallback: Add project root to sys.path to allow absolute imports
    current_file = Path(__file__).resolve()
    # Path is: <root>/test/prep_matlab_file/setup_resampled_files.py
    project_root = current_file.parents[2] 
    sys.path.append(str(project_root))
    from test.prep_matlab_file.convert_input_fif_to_edf import convert_fif_to_mat

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Define paths relative to this script location
# Current dir: .../test/prep_matlab_file/
# Test files dir: .../test/test_files/
CURRENT_DIR = Path(__file__).resolve().parent
TEST_FILES_DIR = CURRENT_DIR.parent / "test_files"

SOURCE_FIF = TEST_FILES_DIR / "ear_eog_raw.fif"
RESAMPLED_FIF = TEST_FILES_DIR / "ear_eog_resamp-20_raw.fif"
RESAMPLED_MAT = TEST_FILES_DIR / "ear_eog_resamp-20_raw.mat"


def ensure_resampled_fif(source: Path, target: Path, target_sfreq: int = 20) -> None:
    """
    Ensure the resampled FIF file exists.

    If the target file is missing, this function reads the source FIF,
    resamples it to the specified frequency, and saves the result.

    Parameters
    ----------
    source : Path
        Path to the original raw FIF file.
    target : Path
        Path where the resampled FIF file should be saved.
    target_sfreq : int, optional
        Target sampling frequency in Hz (default is 20).

    Raises
    ------
    FileNotFoundError
        If the source file does not exist.
    """
    if target.exists():
        logger.info("Resampled FIF already exists: %s", target)
        return

    logger.info("Creating resampled FIF: %s -> %s (sfreq=%s)", source, target, target_sfreq)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    raw = mne.io.read_raw_fif(str(source), preload=True, verbose="ERROR")
    
    # Resample
    if int(round(raw.info["sfreq"])) != target_sfreq:
        raw.resample(target_sfreq)
    
    # Save
    raw.save(str(target), overwrite=True, verbose="ERROR")
    logger.info("Created resampled FIF.")


def ensure_mat_from_fif(mat_path: Path, source_fif: Path) -> None:
    """
    Ensure the MAT file exists, creating it from the resampled FIF file if needed.

    Parameters
    ----------
    mat_path : Path
        Path where the .mat file should be saved.
    source_fif : Path
        Path to the resampled FIF file used to generate the MAT file.

    Raises
    ------
    FileNotFoundError
        If the source FIF required to generate the MAT file is missing.
    """
    if mat_path.exists():
        logger.info("MAT file already exists: %s", mat_path)
        return

    logger.info("MAT file missing. Generating from: %s", source_fif)
    if not source_fif.exists():
            raise FileNotFoundError(f"Source FIF for MAT generation not found: {source_fif}")
    
    # Use the utility to create MAT
    convert_fif_to_mat(
        input_fif=str(source_fif),
        output_mat=str(mat_path),
        srate=20, # Explicitly matching the resampled rate
        channel_name="EEG-E8" # Default expected channel
    )
    logger.info("Created MAT file: %s", mat_path)


def main():
    """Execute the full setup workflow."""
    if not TEST_FILES_DIR.exists():
        TEST_FILES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Create resampled FIF
        ensure_resampled_fif(SOURCE_FIF, RESAMPLED_FIF, target_sfreq=20)

        # Step 2: Create MAT from resampled FIF
        ensure_mat_from_fif(RESAMPLED_MAT, RESAMPLED_FIF)
        
        logger.info("Setup complete. All required files are present.")
        
    except Exception as e:
        logger.error("Setup failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()