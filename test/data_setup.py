"""Utilities for downloading test datasets."""
from __future__ import annotations

import logging
import tempfile
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

MIGRATION_DATA_ID = "1p0x4L37d_jkX6B5e8BF5c2MOR3iC8Kb6"
MIGRATION_DATA_URL = f"https://drive.google.com/uc?id={MIGRATION_DATA_ID}"
MIGRATION_DIR = Path(__file__).resolve().parent / "migration_files"
MIGRATION_REQUIRED_FILES = [
    MIGRATION_DIR / "step1bi_data_input_getBlinkPositions.mat",
    MIGRATION_DIR / "step1bi_data_output_getBlinkPositions.mat",
]

TEST_FILE_DATA_ID = "1gOSPGjCEM5aA3K3QL0OjJex2uqSa0PyI"
TEST_FILE_URL = f"https://drive.google.com/uc?id={TEST_FILE_DATA_ID}"
TEST_FILES_DIR = Path(__file__).resolve().parent / "test_files"
TEST_REQUIRED_FILES = [
    TEST_FILES_DIR / "blink_properties_fits.pkl",
    TEST_FILES_DIR / "data_for_selecting_best_channels.pkl",
    TEST_FILES_DIR / "ear_eog_blink_count_epoch.csv",
    TEST_FILES_DIR / "ear_eog_raw.fif",
    TEST_FILES_DIR / "ear_eog_without_annotation_raw.fif",
    TEST_FILES_DIR / "file_test_blink_position.pkl",
    TEST_FILES_DIR / "file_test_epoch_full_pipeline.pkl",
    TEST_FILES_DIR / "S1_candidate_signal.npy",
]


def _download_and_extract(url: str, target_dir: Path, tmp_name: str) -> None:
    """Download a zip from Google Drive and extract into ``target_dir``.

    Args:
        url: Direct download URL to the zip file.
        target_dir: Destination directory for extracted files.
        tmp_name: Name for the temporary zip file.

    Raises:
        ImportError: If ``gdown`` is not available.
    """
    try:
        import gdown  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency missing
        raise ImportError("gdown is required to download test data") from exc

    tmp_zip = Path(tempfile.gettempdir()) / tmp_name
    logger.info("Downloading test data from %s", url)
    gdown.download(url, str(tmp_zip), quiet=False)

    logger.info("Extracting files to %s", target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        for member in zf.infolist():
            # Some zips don't store directory entries; guard via filename suffix
            if member.filename.endswith('/'):
                continue
            destination = target_dir / Path(member.filename).name
            with zf.open(member) as src:
                data = src.read()
            destination.write_bytes(data)
    tmp_zip.unlink(missing_ok=True)


def download_migration_files() -> None:
    """Download migration test ``.mat`` files if missing."""
    if all(path.exists() for path in MIGRATION_REQUIRED_FILES):
        return

    _download_and_extract(MIGRATION_DATA_URL, MIGRATION_DIR, "migration_files.zip")


def download_test_files() -> None:
    """Download additional test files if any are missing."""

    if all(path.exists() for path in TEST_REQUIRED_FILES):
        return

    _download_and_extract(TEST_FILE_URL, TEST_FILES_DIR, "test_files.zip")


# --- New helper: create an EDF from MNE sample data for MATLAB comparison ---

def ensure_mne_sample_edf(target_dir: Path | None = None, overwrite: bool = False) -> Path:
    """Ensure an EDF made from the MNE sample dataset exists in test files.

    This downloads (via MNE) the public "sample" dataset if needed, reads the
    filtered raw FIFF file, and exports it to EDF so it can be used to compare
    against the MATLAB version of the blinker package.

    Args:
        target_dir: Directory to place the EDF file. Defaults to TEST_FILES_DIR.
        overwrite: If True, overwrite an existing EDF.

    Returns:
        Path to the created (or existing) EDF file.

    Notes:
        - Requires the ``mne`` package and an active internet connection on the
          first run to fetch the sample dataset.
        - We only export a representative raw file from the dataset.
    """
    try:
        import mne
    except Exception as exc:  # pragma: no cover - optional path
        raise ImportError("mne is required to create the sample EDF file") from exc

    if target_dir is None:
        target_dir = TEST_FILES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    edf_path = target_dir / "mne_sample_audvis_raw.edf"
    if edf_path.exists() and not overwrite:
        logger.info("EDF already exists at %s", edf_path)
        return edf_path

    # Fetch the sample dataset and take a commonly used raw file
    logger.info("Fetching MNE sample dataset (this may take a moment on first run)…")
    sample_data_folder = mne.datasets.sample.data_path(verbose=True)
    fif_path = Path(sample_data_folder) / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"

    logger.info("Reading raw FIF: %s", fif_path)
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)

    # Apply projectors for a cleaner export
    try:
        if raw.proj:
            raw.apply_proj()
    except Exception:
        # Non-fatal; continue export
        pass

    # Export to EDF; prefer Raw.export when available, fall back to mne.export
    raw.pick_types(eeg=True)
    raw.filter(0.5, 20.5, fir_design='firwin')
    raw.resample(100)

    drange=[f'EEG 00{X}' for X in [1,2,3,5,8]]
    # drange=[f'EEG 00{X}' for X in range(10)]
    to_drop_ch = list(set(raw.ch_names) - set(drange))
    if to_drop_ch:
        raw = raw.drop_channels(to_drop_ch)

    logger.info("Exporting EDF to %s", edf_path)
    try:
        # MNE >= 1.2
        raw.export(str(edf_path), fmt="edf")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - compatibility path
        try:
            # Older MNE versions
            from mne.export import export_raw  # type: ignore

            export_raw(str(edf_path), raw, fmt="edf")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Failed to export EDF using MNE") from exc

    logger.info("EDF created at %s", edf_path)
    return edf_path


if __name__ == "__main__":
    # Lightweight CLI for local use
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Test data setup utilities")
    parser.add_argument(
        "--make-edf",
        action="store_true",
        help="Download MNE sample data (if needed) and export an EDF into test/test_files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing EDF if present",
    )
    args = parser.parse_args()

    if args.make_edf:
        path = ensure_mne_sample_edf(overwrite=args.overwrite)
        print(f"EDF available at: {path}")
    else:
        parser.print_help()
