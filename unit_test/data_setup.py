"""Utilities for downloading test datasets."""
from __future__ import annotations

import logging
import shutil
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
            if member.is_dir():
                continue
            destination = target_dir / Path(member.filename).name
            with zf.open(member) as src, open(destination, "wb") as dst:
                shutil.copyfileobj(src, dst)
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
