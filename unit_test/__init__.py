"""Unit test utilities and dataset setup."""
import logging
from pathlib import Path
import tempfile
import zipfile

logger = logging.getLogger(__name__)

_DATA_ID = "1p0x4L37d_jkX6B5e8BF5c2MOR3iC8Kb6"
_DATA_URL = f"https://drive.google.com/uc?id={_DATA_ID}"
_MIGRATION_DIR = Path(__file__).resolve().parent / "migration_files"
_REQUIRED_FILES = [
    _MIGRATION_DIR / "step1bi_data_input_getBlinkPositions.mat",
    _MIGRATION_DIR / "step1bi_data_output_getBlinkPositions.mat",
]


def _download_migration_files() -> None:
    """Download test `.mat` files if they are missing."""
    if all(path.exists() for path in _REQUIRED_FILES):
        return

    try:
        import gdown  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency missing
        raise ImportError("gdown is required to download migration test files") from exc

    tmp_zip = Path(tempfile.gettempdir()) / "migration_files.zip"
    logger.info("Downloading migration test data from %s", _DATA_URL)
    gdown.download(_DATA_URL, str(tmp_zip), quiet=False)

    logger.info("Extracting migration files to %s", _MIGRATION_DIR)
    _MIGRATION_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        zf.extractall(_MIGRATION_DIR.parent)
    tmp_zip.unlink(missing_ok=True)


_download_migration_files()
