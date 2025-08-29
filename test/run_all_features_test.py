"""Run only feature unit tests."""
from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path
import sys
import unittest

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))

FEATURE_DIR = ROOT / "features"


def main() -> None:
    """Download datasets and run tests in :mod:`test.features`."""
    from test import download_migration_files, download_test_files

    download_migration_files()
    download_test_files()
    multiprocessing.set_start_method("spawn", force=True)
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(FEATURE_DIR),
        pattern="test_*.py",
        top_level_dir=str(ROOT.parent),
    )
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    main()
