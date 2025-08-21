"""Run all unit tests for migration and feature suites."""
from __future__ import annotations

import logging
import multiprocessing
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from unit_test import download_migration_files, download_test_files

logger = logging.getLogger(__name__)


def main() -> None:
    """Download datasets and execute all tests under :mod:`unit_test`."""
    download_migration_files()
    download_test_files()
    multiprocessing.set_start_method("spawn", force=True)
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    main()
