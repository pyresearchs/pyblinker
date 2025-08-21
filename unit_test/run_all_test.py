"""Run all unit tests across the project.

This aggregation script mirrors the behaviour of individual runners such as
``run_all_features_test.py``, ``run_all_migration_test.py``,
``run_all_right_base_test.py``, ``features/run_all_test.py`` and
``blinker_migration/run_all_test.py``.
"""
from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path
import sys
import unittest

import pytest

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))
REPO_ROOT = ROOT.parent

# Pytest-based migration tests that are not discoverable by ``unittest``.
PYTEST_FILES = [
    ROOT / "blinker_migration" / "test_blink_features.py",
    ROOT / "blinker_migration" / "test_blink_properties.py",
]


def main() -> None:
    """Download datasets and execute every test under :mod:`unit_test`."""
    from unit_test import download_migration_files, download_test_files

    download_migration_files()
    download_test_files()
    multiprocessing.set_start_method("spawn", force=True)

    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(ROOT),
        pattern="test_*.py",
        top_level_dir=str(REPO_ROOT),
    )
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

    # Execute tests that rely on ``pytest`` features.
    pytest.main([str(path) for path in PYTEST_FILES])


if __name__ == "__main__":
    main()
