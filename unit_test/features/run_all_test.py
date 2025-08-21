"""Run all unit tests under :mod:`unit_test.features`."""

from __future__ import annotations

import logging
import unittest
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent


def main() -> None:
    """Discover and execute the feature unit tests."""
    logger.info("Discovering feature tests in %s", ROOT)
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(ROOT),
        pattern="test_*.py",
        top_level_dir=str(REPO_ROOT),
    )
    logger.info("Running feature test suite")
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    main()
