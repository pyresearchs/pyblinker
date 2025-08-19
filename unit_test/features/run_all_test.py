"""Utility script to run all unit tests in the :mod:`unit_test.features` directory.

The script is kept at the repository root so that developers can easily run all
tests without invoking ``pytest`` directly. It discovers any files matching
``test_*.py`` recursively under ``unit_test.features`` and executes them using the standard
``unittest`` test runner.
"""

import sys
import unittest
from pathlib import Path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    # Ensure the project root (one level above ``unit_test.features``) is on ``sys.path``
    sys.path.insert(0, str(base_dir.parent))

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(base_dir), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
