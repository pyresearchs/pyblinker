"""Convenience script to run all unit tests.

This file discovers and executes every module in ``unit_test`` matching the
``test_*.py`` pattern. It is mainly intended for manual debugging during
development.
"""

from pathlib import Path
import unittest
import multiprocessing
import sys

# Ensure the repository root is on the Python path so that imports
# like ``pyblinker`` succeed when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Discover and load all tests in the 'tests' directory
test_loader = unittest.TestLoader()
test_suite = test_loader.discover('unit_test', pattern='test_*.py')
def main() -> None:
    """Execute the discovered test suite."""
    multiprocessing.set_start_method("spawn", force=True)
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_runner.run(test_suite)


if __name__ == "__main__":
    main()
