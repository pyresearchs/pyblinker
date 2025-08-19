"""Run only feature unit tests."""
import multiprocessing
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "features"
sys.path.insert(0, str(ROOT.parent))

def main() -> None:
    """Discover and run tests in :mod:`unit_test.features`."""
    multiprocessing.set_start_method("spawn", force=True)
    loader = unittest.TestLoader()
    suite = loader.discover(str(FEATURE_DIR), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    main()
