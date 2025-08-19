"""Run only migration-related unit tests."""
import multiprocessing
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
MIGRATION_DIR = ROOT / "blinker_migration"
sys.path.insert(0, str(ROOT.parent))

def main() -> None:
    """Discover and run tests in :mod:`unit_test.blinker_migration`."""
    multiprocessing.set_start_method("spawn", force=True)
    loader = unittest.TestLoader()
    suite = loader.discover(str(MIGRATION_DIR), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    main()
