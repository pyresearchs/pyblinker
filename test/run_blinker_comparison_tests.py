# """Run unit tests under the test/blinker_pyblinker_comparison directory."""
#
# import logging
# import unittest
# from pathlib import Path
# import sys
#
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Define paths
# ROOT = Path(__file__).resolve().parent
# COMPARISON_TEST_DIR = ROOT / "blinker_pyblinker_comparison"
# REPO_ROOT = ROOT.parent
#
# def main() -> None:
#     """Discover and execute tests in blinker_pyblinker_comparison."""
#     logger.info("Discovering tests in %s", COMPARISON_TEST_DIR)
#
#     # Ensure the project root is in sys.path
#     if str(REPO_ROOT) not in sys.path:
#         sys.path.insert(0, str(REPO_ROOT))
#
#     loader = unittest.TestLoader()
#     try:
#         suite = loader.discover(
#             start_dir=str(COMPARISON_TEST_DIR),
#             pattern="test_*.py",
#             top_level_dir=str(REPO_ROOT),
#         )
#     except Exception as e:
#         logger.error(f"Failed to discover tests: {e}")
#         sys.exit(1)
#
#     logger.info("Running comparison test suite")
#     runner = unittest.TextTestRunner(verbosity=0)
#     result = runner.run(suite)
#
#     if not result.wasSuccessful():
#         sys.exit(1)
#
# if __name__ == "__main__":
#     main()
