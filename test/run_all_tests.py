# """Run all unit tests under the test directory."""
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
# ROOT = Path(__file__).resolve().parent
# REPO_ROOT = ROOT.parent
#
# def main() -> None:
#     """Discover and execute all unit tests."""
#     logger.info("Discovering tests in %s", ROOT)
#
#     # Ensure the project root is in sys.path
#     if str(REPO_ROOT) not in sys.path:
#         sys.path.insert(0, str(REPO_ROOT))
#
#     loader = unittest.TestLoader()
#     try:
#         suite = loader.discover(
#             start_dir=str(ROOT),
#             pattern="test_*.py",
#             top_level_dir=str(REPO_ROOT),
#         )
#     except Exception as e:
#         logger.error(f"Failed to discover tests: {e}")
#         sys.exit(1)
#
#     logger.info("Running test suite")
#     runner = unittest.TextTestRunner(verbosity=2)
#     result = runner.run(suite)
#
#     if not result.wasSuccessful():
#         sys.exit(1)
#
# if __name__ == "__main__":
#     main()
