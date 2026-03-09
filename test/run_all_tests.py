"""Run all unit tests under the ``test`` directory with deterministic progress output."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


class ProgressTextTestResult(unittest.TextTestResult):
    """Print per-module banners and per-test identifiers while running tests."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self._current_module: str | None = None

    def startTest(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        test_id = test.id()
        module_name = test_id.rsplit(".", 2)[0]

        if module_name != self._current_module:
            self._current_module = module_name
            self.stream.writeln(f"\n=== Module: {module_name} ===")

        self.stream.writeln(f"→ Running: {test_id}")
        super().startTest(test)


class ProgressTextTestRunner(unittest.TextTestRunner):
    """Text runner using ``ProgressTextTestResult`` for visible runtime progress."""

    resultclass = ProgressTextTestResult


def _ensure_repo_root_on_path() -> None:
    """Ensure imports are consistent for both direct and discovered test execution."""
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _iter_test_cases(suite: unittest.TestSuite):
    """Yield ``unittest.TestCase`` objects from an arbitrarily nested suite."""
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_test_cases(item)
        else:
            yield item


def _build_discovered_suite() -> unittest.TestSuite:
    """Discover tests from ``test/`` with explicit discovery roots."""
    loader = unittest.TestLoader()
    return loader.discover(
        start_dir=str(ROOT),
        pattern="test_*.py",
        top_level_dir=str(REPO_ROOT),
    )


def _build_deterministic_suite(discovered: unittest.TestSuite) -> unittest.TestSuite:
    """Return a deterministically ordered suite sorted by full test id."""
    sorted_tests = sorted(_iter_test_cases(discovered), key=lambda test: test.id())
    return unittest.TestSuite(sorted_tests)


def main() -> int:
    """Discover and execute all unit tests."""
    _ensure_repo_root_on_path()

    try:
        discovered = _build_discovered_suite()
    except Exception as exc:  # pragma: no cover
        print(f"Failed to discover tests: {exc}", file=sys.stderr)
        return 1

    suite = _build_deterministic_suite(discovered)
    runner = ProgressTextTestRunner(verbosity=1)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
