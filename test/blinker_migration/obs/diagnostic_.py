import importlib.util
import sys
import unittest
from pathlib import Path

# Ensure root is in the path for imports
ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT))

test_path = Path(__file__).resolve().parent  # This points to test

for py_file in test_path.glob("test_*.py"):
    module_name = py_file.stem  # e.g., test_blink_features
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        num_tests = suite.countTestCases()
        print(f"{py_file.name}: ✅ {num_tests} test(s) found")
    except Exception as e:
        print(f"{py_file.name}: ❌ ERROR loading tests - {e}")
