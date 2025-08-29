import pytest
import os

if __name__ == '__main__':
    # Base directory is the folder where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Update paths to point to the pyblinker subdirectory
    test_files = [
        os.path.join(base_dir, 'pyblinker', 'test_blink_features.py'),
        os.path.join(base_dir, 'pyblinker', 'test_blink_properties.py')
    ]

    # Run only those test files
    exit_code = pytest.main(test_files)

    exit(exit_code)
