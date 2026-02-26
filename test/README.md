# Test execution guide

This repository uses Python's standard-library `unittest` test runner.

## Recommended commands (from repo root)

### Run one test module (recommended)

Use `python -m unittest` with a dotted module path:

- `python -m unittest test.blink_features.kinematics.test_kinematics_ear_only_config`
- `python -m unittest test.blink_features.kinematics.test_kinematics_eeg_only_config`
- `python -m unittest test.blink_features.morphology.test_epoch_morphology_features_aggregation`

This is the recommended single-file workflow because it is consistent across local and CI environments.

### Run one test module by file path (supported)

You can also pass a file path to unittest:

- `python -m unittest test/blink_features/kinematics/test_kinematics_ear_only_config.py`

### Run all tests in batch

- `python test/run_all_tests.py`

The batch runner discovers tests from `test/` with `top_level_dir` set to the repository root, keeps ordering deterministic by full test id, and prints progress with module and test identifiers.
