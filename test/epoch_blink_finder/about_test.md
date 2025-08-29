
# Epoch Blink Finder — Tests Overview

This folder contains unit tests that validate blink detection and reporting on **MNE Epochs**.

* **`test_blink_finder.py`**
  Tests the core blink detection pipeline. Ensures detected blinks are mapped into epoch metadata and that total counts are consistent with reference data.

* **`test_blink_finder_drop.py`**
  Tests robustness when some epochs are dropped (e.g., due to artifacts, flat/saturated channels, or QC failures). Confirms blink counts remain consistent even after discarding data.

* **`test_blink_report.py`**
  Tests report generation. Verifies that an `mne.Report` is created and matches the detected blink counts, supporting validation and transparency.

