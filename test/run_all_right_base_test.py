"""Run a subset of unit tests focused on the right-base pipeline."""
from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path
import sys
import unittest

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))


def main() -> None:
    """Download datasets, load, and execute the selected test cases."""
    from test import download_migration_files, download_test_files

    download_migration_files()
    download_test_files()
    multiprocessing.set_start_method("spawn", force=True)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_names = [
        (
            "test.features.segment_raw_feature_pipeline.",
            "test_segment_raw_feature_pipeline.",
            "TestSegmentRawFeaturePipeline.test_pipeline_run_fit_false",
        ),
        (
            "test.features.segment_raw_feature_pipeline.",
            "test_segment_raw_feature_pipeline.",
            "TestSegmentRawFeaturePipeline.test_pipeline_run_fit_true",
        ),
        (
            "test.features.segmented_continous_annotated_raw.",
            "test_segment_blink_counts.",
            "TestSegmentBlinkCounts.test_blink_counts",
        ),
        (
            "test.features.segmented_continous_annotated_raw.",
            "test_segment_blink_properties.",
            "TestSegmentBlinkProperties.test_properties_dataframe",
        ),
        (
            "test.features.segmented_continous_annotated_raw.",
            "test_segment_blink_properties.",
            "TestSegmentBlinkProperties.test_properties_dataframe_with_fit",
        ),
    ]
    suite.addTests(loader.loadTestsFromNames(test_names))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    main()
