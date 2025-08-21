"""Run a subset of unit tests focused on the right-base pipeline."""
from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path
import unittest

from . import download_migration_files, download_test_files

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Download datasets, load, and execute the selected test cases."""
    download_migration_files()
    download_test_files()
    multiprocessing.set_start_method("spawn", force=True)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_names = [
        (
            "unit_test.features.segment_raw_feature_pipeline."
            "test_segment_raw_feature_pipeline."
            "TestSegmentRawFeaturePipeline.test_pipeline_run_fit_false"
        ),
        (
            "unit_test.features.segment_raw_feature_pipeline."
            "test_segment_raw_feature_pipeline."
            "TestSegmentRawFeaturePipeline.test_pipeline_run_fit_true"
        ),
        (
            "unit_test.features.segmented_continous_annotated_raw."
            "test_segment_blink_counts."
            "TestSegmentBlinkCounts.test_blink_counts"
        ),
        (
            "unit_test.features.segmented_continous_annotated_raw."
            "test_segment_blink_properties."
            "TestSegmentBlinkProperties.test_properties_dataframe"
        ),
        (
            "unit_test.features.segmented_continous_annotated_raw."
            "test_segment_blink_properties."
            "TestSegmentBlinkProperties.test_properties_dataframe_with_fit"
        ),
    ]
    suite.addTests(loader.loadTestsFromNames(test_names))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    main()
