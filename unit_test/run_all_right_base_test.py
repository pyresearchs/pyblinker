"""Run a subset of unit tests focused on the right-base pipeline."""
import multiprocessing
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))


def main() -> None:
    """Load and execute the selected test cases."""
    multiprocessing.set_start_method("spawn", force=True)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_names = [
        (
            "unit_test.features.segment_raw_feature_pipeline.test_segment_raw_feature_pipeline.TestSegmentRawFeaturePipeline.test_pipeline_run_fit_false"
        ),
        (
            "unit_test.features.segment_raw_feature_pipeline.test_segment_raw_feature_pipeline.TestSegmentRawFeaturePipeline.test_pipeline_run_fit_true"
        ),
        (
            "unit_test.features.segmented_continous_annotated_raw.test_segment_blink_counts.TestSegmentBlinkCounts.test_blink_counts"
        ),
        (
            "unit_test.features.segmented_continous_annotated_raw.test_segment_blink_properties.TestSegmentBlinkProperties.test_properties_dataframe"
        ),
        (
            "unit_test.features.segmented_continous_annotated_raw.test_segment_blink_properties.TestSegmentBlinkProperties.test_properties_dataframe_with_fit"
        ),
    ]
    suite.addTests(loader.loadTestsFromNames(test_names))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    main()
