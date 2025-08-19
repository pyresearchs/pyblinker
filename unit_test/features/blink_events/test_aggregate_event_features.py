import unittest
from pyblinker.features.blink_events.event_features import aggregate_blink_event_features
from unit_test.features.fixtures.mock_ear_generation import _generate_refined_ear


class TestAggregateBlinkFeatures(unittest.TestCase):
    """Tests for selecting blink features."""

    def setUp(self) -> None:
        blinks, sfreq, epoch_len, n_epochs = _generate_refined_ear()
        self.blinks = blinks
        self.sfreq = sfreq
        self.epoch_len = epoch_len
        self.n_epochs = n_epochs

    def test_default_features(self) -> None:
        """By default, all feature columns are returned."""
        df = aggregate_blink_event_features(
            self.blinks, self.sfreq, self.epoch_len, self.n_epochs
        )
        self.assertIn("blink_count", df.columns)
        self.assertIn("blink_rate", df.columns)
        self.assertIn("ibi_mean", df.columns)

    def test_select_subset(self) -> None:
        """Selecting only blink_count should omit other columns."""
        df = aggregate_blink_event_features(
            self.blinks,
            self.sfreq,
            self.epoch_len,
            self.n_epochs,
            features=["blink_count"],
        )
        self.assertEqual(list(df.columns), ["blink_count"])


if __name__ == "__main__":
    unittest.main()

