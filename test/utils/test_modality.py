# """Tests for the shared modality inference helper."""
#
# from __future__ import annotations
#
# import unittest
#
# from pyblinker.utils.modality import infer_modality
#
#
# class TestInferModality(unittest.TestCase):
#     """Validate modality detection from channel names."""
#
#     def test_keyword_detection(self) -> None:
#         """Channels containing modality keywords map accordingly."""
#         self.assertEqual(infer_modality("EEG-E8"), "eeg")
#         self.assertEqual(infer_modality("EOG-EEG-eog_vert_left"), "eog")
#         self.assertEqual(infer_modality("EAR-avg_ear"), "ear")
#
#     def test_prefix_fallback(self) -> None:
#         """The channel prefix is used when no keyword is present."""
#         self.assertEqual(infer_modality("ECG-Lead1"), "ecg")
#         self.assertEqual(infer_modality("Fp1"), "fp1")
#
#
# if __name__ == "__main__":
#     unittest.main()
