# """Tests for channel utility helpers."""
# from __future__ import annotations
#
# import unittest
#
# import mne
# import numpy as np
#
# from pyblinker.utils.channel_utils import (
#     pick_ear_channels_from_info,
#     pick_ear_channels_from_raw,
# )
#
#
# class TestChannelUtils(unittest.TestCase):
#     def setUp(self) -> None:
#         ch_names = ["EEG-Fz", "EAR_left", "eye_aspect_ratio", "A1"]
#         info = mne.create_info(ch_names=ch_names, sfreq=100.0, ch_types=["eeg", "misc", "misc", "eeg"])
#         data = np.random.randn(len(ch_names), 100)
#         self.raw = mne.io.RawArray(data, info)
#
#     def test_pick_ear_channels_from_info(self) -> None:
#         indices = pick_ear_channels_from_info(self.raw.info)
#         self.assertEqual(indices, [1, 2])
#
#     def test_pick_ear_channels_from_raw(self) -> None:
#         indices = pick_ear_channels_from_raw(self.raw)
#         self.assertEqual(indices, [1, 2])
#
#
# if __name__ == "__main__":
#     unittest.main()
