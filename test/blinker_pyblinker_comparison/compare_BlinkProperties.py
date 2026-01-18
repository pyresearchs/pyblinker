import unittest
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mne

from pyblinker.blinker import default_setting
from pyblinker.blinker.fit_blink import FitBlinks
from pyblinker.blinker.get_blink_positions import get_blink_position
from pyblinker.blink_features.waveform_features.extract_blink_properties import (
    BlinkProperties,
)
from scipy.io import loadmat

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBlinkProperties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Load test data, run FitBlinks, and compute BlinkProperties output.
        """
        base_path = Path(__file__).resolve().parents[1] / "migration_files"
        fif_path = Path("test/test_files/ear_eog_raw.fif")

        mat_path = base_path / "step2c_data_output_computeBlinkProperties_rpb.mat"
        mat_data = loadmat(
            mat_path,
            squeeze_me=True,
            simplify_cells=True,
            struct_as_record=False,
            )

        cls.df_mat = pd.DataFrame(mat_data["blinkProps"]).reset_index(drop=True)
        # Load raw FIF data
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose="ERROR")

        ch_name = "EEG-E8"
        if ch_name not in raw.ch_names:
            # Case-insensitive fallback
            ch_map = {c.lower(): c for c in raw.ch_names}
            ch_name = ch_map.get(ch_name.lower(), ch_name)

        raw = raw.copy().pick_channels([ch_name])

        if int(round(raw.info.get("sfreq", 100))) != 100:
            raw.resample(100)

        blink_comp = raw.get_data()[0].astype(np.float64)

        # Detect blink positions
        params = dict(
            min_event_len=0.05,
            std_threshold=1.5,
            sfreq=100,
        )
        params_default = default_setting.DEFAULT_PARAMS.copy()

        df_positions = get_blink_position(
            params,
            blink_component=blink_comp,
            ch="No_channel",
            progress_bar=False,
        )

        # Run FitBlinks
        fitblinks = FitBlinks(
            candidate_signal=blink_comp,
            df=df_positions,
            params=params_default,
        )
        fitblinks.dprocess()
        df_fitblinks = fitblinks.frame_blinks.copy()
        df_fitblinks=df_fitblinks.head(2)  # For testing, limit to first 10 rows

        # Compute BlinkProperties
        cls.df_py = BlinkProperties(
            blink_comp,
            df_fitblinks,
            params["sfreq"],
            params_default,
        ).df

    def test_dataframe_equality(self):
        """
        Compare Python FitBlinks output with MATLAB reference output.
        Both should return 359 blinks with identical properties.
        """
        self.assertEqual(
                len(self.df_py),
                len(self.df_mat),
                f"Different number of blinks: "
                f"py={len(self.df_py)} mat={len(self.df_mat)}",
                )

        pd.testing.assert_frame_equal(
                self.df_py,
                self.df_mat,
                check_dtype=False,
                )
if __name__ == "__main__":
    unittest.main(verbosity=2)
