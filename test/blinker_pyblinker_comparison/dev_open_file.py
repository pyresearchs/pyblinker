import unittest
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.io import loadmat
from tqdm import tqdm

from pyblinker.blinker import default_setting
from pyblinker.blinker.fit_blink import FitBlinks
from pyblinker.blinker.get_blink_positions import get_blink_position
from pyblinker.blink_features.waveform_features.extract_blink_properties import (
	BlinkProperties,
	)
from pyblinker.logging import get_logger
from pyblinker.utils.statistics_utils import get_good_blink_mask, get_blink_statistic
from pyblinker.blinker.get_representative_channel import channel_selection

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBlinkProperties(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		"""
		Load test data, run FitBlinks, and compute BlinkProperties output.
		"""
		base_path = Path(__file__).resolve().parents[1]
		mat_path = base_path / "test_files/step_c_blink_properties.mat"
		
		if not mat_path.exists():
			raise FileNotFoundError(f"File not found: {mat_path}")
			
		mat_data = loadmat(
			mat_path,
			squeeze_me=True,
			simplify_cells=True,
			struct_as_record=False,
			)
	def test_dataframe_equality(self):
		pass
if __name__ == "__main__":
	unittest.main(verbosity=2)