"""
Script to convert ear_eog_raw.fif to ear_eog_raw.mat directly without resampling.
This preserves the original sampling frequency and includes the full params structure.
"""

from pathlib import Path
import mne
import numpy as np
from scipy.io import savemat
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TEST_FILES_DIR = Path(__file__).resolve().parent.parent / "test_files"
INPUT_FIF = TEST_FILES_DIR / "ear_eog_raw.fif"
OUTPUT_MAT = TEST_FILES_DIR / "ear_eog_raw.mat"

def convert_direct(input_fif: Path, output_mat: Path, channel_name: str = "EEG-E8", std_threshold: float = 1.5):
    """
    Convert FIF to MAT without resampling, including params.
    """
    logger.info("Reading FIF: %s", input_fif)
    raw = mne.io.read_raw_fif(str(input_fif), preload=True, verbose="ERROR")

    # Find channel
    if channel_name not in raw.ch_names:
        lower_map = {ch.lower(): ch for ch in raw.ch_names}
        key = channel_name.lower()
        if key in lower_map:
            channel_name = lower_map[key]
        else:
            raise RuntimeError(f"Channel {channel_name} not found.")

    raw_pick = raw.copy().pick_channels([channel_name])
    
    # Get original sfreq
    srate = raw_pick.info["sfreq"]
    logger.info("Original sfreq: %s Hz", srate)

    data = raw_pick.get_data()
    blink_comp = np.asarray(data[0], dtype=np.float64)

    # Params structure (mirrored from convert_input_fif_to_edf.py)
    params = {
        'srate': float(srate),
        'stdThreshold': float(std_threshold),
        'subjectID': 'Subject1_Task1_Experiment1_Rep1',
        'uniqueName': 'Unknown',
        'experiment': 'Experiment1',
        'task': 'Task1',
        'startDate': '01-Jan-2016',
        'startTime': '00:00:00',
        'signalTypeIndicator': 'UseNumbers',
        'signalNumbers': 1,
        'signalLabels': np.array(['002'], dtype=object),
        'excludeLabels': np.array(['exg5', 'exg6', 'exg7', 'exg8', 'vehicle position'], dtype=object),
        'dumpBlinkerStructures': 0,
        'showMaxDistribution': 1,
        'dumpBlinkImages': 0,
        'verbose': 1,
        'dumpBlinkPositions': 0,
        'fileName': '',
        'blinkerSaveFile': r'C:\eeg_lab_matlab\eeglab2024.2\_blinks.mat',
        'blinkerDumpDir': r'C:\eeg_lab_matlab\eeglab2024.2\blinkDump',
        'lowCutoffHz': 1,
        'highCutoffHz': 20,
        'minGoodBlinks': 10,
        'blinkAmpRange': np.array([3, 50]),
        'goodRatioThreshold': 0.7,
        'pAVRThreshold': 3,
        'correlationThresholdTop': 0.98,
        'correlationThresholdBottom': 0.9,
        'correlationThresholdMiddle': 0.95,
        'keepSignals': 0,
        'shutAmpFraction': 0.9,
        'zThresholds': np.array([[0.9, 2.], [0.98, 5.]]),
        'ICSimilarityThreshold': 0.85,
        'ICFOMThreshold': 1,
        'numberMaxBins': 80
    }

    mat_dict = {
        "blinkComp": blink_comp,
        "srate": float(srate),
        "stdThreshold": float(std_threshold),
        "channelName": channel_name,
        "params": params
    }

    logger.info("Saving MAT: %s", output_mat)
    savemat(str(output_mat), mat_dict)

if __name__ == "__main__":
    if not INPUT_FIF.exists():
        logger.error("Input file not found: %s", INPUT_FIF)
    else:
        convert_direct(INPUT_FIF, OUTPUT_MAT)
