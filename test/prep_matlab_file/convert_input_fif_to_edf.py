# Convert a FIF file to MATLAB .mat containing only EEG-E8 channel
# Saved variables: blinkComp (waveform), srate (=100), stdThreshold (=1.5)

from pathlib import Path

import mne
import numpy as np
from scipy.io import savemat


def convert_fif_to_mat(
    input_fif: str,
    output_mat: str | None = None,
    channel_name: str = "EEG-E8",
    srate: int = 100,
    std_threshold: float = 1.5,
) -> str:
    """
    Convert a raw FIF file to a MATLAB .mat file containing a single EEG channel.

    This function loads a FIF file, extracts a specified channel, resamples it to
    a target sampling rate, and saves it as a .mat file. The output .mat file
    contains the following variables:
        - blinkComp: The EEG data as a 1D array.
        - srate: The sampling rate of the data.
        - stdThreshold: A threshold value for blink detection.
        - channelName: The name of the extracted channel.

    Parameters
    ----------
    input_fif : str
        Path to the input raw FIF file.
    output_mat : str | None, optional
        Path to the output .mat file. If None, the output filename is derived
        from the input filename and channel name.
    channel_name : str, optional
        The name of the channel to extract (default is "EEG-E8").
    srate : int, optional
        The target sampling rate in Hz (default is 100).
    std_threshold : float, optional
        The standard deviation threshold to be saved in the .mat file (default is 1.5).

    Returns
    -------
    str
        The absolute path to the generated .mat file.

    Raises
    ------
    RuntimeError
        If the specified channel is not found in the FIF file.
    """
    input_path = Path(input_fif)
    if output_mat is None:
        output_mat = (
            str(input_path.with_suffix("")) + f"_{channel_name.replace('-', '_')}.mat"
        )

    # Load raw FIF
    raw = mne.io.read_raw_fif(str(input_path), preload=True, verbose="ERROR")

    # Ensure channel exists
    if channel_name not in raw.ch_names:
        # Try case-insensitive match
        lower_map = {ch.lower(): ch for ch in raw.ch_names}
        key = channel_name.lower()
        if key in lower_map:
            channel_name = lower_map[key]
        else:
            raise RuntimeError(
                f"Channel '{channel_name}' not found in FIF. Available: {raw.ch_names}"
            )

    # Pick the single channel
    raw_pick = raw.copy().pick_channels([channel_name])

    # Resample to target srate
    if int(round(raw_pick.info.get("sfreq", srate))) != srate:
        raw_pick.resample(sfreq=srate)

    # Extract data for the channel as 1D numpy array
    data = raw_pick.get_data()
    if data.shape[0] != 1:
        raise RuntimeError(
            f"Expected single channel after picking, got shape {data.shape}"
        )
    blink_comp = np.asarray(data[0], dtype=np.float64)

    # Replicate parameters from test/params_file.mat
    params = {
        "srate": float(srate),
        "stdThreshold": float(std_threshold),
        "subjectID": "Subject1_Task1_Experiment1_Rep1",
        "uniqueName": "Unknown",
        "experiment": "Experiment1",
        "task": "Task1",
        "startDate": "01-Jan-2026",
        "startTime": "00:00:00",
        "signalTypeIndicator": "UseNumbers",
        "signalNumbers": 1,
        "signalLabels": np.array(["002"], dtype=object),
        "excludeLabels": np.array(
            ["exg5", "exg6", "exg7", "exg8", "vehicle position"], dtype=object
        ),
        "dumpBlinkerStructures": 0,
        "showMaxDistribution": 1,
        "dumpBlinkImages": 0,
        "verbose": 1,
        "dumpBlinkPositions": 0,
        "fileName": "",
        "blinkerSaveFile": r"C:\eeg_lab_matlab\eeglab2024.2\_blinks.mat",
        "blinkerDumpDir": r"C:\eeg_lab_matlab\eeglab2024.2\blinkDump",
        "lowCutoffHz": 1,
        "highCutoffHz": 20,
        "minGoodBlinks": 10,
        "blinkAmpRange": np.array([3, 50]),
        "goodRatioThreshold": 0.7,
        "pAVRThreshold": 3,
        "correlationThresholdTop": 0.98,
        "correlationThresholdBottom": 0.9,
        "correlationThresholdMiddle": 0.95,
        "keepSignals": 0,
        "shutAmpFraction": 0.9,
        "zThresholds": np.array([[0.9, 2.0], [0.98, 5.0]]),
        "ICSimilarityThreshold": 0.85,
        "ICFOMThreshold": 1,
        "numberMaxBins": 80,
    }

    # Prepare MATLAB dict
    mat_dict = {
        "blinkComp": blink_comp,
        "srate": float(srate),
        "stdThreshold": float(std_threshold),
        "channelName": channel_name,
        "params": params,
    }

    # Ensure output directory exists
    out_path = Path(output_mat)
    out_dir = out_path.parent
    if out_dir and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Save .mat
    savemat(str(out_path), mat_dict)
    return str(out_path)
