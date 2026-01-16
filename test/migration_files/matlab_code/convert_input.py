# Convert a FIF file to MATLAB .mat containing only EEG-E8 channel
# Saved variables: blinkComp (waveform), srate (=100), stdThreshold (=1.5)

import argparse
import sys
from pathlib import Path

import mne
import numpy as np
from scipy.io import savemat


def convert_fif_to_mat(input_fif: str, output_mat: str | None = None,
                       channel_name: str = "EEG-E8",
                       srate: int = 100,
                       std_threshold: float = 1.5) -> str:
    """
    Convert a FIF file to a MATLAB .mat file with only one EEG channel.

    Parameters
    ----------
    input_fif : str
        Path to the input raw FIF file.
    output_mat : str | None
        Path to the output .mat file. If None, will be derived from input path.
    channel_name : str
        Channel name to extract (default: 'EEG-E8').
    srate : int
        Target sampling rate in Hz (default: 100).
    std_threshold : float
        Value saved in the .mat as 'stdThreshold' (default: 1.5).

    Returns
    -------
    output_mat_path : str
        Path to the written .mat file.
    """
    input_path = Path(input_fif)
    if output_mat is None:
        output_mat = str(input_path.with_suffix("") ) + f"_{channel_name.replace('-', '_')}.mat"

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
        raise RuntimeError(f"Expected single channel after picking, got shape {data.shape}")
    blink_comp = np.asarray(data[0], dtype=np.float64)

    # Prepare MATLAB dict
    mat_dict = {
        "blinkComp": blink_comp,
        "srate": float(srate),
        "stdThreshold": float(std_threshold),
        "channelName": channel_name,
    }

    # Ensure output directory exists
    out_path = Path(output_mat)
    out_dir = out_path.parent
    if out_dir and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Save .mat
    savemat(str(out_path), mat_dict)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Convert FIF to MATLAB .mat for EEG-E8")
    parser.add_argument("input", nargs="?", help="Path to input FIF file")
    parser.add_argument("--output", "-o", help="Output .mat path", default=None)
    parser.add_argument("--channel", "-c", help="Channel name", default="EEG-E8")
    parser.add_argument("--srate", "-r", type=int, help="Target sampling rate", default=100)
    parser.add_argument("--std-threshold", "-t", type=float, help="stdThreshold value", default=1.5)
    args = parser.parse_args()

    # If no input arg provided, try reading the first line of this file (legacy pointer)
    input_path = args.input
    if not input_path:
        try:
            # Legacy behavior: the original file contained the relative path on the first line
            with open(__file__, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line and first_line.endswith(".fif"):
                    input_path = first_line
        except Exception:
            input_path = None

    if not input_path:
        print("Error: no input FIF path provided.")
        parser.print_help()
        sys.exit(2)

    output_path = convert_fif_to_mat(
        input_fif=input_path,
        output_mat=args.output,
        channel_name=args.channel,
        srate=args.srate,
        std_threshold=args.std_threshold,
    )
    print(f"Wrote MATLAB file: {output_path}")


if __name__ == "__main__":
    main()

