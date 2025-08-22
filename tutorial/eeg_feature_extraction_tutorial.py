"""
Tutorial: Computing Segment-Level Features for Electrophysiological Data

This tutorial demonstrates a comprehensive workflow for extracting various features
from segmented electrophysiological data. We will replicate the core logic of a feature
computation script, focusing on both time-domain complexity and frequency-domain
metrics.

The process involves these key steps:
1.  **Loading Data**: We start by loading a raw electrophysiological recording from a
    `.fif` file, a common format in MNE-Python. For this tutorial, we assume the annotations of
    blinks are already present in the raw data
2.  **Segmentation**: The continuous raw data is sliced into fixed-length epochs,
    in this case, 30-second segments. This is a standard preprocessing step for
    analyzing time-varying signals.
3.  **Blink Detection**: We identify and count blink events within each segment from an
    EEG channel. This is a practical example of event detection in physiological signals.
4.  **Feature Computation**: For each segment and for each specified channel (EAR, EOG, EEG),
    we compute a suite of features:
    *   **Time-Domain Features**: These include metrics like energy, Teager energy,
        and line length, which capture the complexity and magnitude of the signal
        in the time domain.
    *   **Frequency-Domain Features**: These metrics, such as band power, spectral
        entropy, and 1/f slope, provide insights into the signal's characteristics
        in the frequency domain.
5.  **Aggregation**: All computed features, along with the blink counts, are
    aggregated into a structured `pandas.DataFrame` for easy analysis and
    downstream machine learning tasks.

This script is designed to be a practical guide for researchers and developers
working with physiological time-series data, providing a clear example of how to
implement a feature extraction pipeline.
"""
import logging
from pathlib import Path
import mne
import pandas as pd
from pyblinker.utils.epochs import slice_raw_into_epochs
from pyblinker.features.energy_complexity.segment_features import compute_time_domain_features
from pyblinker.features.frequency_domain.segment_features import compute_frequency_domain_features
from pyblinker.features.blink_events import generate_blink_dataframe

# Configure basic logging to see informational messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the project root to locate the test file.
# This assumes a specific directory structure. You may need to adjust this path.
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
except NameError:
    # Fallback for interactive environments like Jupyter notebooks
    PROJECT_ROOT = Path.cwd()


def load_and_segment_data(raw_path: Path, epoch_len: float = 30.0) -> tuple:
    """Load raw data and segment it into epochs.

    This function reads a MNE raw file, slices it into epochs of a specified
    length, and extracts relevant information like sampling frequency and
    channel names.

    Args:
        raw_path (Path): The file path to the raw MNE data (`.fif` format).
        epoch_len (float): The desired length of each epoch in seconds.

    Returns:
        tuple: A tuple containing:
            - segments (mne.Epochs): The segmented data.
            - sfreq (float): The sampling frequency of the data.
            - channels (list[str]): A list of channels to be processed.
            - raw (mne.io.Raw): The original raw data object.
    """
    logger.info(f"Loading raw data from: {raw_path}")
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    # raw.plot(block=True)  # Display the raw data for initial inspection
    logger.info(f"Slicing raw data into {epoch_len}-second segments.")
    segments, _, _, _ = slice_raw_into_epochs(
        raw, epoch_len=epoch_len, blink_label=None, progress_bar=False
    )

    sfreq = raw.info["sfreq"]
    channels = [
        ch for ch in raw.ch_names if ch.startswith(("EAR", "EOG", "EEG"))
    ]
    logger.info(f"Identified {len(channels)} channels for feature extraction.")

    return segments, sfreq, channels, raw


def count_blinks_per_segment(segments: mne.Epochs, raw: mne.io.Raw) -> dict:
    """Generate a blink dataframe and count blinks for each segment.

    This function identifies an appropriate channel for blink detection, generates
    a dataframe of blink events, and then counts the number of blinks occurring
    in each segment.

    Args:
        segments (mne.Epochs): The segmented data.
        raw (mne.io.Raw): The original raw data object, used to find a suitable
            channel for blink detection.

    Returns:
        dict: A dictionary mapping segment index to the count of blinks.
    """
    # Select an EEG channel for blink detection; fall back to the first channel if none found.
    blink_channel = next((ch for ch in raw.ch_names if ch.startswith("EEG")), raw.ch_names[0])
    logger.info(f"Using channel '{blink_channel}' for blink detection.")

    blink_df = generate_blink_dataframe(
        segments, channel=blink_channel, blink_label=None, progress_bar=False
    )

    # Group by segment ID and count the number of blinks in each.
    blink_counts = blink_df.groupby("seg_id").size().to_dict()
    logger.info("Counted blinks per segment.")

    return blink_counts


def compute_features_for_channel(
        segments: mne.Epochs,
        channel: str,
        sfreq: float,
        blink_counts: dict,
) -> pd.DataFrame:
    """Compute and return a DataFrame of features for a single channel.

    This function iterates through all segments of the data and computes a set of
    time-domain and frequency-domain features for the specified channel.

    Args:
        segments (mne.Epochs): The segmented data.
        channel (str): The name of the channel to process.
        sfreq (float): The sampling frequency of the data.
        blink_counts (dict): A dictionary mapping segment index to blink count.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a segment and
            each column represents a computed feature for the given channel.
    """
    records = []
    logger.info(f"Computing features for channel: {channel}")
    for seg_idx, segment in enumerate(segments):
        # Extract the signal for the current channel and segment
        signal = segment.get_data(picks=channel)[0]

        # Compute features from different domains
        time_feats = compute_time_domain_features(signal, sfreq)
        freq_feats = compute_frequency_domain_features([], signal, sfreq)
        blink_count = blink_counts.get(seg_idx, 0)

        # Combine all information into a single record
        record = {
            "channel": channel,
            "segment_index": seg_idx,
            "blink_count": blink_count,
        }
        record.update(time_feats)
        record.update(freq_feats)
        records.append(record)

    return pd.DataFrame(records)


def validate_feature_dataframe(df: pd.DataFrame, n_segments: int) -> None:
    """Perform basic validation on the resulting feature DataFrame.

    This function checks for the expected number of rows, the presence of all
    expected columns, and ensures there are no missing values.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        n_segments (int): The expected number of segments (rows).

    Raises:
        AssertionError: If any of the validation checks fail.
    """
    expected_cols = {
        "channel",
        "segment_index",
        "blink_count",
        "energy",
        "teager",
        "line_length",
        "velocity_integral",
        "blink_rate_peak_freq",
        "blink_rate_peak_power",
        "broadband_power_0_5_2",
        "broadband_com_0_5_2",
        "high_freq_entropy_2_13",
        "one_over_f_slope",
        "band_power_ratio",
        "wavelet_energy_d1",
        "wavelet_energy_d2",
        "wavelet_energy_d3",
        "wavelet_energy_d4",
    }
    assert len(df) == n_segments, "Number of rows does not match number of segments."
    assert set(df.columns) == expected_cols, "DataFrame columns do not match expected columns."
    assert not df.isna().any().any(), "Found NaN values in the DataFrame."
    logger.info(f"Validation passed for DataFrame of channel '{df['channel'].iloc[0]}'.")


def main():
    """Main execution function for the tutorial."""
    # Define the path to the example data file
    # raw_path = PROJECT_ROOT / "unit_test" / "test_files" / "ear_eog_raw.fif"
    raw_path=r'C:\Users\balan\IdeaProjects\pyblinker\unit_test\test_files\ear_eog_raw.fif'
    # --- Step 1 & 2: Load and Segment Data ---
    segments, sfreq, channels, raw = load_and_segment_data(raw_path)

    # --- Step 3: Count Blinks ---
    blink_counts = count_blinks_per_segment(segments, raw)

    # --- Step 4 & 5: Compute and Aggregate Features for All Channels ---
    result_frames = []
    for ch in channels:
        df_ch = compute_features_for_channel(segments, ch, sfreq, blink_counts)
        validate_feature_dataframe(df_ch, len(segments))
        result_frames.append(df_ch)

    # Combine the DataFrames from all channels into a single master DataFrame
    combined_df = pd.concat(result_frames, ignore_index=True)

    logger.info(
        f"Successfully created a combined DataFrame with shape: {combined_df.shape}"
    )
    logger.info("Displaying the first 5 rows of the final DataFrame:")
    print(combined_df.head())

    # --- Example Analysis: Check a specific value ---
    logger.info("\n--- Example Analysis ---")
    df_ear = combined_df[combined_df["channel"] == "EAR-avg_ear"].reset_index()
    first_segment_features = df_ear.iloc[0]

    print("\nFeatures for the first segment of channel 'EAR-avg_ear':")
    print(first_segment_features)

    # Verify a few key values to ensure correctness
    energy_val = first_segment_features["energy"]
    power_val = first_segment_features["broadband_power_0_5_2"]
    blink_val = first_segment_features["blink_count"]

    assert abs(energy_val - 2.608998) < 1e-5
    assert abs(power_val - 0.13316447) < 1e-5
    assert blink_val == 2
    logger.info("\nSpecific feature values for the first segment match expected results.")


if __name__ == "__main__":
    main()