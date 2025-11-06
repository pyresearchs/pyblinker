"""Basic usage example showing how to run ``BlinkDetector`` on sample data."""

import os

import mne

from tutorial.utils.basic_usage import plot_blinks


if __name__ == "__main__":
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = os.path.join(
        sample_data_folder, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif"
    )
    plot_blinks(sample_data_raw_file)
