"""Helpers for the basic blink-detection tutorial."""

from __future__ import annotations

import mne

from pyblinker.blinker.pyblinker import BlinkDetector


def plot_blinks(raw_file: str) -> None:
    """Load an EEG recording and plot detected blink events."""

    raw = mne.io.read_raw_fif(raw_file, preload=True)
    raw.pick_types(eeg=True)
    raw.filter(0.5, 20.5, fir_design="firwin")
    raw.resample(100)

    channel_range = [f"EEG 00{idx}" for idx in range(10)]
    to_drop = list(set(raw.ch_names) - set(channel_range))
    if to_drop:
        raw = raw.drop_channels(to_drop)

    detector = BlinkDetector(
        raw,
        visualize=False,
        annot_label=None,
        filter_low=0.5,
        filter_high=30.0,
        resample_rate=100,
        n_jobs=2,
        use_multiprocessing=True,
    )
    annotations, channel, _good, _df, _fig_data, _selected = detector.get_blink()
    raw.set_annotations(annotations)
    raw.plot(block=True, title=f"Eye close based on channel {channel}")
