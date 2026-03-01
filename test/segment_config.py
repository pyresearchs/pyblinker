"""Helpers to build segmentation configurations for tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import mne

# In the test files, there are only 3 channels, which are EAR-avg_ear, EEG-E8, and EOG-EEG-eog_vert_left. This is to make the file smaller and easier to manage.
DEFAULT_EAR_CHANNEL = "EAR-avg_ear"
DEFAULT_EEG_CHANNEL = "EEG-E8"
DEFAULT_EOG_CHANNEL = "EOG-EEG-eog_vert_left"


def build_segment_config(
    raw: mne.io.BaseRaw,
    *,
    ear_channel: str | None = DEFAULT_EAR_CHANNEL,
    eeg_channel: str | None = DEFAULT_EEG_CHANNEL,
    eog_channel: str | None = DEFAULT_EOG_CHANNEL,
    base_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return a segmentation config with explicit single-channel entries.

    Args:
        raw: Raw recording used for channel validation.
        ear_channel: EAR channel name; required by the caller when not ``None``.
        eeg_channel: EEG channel name or ``None`` to disable EEG refinement.
        eog_channel: EOG channel name or ``None`` to disable EOG refinement.
        base_config: Optional baseline config to merge under each modality.

    Raises:
        ValueError: If a requested channel is missing from ``raw``.
    """

    config: Dict[str, Any] = deepcopy(base_config) if base_config is not None else {}

    def _normalize_channel(channel_name: str) -> str:
        if channel_name in raw.ch_names:
            return channel_name
        if channel_name.startswith("EEG") and "-" not in channel_name:
            dashed = f"EEG-{channel_name[3:]}"
            if dashed in raw.ch_names:
                return dashed
        return channel_name

    def _section(modality: str, channel: str | None) -> Dict[str, Any]:
        section = deepcopy(config.get(modality, {}))
        if channel is None:
            section.pop("channel", None)
            return section
        channel = _normalize_channel(channel)
        if channel not in raw.ch_names:
            raise ValueError(
                f"Channel '{channel}' not found in raw data for modality '{modality}'."
            )
        section["channel"] = channel
        section.setdefault("seg_type", "outer")
        return section

    if ear_channel is None:
        raise ValueError("EAR channel is required for segmentation tests.")

    config["ear"] = _section("ear", ear_channel)
    config["eeg"] = _section("eeg", eeg_channel)
    config["eog"] = _section("eog", eog_channel)
    return config
