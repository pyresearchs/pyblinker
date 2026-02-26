"""Shared style-window resolution for frame-indexed blink metadata."""

from __future__ import annotations

from typing import Dict, List, Mapping, Set

from .utils.style_windows import extract_windows


def style_windows_from_metadata(
    metadata_row: Mapping[str, object],
    modality: str,
    available_styles: Set[str],
    n_times: int,
    *,
    include_half: bool = True,
    include_peak: bool = True,
    ear_mode: str = "keep",
) -> Dict[str, List[tuple[int, int]]]:
    """Resolve output style names to frame windows for one modality.

    Parameters
    ----------
    ear_mode:
        ``"keep"`` keeps ``th_interpolation`` as key.
        ``"map_to_th_point"`` maps EAR interpolation windows to ``th_point``.
    """

    style_windows: Dict[str, List[tuple[int, int]]] = {}
    if modality in {"eeg", "eog"}:
        if "zero" in available_styles:
            style_windows["zero"] = extract_windows(metadata_row, modality, "zero", n_times)
        if "base" in available_styles:
            style_windows["base"] = extract_windows(metadata_row, modality, "base", n_times)
        if "tent" in available_styles:
            style_windows["tent"] = extract_windows(metadata_row, modality, "tent", n_times)

        if include_half:
            if "half_base" in available_styles:
                style_windows["half"] = extract_windows(metadata_row, modality, "half_base", n_times)
            elif "half_zero" in available_styles:
                style_windows["half"] = extract_windows(metadata_row, modality, "half_zero", n_times)

        if include_peak:
            if "tent" in style_windows:
                style_windows["peak"] = style_windows["tent"]
            elif "base" in style_windows:
                style_windows["peak"] = style_windows["base"]

    elif modality == "ear":
        if "th_point" in available_styles:
            style_windows["th_point"] = extract_windows(metadata_row, modality, "th_point", n_times)
        elif "th_interpolation" in available_styles:
            key = "th_point" if ear_mode == "map_to_th_point" else "th_interpolation"
            style_windows[key] = extract_windows(metadata_row, modality, "th_interpolation", n_times)

    return style_windows
