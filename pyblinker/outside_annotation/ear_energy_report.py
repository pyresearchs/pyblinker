"""EAR-specific blink reporting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _slot_value(row: pd.Series, key: str, idx: int) -> float | int | None:
    """Extract a possibly list-valued metadata entry for blink ``idx``."""

    value = row.get(key)
    if isinstance(value, list):
        return value[idx] if idx < len(value) else np.nan
    return value


def _plot_blink(
    *,
    ear_data: np.ndarray,
    eeg_data: np.ndarray | None,
    sfreq: float,
    epoch_idx: int,
    blink_idx: int,
    md_row: pd.Series,
    threshold: float | None,
) -> plt.Figure:
    """Render a single blink overlay plot."""

    refined_start = _slot_value(md_row, "refined_start_sample", blink_idx)
    refined_end = _slot_value(md_row, "refined_end_sample", blink_idx)
    refined_lowest = _slot_value(md_row, "refined_lowest_point_sample", blink_idx)
    left_interp_time = _slot_value(md_row, "left_interpolated_threshold", blink_idx)
    right_interp_time = _slot_value(md_row, "right_interpolated_threshold", blink_idx)
    left_interp_sample = _slot_value(md_row, "left_interpolated_threshold_sample", blink_idx)
    right_interp_sample = _slot_value(md_row, "right_interpolated_threshold_sample", blink_idx)
    win_start = _slot_value(md_row, "search_window_start_sample", blink_idx)
    win_end = _slot_value(md_row, "search_window_end_sample", blink_idx)

    total_samples = ear_data.shape[0]
    left_candidates = [
        refined_start,
        left_interp_sample,
        win_start,
    ]
    right_candidates = [
        refined_end,
        right_interp_sample,
        win_end,
    ]
    left_anchor = min(int(x) for x in left_candidates if x is not None and np.isfinite(x)) if any(
        x is not None and np.isfinite(x) for x in left_candidates
    ) else 0
    right_anchor = max(int(x) for x in right_candidates if x is not None and np.isfinite(x)) if any(
        x is not None and np.isfinite(x) for x in right_candidates
    ) else total_samples - 1
    start_sample = max(0, min(total_samples - 1, left_anchor - 6))
    end_sample = max(start_sample, min(total_samples - 1, right_anchor + 6))

    slice_time = np.arange(start_sample, end_sample + 1) / sfreq
    ear_slice = ear_data[start_sample : end_sample + 1]
    eeg_slice = eeg_data[start_sample : end_sample + 1] if eeg_data is not None else None

    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(10, 3),
        gridspec_kw={"width_ratios": [5, 1]},
    )
    legend_ax.axis("off")
    ax.scatter(slice_time, ear_slice, label="EAR-avg_ear", color="C0", s=28, alpha=0.9, zorder=4)
    ax.plot(slice_time, ear_slice, color="C0", alpha=0.35, linewidth=0.9, zorder=2)
    ax.set_ylabel("EAR-avg_ear")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Epoch {epoch_idx} • Blink {blink_idx} • EAR-avg_ear")

    if threshold is not None:
        ax.axhline(threshold, color="C5", linestyle=":", linewidth=1.0, alpha=0.9, label=f"Threshold = {threshold:.3f}")

    def _sample_value(sample: float | int | None) -> tuple[float, float] | None:
        if sample is None or np.isnan(sample):
            return None
        idx = int(np.clip(round(float(sample)), 0, total_samples - 1))
        return idx / sfreq, ear_data[idx]

    markers = [
        ("Refined start", refined_start, "D", "C3"),
        ("Refined end", refined_end, ">", "C4"),
        ("Refined lowest point", refined_lowest, "p", "C7"),
    ]
    for label, sample, marker, color in markers:
        point = _sample_value(sample)
        if point is None:
            continue
        t, val = point
        if slice_time[0] <= t <= slice_time[-1]:
            ax.scatter([t], [val], marker=marker, color=color, s=64, zorder=6, alpha=0.95, label=label)

    interpolated_markers = []
    if left_interp_time is not None and np.isfinite(left_interp_time):
        interpolated_markers.append(("Left interpolated threshold", float(left_interp_time)))
    if right_interp_time is not None and np.isfinite(right_interp_time):
        interpolated_markers.append(("Right interpolated threshold", float(right_interp_time)))

    if interpolated_markers:
        marker_styles = {
            "Left interpolated threshold": ("^", "C1"),
            "Right interpolated threshold": ("v", "C2"),
        }
        for label, time_val in interpolated_markers:
            marker, color = marker_styles.get(label, ("x", "0.3"))
            if slice_time[0] <= time_val <= slice_time[-1]:
                interp_val = float(np.interp(time_val, slice_time, ear_slice))
                ax.scatter(
                    [time_val],
                    [interp_val],
                    color=color,
                    marker=marker,
                    s=64,
                    zorder=7,
                    alpha=0.95,
                    label=label,
                )

    overlay_ax = None
    if eeg_slice is not None:
        ax2 = ax.twinx()
        overlay_ax = ax2
        ax2.plot(slice_time, eeg_slice, color="mediumorchid", alpha=0.6, linewidth=1.0, label="EEG-E8")
        ax2.set_ylabel("EEG-E8")
        ax2.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    if overlay_ax is not None:
        h2, l2 = overlay_ax.get_legend_handles_labels()
        handles.extend(h2)
        labels.extend(l2)

    caption_parts = []
    if left_interp_sample is not None and np.isfinite(left_interp_sample):
        caption_parts.append(f"Left interpolated threshold sample: {int(round(float(left_interp_sample)))}")
    if right_interp_sample is not None and np.isfinite(right_interp_sample):
        caption_parts.append(f"Right interpolated threshold sample: {int(round(float(right_interp_sample)))}")
    if refined_lowest is not None and np.isfinite(refined_lowest):
        caption_parts.append(f"Lowest point sample: {int(round(float(refined_lowest)))}")
    if caption_parts:
        ax.set_title(f"Epoch {epoch_idx} • Blink {blink_idx} • EAR-avg_ear\n" + " | ".join(caption_parts))

    if handles:
        seen: set[str] = set()
        uniq_handles = []
        uniq_labels = []
        for handle, label in zip(handles, labels):
            if label in seen:
                continue
            seen.add(label)
            uniq_handles.append(handle)
            uniq_labels.append(label)
        legend_ax.legend(
            uniq_handles,
            uniq_labels,
            loc="upper left",
            fontsize=8,
            frameon=True,
            borderpad=0.6,
            labelspacing=0.3,
            ncol=1,
        )

    fig.subplots_adjust(wspace=0.05)

    fig.tight_layout()
    return fig


def build_ear_energy_report(
    *,
    epochs: mne.Epochs,
    ear_channel: str,
    eeg_channel: str | None,
    threshold: float | None,
    output_path: Path,
) -> None:
    """Create a single HTML report with per-epoch sections and per-blink figures."""

    report = mne.Report(title="EAR Energy Blink Report")
    sfreq = float(epochs.info["sfreq"])
    blink_summary = pd.DataFrame(
        {
            "epoch": epochs.selection,
            "blink_count": epochs.metadata["n_blinks"].to_numpy(),
        }
    )
    report.add_html(blink_summary.to_html(index=False), title="Blink counts per epoch", section="Summary")

    ear_data_all = epochs.get_data(picks=[ear_channel])
    eeg_data_all = epochs.get_data(picks=[eeg_channel]) if eeg_channel is not None else None

    for epoch_idx, md_row in epochs.metadata.iterrows():
        n_blinks = int(md_row.get("n_blinks", 0))
        if n_blinks <= 0:
            continue
        for blink_idx in range(n_blinks):
            fig = _plot_blink(
                ear_data=ear_data_all[epoch_idx, 0],
                eeg_data=eeg_data_all[epoch_idx, 0] if eeg_data_all is not None else None,
                sfreq=sfreq,
                epoch_idx=epoch_idx,
                blink_idx=blink_idx,
                md_row=md_row,
                threshold=threshold,
            )
            report.add_figure(
                fig,
                title=f"Blink {blink_idx}",
                section=f"Epoch {epoch_idx}",
                caption=f"Epoch {epoch_idx} • Blink {blink_idx}",
            )
            plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(output_path, overwrite=True, open_browser=False)


__all__: Sequence[str] = ["build_ear_energy_report"]
