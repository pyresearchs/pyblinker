"""Reporting utilities for refined blink flow outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


def _format_metrics(row: pd.Series, keys: Sequence[str]) -> str:
    lines = []
    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, float):
            value = round(value, 4)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def build_refined_blink_report(
    *,
    results: pd.DataFrame,
    signal: np.ndarray,
    sfreq: float,
    channel_name: str,
    output_path: Path | None = None,
    pad_seconds: float = 0.1,
    max_plots: int | None = None,
    metrics_keys: Iterable[str] = (
        "peak_max_blink",
        "peak_time_blink",
        "duration_zero",
        "duration_base",
        "closing_time_zero",
        "reopening_time_zero",
    ),
) -> mne.Report:
    """Generate an MNE report visualizing refined blink boundaries and metrics."""

    report = mne.Report(title="Refined Blink Validation")
    n_samples = signal.shape[0]
    pad_samples = int(round(pad_seconds * sfreq))

    rows = results.itertuples(index=False)
    if max_plots is not None:
        rows = list(rows)[: int(max_plots)]

    for idx, row in enumerate(rows):
        left = int(getattr(row, "refined_left_zero", getattr(row, "left_zero", 0)))
        right = int(getattr(row, "refined_right_zero", getattr(row, "right_zero", 0)))
        start = max(0, left - pad_samples)
        end = min(n_samples - 1, right + pad_samples)

        window_times = np.arange(start, end + 1, dtype=float) / sfreq
        window_signal = signal[start : end + 1]

        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(window_times, window_signal, lw=1.0, alpha=0.85, color="C0")
        ax.axvline(left / sfreq, color="C1", linestyle="--", label="Left zero crossing")
        ax.axvline(right / sfreq, color="C2", linestyle="--", label="Right zero crossing")

        # Mark key landmarks directly on the plot for clarity.
        zero_x = [left / sfreq, right / sfreq]
        zero_y = [0.0, 0.0]
        ax.scatter(zero_x, zero_y, color="C1", zorder=4)
        ax.annotate(
            "Left zero crossing",
            xy=(zero_x[0], zero_y[0]),
            xytext=(zero_x[0], min(window_signal)),
            arrowprops=dict(arrowstyle="->", color="C1"),
            fontsize=8,
            ha="center",
        )
        ax.annotate(
            "Right zero crossing",
            xy=(zero_x[1], zero_y[1]),
            xytext=(zero_x[1], max(window_signal)),
            arrowprops=dict(arrowstyle="->", color="C2"),
            fontsize=8,
            ha="center",
        )

        # Maximum absolute amplitude within the window.
        max_idx = int(np.argmax(np.abs(window_signal)))
        max_time = window_times[max_idx]
        max_amp = window_signal[max_idx]
        ax.scatter([max_time], [max_amp], color="C3", zorder=5, label="Max |amplitude|")
        ax.annotate(
            f"Max |amp| = {max_amp:.3f}",
            xy=(max_time, max_amp),
            xytext=(max_time, max_amp * 1.1 if max_amp != 0 else 0.1),
            arrowprops=dict(arrowstyle="->", color="C3"),
            fontsize=8,
            ha="center",
        )

        ax.set_title(f"Candidate {getattr(row, 'candidate_id', idx)} • {channel_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.25)

        metrics_text = _format_metrics(pd.Series(row._asdict()), list(metrics_keys))
        if metrics_text:
            ax.text(
                0.99,
                0.99,
                metrics_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.7"),
            )

        caption = (
            f"Zero crossings at {left / sfreq:.3f}s and {right / sfreq:.3f}s. "
            f"Sampling rate: {sfreq:.2f} Hz. "
            f"Segment {start}–{end} ({(end - start) / sfreq:.3f}s)."
        )
        report.add_figure(
            fig=fig,
            title=f"Blink {idx}",
            caption=caption,
            section="Refined blinks",
            tags=("blink", "refined", channel_name),
        )
        plt.close(fig)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save(output_path, overwrite=True, open_browser=False)

    return report


__all__ = ["build_refined_blink_report"]
