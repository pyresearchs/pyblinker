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


def _compute_overlay_indices(
    start: int,
    end: int,
    base_sfreq: float,
    overlay_len: int,
    overlay_sfreq: float | None,
) -> tuple[int, int]:
    """Compute overlay index range aligned to base sampling.

    Uses floating point conversion to seconds for robustness, then maps into the overlay
    signal index space while clamping to valid bounds.
    """

    derived_sfreq = base_sfreq if overlay_sfreq is None else overlay_sfreq
    derived_sfreq = float(derived_sfreq)

    start_time = start / base_sfreq
    end_time = end / base_sfreq

    overlay_start = int(np.clip(round(start_time * derived_sfreq), 0, overlay_len - 1))
    overlay_end = int(np.clip(round(end_time * derived_sfreq), overlay_start, overlay_len - 1))
    return overlay_start, overlay_end


def build_refined_blink_report(
    *,
    results: pd.DataFrame,
    signal: np.ndarray,
    sfreq: float,
    channel_name: str,
    overlay_signal: np.ndarray | None = None,
    overlay_sfreq: float | None = None,
    overlay_label: str = "EAR-avg_ear",
    plot_overlay: bool = False,
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

    total_candidates = len(results)
    rows = list(results.itertuples(index=False))
    if max_plots is not None:
        rows = rows[: int(max_plots)]

    plotted_count = len(rows)
    skipped_count = max(total_candidates - plotted_count, 0)
    zero_crossing_failures = None
    if "zero_crossing_found" in results.columns:
        zero_crossing_failures = int((~results["zero_crossing_found"].astype(bool)).sum())

    for idx, row in enumerate(rows):
        left = int(getattr(row, "refined_left_zero", getattr(row, "left_zero", 0)))
        right = int(getattr(row, "refined_right_zero", getattr(row, "right_zero", 0)))
        start = max(0, left - pad_samples)
        end = min(n_samples - 1, right + pad_samples)

        window_times = np.arange(start, end + 1, dtype=float) / sfreq
        window_signal = signal[start : end + 1]

        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(
            window_times,
            window_signal,
            lw=1.0,
            alpha=0.85,
            color="C0",
            label=channel_name,
        )
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

        overlay_ax = None
        if plot_overlay and overlay_signal is not None:
            overlay_start, overlay_end = _compute_overlay_indices(
                start=start,
                end=end,
                base_sfreq=sfreq,
                overlay_len=overlay_signal.shape[0],
                overlay_sfreq=overlay_sfreq,
            )
            overlay_times = np.arange(overlay_start, overlay_end + 1, dtype=float) / (
                sfreq if overlay_sfreq is None else float(overlay_sfreq)
            )
            overlay_window = overlay_signal[overlay_start : overlay_end + 1]

            overlay_ax = ax.twinx()
            overlay_ax.plot(
                overlay_times,
                overlay_window,
                lw=1.0,
                alpha=0.75,
                color="C4",
                label=overlay_label,
            )
            overlay_ax.set_ylabel(overlay_label)

        ax.set_title(f"Candidate {getattr(row, 'candidate_id', idx)} • {channel_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(channel_name)
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

        handles, labels = ax.get_legend_handles_labels()
        if overlay_ax is not None:
            overlay_handles, overlay_labels = overlay_ax.get_legend_handles_labels()
            handles.extend(overlay_handles)
            labels.extend(overlay_labels)
        if handles:
            ax.legend(handles, labels, loc="upper right")

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

    summary_rows = [
        ("Total refined blinks", total_candidates),
        ("Blinks plotted", plotted_count),
    ]
    if skipped_count:
        reason = "max_plots limit" if max_plots is not None else "not plotted"
        summary_rows.append((f"Skipped ({reason})", skipped_count))
    if zero_crossing_failures is not None:
        summary_rows.append(("Zero-crossing failures", zero_crossing_failures))

    summary_html = """<table style='border-collapse: collapse;'>
    <thead><tr><th style='text-align:left;padding:4px;'>Metric</th>
    <th style='text-align:left;padding:4px;'>Value</th></tr></thead><tbody>"""
    for label, value in summary_rows:
        summary_html += (
            f"<tr><td style='padding:4px;border-top:1px solid #ddd;'>{label}</td>"
            f"<td style='padding:4px;border-top:1px solid #ddd;'>{value}</td></tr>"
        )
    summary_html += "</tbody></table>"
    report.add_html(
        title="Refined blink summary",
        html=summary_html,
        section="Summary",
        tags=("summary", "refined", channel_name),
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.save(output_path, overwrite=True, open_browser=False)

    return report


__all__ = ["build_refined_blink_report"]
