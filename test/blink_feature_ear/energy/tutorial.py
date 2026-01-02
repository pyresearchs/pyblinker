from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from pyblinker.blink_features.energy.energy_features import compute_energy_features
from pyblinker.outside_annotation import build_refined_blink_report
from pyblinker.utils.evaluation import mat_data
from pyblinker.utils.refinement_utils import (
    slice_raw_into_mne_epochs_refine_annot,
)

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# -----------------------------------------------------------------------------
# Load raw data and annotations
# -----------------------------------------------------------------------------
raw_path = (
    PROJECT_ROOT
    / "manual_annotation_feature_calculation_data"
    / "ear_eog.fif"
)
csv_path = (
    PROJECT_ROOT
    / "manual_annotation_feature_calculation_data"
    / "ear_eog.csv"
)

# Load raw FIF file
raw = mne.io.read_raw_fif(
    raw_path,
    preload=True,
    verbose=False,
)

# Attach manual CSV annotations
# CSV columns: onset (sec), duration (sec), description (label)
raw.set_annotations(
    mat_data.read_annotations_as_mne(csv_path)
)

# -----------------------------------------------------------------------------
# Select EAR channel
# -----------------------------------------------------------------------------
ear_channel = "EAR-avg_ear"
eeg_channel = "EEG-E8"
for required in (ear_channel, eeg_channel):
    if required not in raw.ch_names:
        raise ValueError(
            f"Required channel '{required}' not found in raw data."
        )

raw.pick([ear_channel, eeg_channel])
SEGMENT_CONFIG = {
    "ear": {
        "seg_type": "threshold_interpolation",
        "threshold": 0.22,
        "annotation_time_unit": "seconds",
        "max_extension": 0.35,
        "extension_step": 0.05,
        "padding": 0.05,
        "extend_before": True,
        "extend_after": True,
    },
    "eeg": {
        "seg_type": [],
        "threshold": None,
    },
    "eog": {
        "seg_type": [],
        "threshold": None,
    },
}
# Slice raw data into epochs
epochs = slice_raw_into_mne_epochs_refine_annot(
    raw,
    epoch_len=30.0,
    blink_label=None,
    segmentation_type=SEGMENT_CONFIG,
)

# Compute energy features
df = compute_energy_features(
    epochs,
    picks=ear_channel,
)


def _slot_value(row, key, idx):
    val = row.get(key)
    if isinstance(val, list):
        if idx < len(val):
            return val[idx]
        return np.nan
    return val


def _make_epoch_report(epoch_idx: int, output_dir: Path) -> None:
    """Generate a per-epoch blink plot overlaying EEG and EAR landmarks."""

    md_row = epochs.metadata.iloc[epoch_idx]
    n_blinks = int(md_row.get("n_blinks", 0))
    if n_blinks <= 0:
        return

    eeg_data = epochs.get_data(picks=[eeg_channel])[epoch_idx, 0]
    ear_data = epochs.get_data(picks=[ear_channel])[epoch_idx, 0]
    sfreq = float(epochs.info["sfreq"])
    time_axis = np.arange(eeg_data.size) / sfreq

    for blink_idx in range(n_blinks):
        refined_start = _slot_value(md_row, "refined_start_sample", blink_idx)
        refined_end = _slot_value(md_row, "refined_end_sample", blink_idx)
        trough = _slot_value(md_row, "refined_lowest_point_sample", blink_idx)
        left_interp = _slot_value(md_row, "left_interpolated_threshold", blink_idx)
        right_interp = _slot_value(md_row, "right_interpolated_threshold", blink_idx)
        left_interp_sample = _slot_value(md_row, "left_interpolated_threshold_sample", blink_idx)
        right_interp_sample = _slot_value(md_row, "right_interpolated_threshold_sample", blink_idx)
        win_start = _slot_value(md_row, "search_window_start_sample", blink_idx)
        win_end = _slot_value(md_row, "search_window_end_sample", blink_idx)

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        ax1.plot(time_axis, eeg_data, label=eeg_channel, color="tab:blue")
        ax2.plot(time_axis, ear_data, label=ear_channel, color="tab:orange", alpha=0.8)

        def _mark(ax, sample, color, label, linestyle="--"):
            if sample is None or np.isnan(sample):
                return
            t = float(sample) / sfreq
            ax.axvline(t, color=color, linestyle=linestyle, label=label)

        _mark(ax1, refined_start, "green", "Refined start")
        _mark(ax1, refined_end, "red", "Refined end")
        _mark(ax1, trough, "purple", "Trough", linestyle="-")
        _mark(ax1, win_start, "gray", "Search start", linestyle=":")
        _mark(ax1, win_end, "gray", "Search end", linestyle=":")
        _mark(ax1, left_interp_sample, "black", "Left interpolated", linestyle="-.")
        _mark(ax1, right_interp_sample, "black", "Right interpolated", linestyle="-.")

        # Threshold overlays
        if np.isfinite(left_interp):
            ax2.axvline(left_interp, color="black", linestyle="--", alpha=0.6)
        if np.isfinite(right_interp):
            ax2.axvline(right_interp, color="black", linestyle="--", alpha=0.6)
        ax2.axhline(SEGMENT_CONFIG["ear"]["threshold"], color="tab:gray", linestyle=":")

        ax1.set_title(f"Epoch {epoch_idx} Blink {blink_idx}")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel(f"EEG ({eeg_channel})")
        ax2.set_ylabel(f"EAR ({ear_channel})")
        handles: list = []
        labels: list = []
        for ax in (ax1, ax2):
            ax_handles, ax_labels = ax.get_legend_handles_labels()
            handles.extend(ax_handles)
            labels.extend(ax_labels)
        fig.legend(handles, labels, loc="upper right")
        fig.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / f"epoch_{epoch_idx}_blink_{blink_idx}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)


report_dir = PROJECT_ROOT / "tutorial_outputs" / "ear_energy"
for epoch_idx in range(len(epochs)):
    _make_epoch_report(epoch_idx, report_dir)

blink_summary = pd.DataFrame(
    {
        "epoch": epochs.selection,
        "blink_count": epochs.metadata["n_blinks"].to_numpy(),
    }
)

# Optionally build an HTML report per epoch using the refined blink metadata
for epoch_idx in range(len(epochs)):
    md_row = epochs.metadata.iloc[epoch_idx]
    n = int(md_row.get("n_blinks", 0))
    if n <= 0:
        continue
    rows = []
    for blink_idx in range(n):
        rec = {"epoch_index": epoch_idx, "blink_id": blink_idx}
        for key, val in md_row.items():
            rec[key] = _slot_value(md_row, key, blink_idx) if key != "n_blinks" else val
        rows.append(rec)

    if not rows:
        continue

    df_epoch = pd.DataFrame(rows)
    report = build_refined_blink_report(
        results=df_epoch,
        signal=epochs.get_data(picks=[ear_channel])[epoch_idx, 0],
        sfreq=float(epochs.info["sfreq"]),
        channel_name=ear_channel,
        overlay_signal=epochs.get_data(picks=[eeg_channel])[epoch_idx, 0],
        overlay_label=eeg_channel,
        plot_overlay=True,
        threshold_value=SEGMENT_CONFIG["ear"]["threshold"],
        mark_threshold_crossings=True,
        pad_seconds=0.25,
        epoch_label=f"Epoch {epoch_idx}",
    )
    report.add_html(
        blink_summary.to_html(index=False),
        title="Blink counts per epoch",
        section="Summary",
    )
    html_path = report_dir / f"ear_energy_report_epoch_{epoch_idx}.html"
    report.save(html_path, overwrite=True, open_browser=False)
