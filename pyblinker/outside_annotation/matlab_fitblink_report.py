"""HTML report generation for MATLAB FitBlinks comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

from pyblinker.blinker import default_setting
from pyblinker.blinker.fit_blink import FitBlinks
from pyblinker.blinker.get_blink_positions import get_blink_position

LANDMARK_ORDER = [
    "max_blink",
    "max_value",
    "left_zero",
    "right_zero",
    "left_base",
    "right_base",
    "left_base_half_height",
    "right_base_half_height",
    "left_zero_half_height",
    "right_zero_half_height",
    "left_range",
    "right_range",
    "left_slope",
    "right_slope",
    "aver_left_velocity",
    "aver_right_velocity",
    "leftR2",
    "rightR2",
    "x_intersect",
    "y_intersect",
    "left_x_intercept",
    "right_x_intercept",
]

MATLAB_FIELD_MAP = {
    "maxFrame": "max_blink",
    "maxValue": "max_value",
    "leftZero": "left_zero",
    "rightZero": "right_zero",
    "leftBase": "left_base",
    "rightBase": "right_base",
    "leftBaseHalfHeight": "left_base_half_height",
    "rightBaseHalfHeight": "right_base_half_height",
    "leftZeroHalfHeight": "left_zero_half_height",
    "rightZeroHalfHeight": "right_zero_half_height",
    "leftRange": "left_range",
    "rightRange": "right_range",
    "leftSlope": "left_slope",
    "rightSlope": "right_slope",
    "averLeftVelocity": "aver_left_velocity",
    "averRightVelocity": "aver_right_velocity",
    "leftR2": "leftR2",
    "rightR2": "rightR2",
    "xIntersect": "x_intersect",
    "yIntersect": "y_intersect",
    "leftXIntercept": "left_x_intercept",
    "rightXIntercept": "right_x_intercept",
}

INDEX_COLUMNS_TO_INCREMENT = [
    "max_blink",
    "start_blink",
    "end_blink",
    "outer_start",
    "outer_end",
    "left_zero",
    "right_zero",
    "max_pos_vel_frame",
    "max_neg_vel_frame",
    "left_base",
    "right_base",
    "left_zero_half_height",
    "right_zero_half_height",
    "left_base_half_height",
    "right_base_half_height",
    "x_intersect",
    "right_x_intercept",
    "left_x_intercept",
]


def _is_finite(value: float | int | None) -> bool:
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(numeric)


def _format_numeric(value: float | int | None, *, fmt: str = "{:.4f}") -> str:
    if not _is_finite(value):
        return "n/a"
    return fmt.format(float(value))


def _coerce_range(value: object) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    if _is_finite(value):
        return [float(value)]
    return []


def _increment_range(value: object) -> object:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [v + 1 for v in value]
    return value


def _value_at_sample(signal: np.ndarray, sample: float) -> float | None:
    if not _is_finite(sample):
        return None
    x_full = np.arange(1, signal.size + 1, dtype=float)
    return float(np.interp(float(sample), x_full, signal))


def _load_matlab_signal(matlab_signal_path: Path) -> tuple[np.ndarray, float, float, str | None]:
    mat_data = loadmat(matlab_signal_path, squeeze_me=True, simplify_cells=True)
    signal = np.asarray(mat_data["blinkComp"], dtype=float)
    sfreq = float(mat_data.get("srate", 100.0))
    std_threshold = float(mat_data.get("stdThreshold", 1.5))
    channel_name = mat_data.get("channelName")
    channel_label = str(channel_name) if channel_name is not None else None
    return signal, sfreq, std_threshold, channel_label


def _load_matlab_blink_positions(matlab_positions_path: Path) -> pd.DataFrame:
    mat_data = loadmat(matlab_positions_path, squeeze_me=True, simplify_cells=True)
    blink_positions = np.asarray(mat_data["blinkPositions"], dtype=float)
    if blink_positions.shape[0] != 2:
        raise ValueError("blinkPositions must have two rows: start and end indices.")
    return pd.DataFrame(
        {
            "start_blink": blink_positions[0, :],
            "end_blink": blink_positions[1, :],
        }
    )


def _load_matlab_fitblinks(matlab_output_path: Path) -> pd.DataFrame:
    mat_data = loadmat(
        matlab_output_path,
        squeeze_me=True,
        simplify_cells=True,
        struct_as_record=False,
    )
    df_mat = pd.DataFrame(mat_data["blinkFits"]).rename(columns=MATLAB_FIELD_MAP)
    for key in ("left_range", "right_range"):
        if key in df_mat.columns:
            df_mat[key] = df_mat[key].apply(_coerce_range)
    return df_mat[LANDMARK_ORDER].reset_index(drop=True)


def _compute_python_fitblinks(
    signal: np.ndarray,
    *,
    sfreq: float,
    std_threshold: float,
    min_event_len: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = {
        "min_event_len": min_event_len,
        "std_threshold": std_threshold,
        "sfreq": sfreq,
    }
    params_default = default_setting.DEFAULT_PARAMS.copy()
    df_positions = get_blink_position(
        params,
        blink_component=signal,
        ch="No_channel",
        progress_bar=False,
    )
    fitblinks = FitBlinks(
        candidate_signal=signal,
        df=df_positions,
        params=params_default,
    )
    fitblinks.dprocess()
    df_output = fitblinks.frame_blinks.copy()

    for column in INDEX_COLUMNS_TO_INCREMENT:
        if column in df_output.columns:
            df_output[column] = df_output[column] + 1

    for column in ("left_range", "right_range"):
        if column in df_output.columns:
            df_output[column] = df_output[column].apply(_increment_range)

    df_output = df_output[LANDMARK_ORDER].reset_index(drop=True)

    df_positions = df_positions.copy()
    df_positions[["start_blink", "end_blink"]] = df_positions[
        ["start_blink", "end_blink"]
    ].astype(float) + 1
    return df_positions.reset_index(drop=True), df_output


def _build_caption(
    *,
    signal: np.ndarray,
    row: pd.Series,
    blink_index: int,
    start_blink: float,
    end_blink: float,
    plot_end: float,
) -> str:
    lines = [
        (
            f"Blink {blink_index}: start_blink={_format_numeric(start_blink, fmt='{:.0f}')}, "
            f"end_blink={_format_numeric(end_blink, fmt='{:.0f}')}, "
            f"plot_end={_format_numeric(plot_end, fmt='{:.0f}')}."
        )
    ]

    def _sample_line(label: str, sample: float | int | None) -> str:
        if not _is_finite(sample):
            return f"{label}: n/a"
        value = _value_at_sample(signal, float(sample))
        value_text = _format_numeric(value) if value is not None else "n/a"
        return (
            f"{label}: sample {_format_numeric(sample, fmt='{:.2f}')}, "
            f"value {value_text}"
        )

    lines.append(_sample_line("max_blink", row.get("max_blink")))
    lines.append(f"max_value: {_format_numeric(row.get('max_value'))}")
    for key in (
        "left_zero",
        "right_zero",
        "left_base",
        "right_base",
        "left_base_half_height",
        "right_base_half_height",
        "left_zero_half_height",
        "right_zero_half_height",
    ):
        lines.append(_sample_line(key, row.get(key)))

    for key in ("left_range", "right_range"):
        values = _coerce_range(row.get(key))
        if len(values) >= 2:
            start, end = values[0], values[1]
            start_val = _value_at_sample(signal, start)
            end_val = _value_at_sample(signal, end)
            lines.append(
                "{}: start {} (value {}), end {} (value {})".format(
                    key,
                    _format_numeric(start, fmt="{:.2f}"),
                    _format_numeric(start_val),
                    _format_numeric(end, fmt="{:.2f}"),
                    _format_numeric(end_val),
                )
            )
        else:
            lines.append(f"{key}: n/a")

    for key in (
        "left_slope",
        "right_slope",
        "aver_left_velocity",
        "aver_right_velocity",
        "leftR2",
        "rightR2",
    ):
        lines.append(f"{key}: {_format_numeric(row.get(key))}")

    lines.append(f"x_intersect: {_format_numeric(row.get('x_intersect'))}")
    lines.append(f"y_intersect: {_format_numeric(row.get('y_intersect'))}")
    lines.append(
        "left_x_intercept: {} (y=0)".format(
            _format_numeric(row.get("left_x_intercept"))
        )
    )
    lines.append(
        "right_x_intercept: {} (y=0)".format(
            _format_numeric(row.get("right_x_intercept"))
        )
    )
    return "<br>".join(lines)


def _plot_blink_landmarks(
    *,
    signal: np.ndarray,
    row: pd.Series,
    blink_index: int,
    start_blink: float,
    end_blink: float,
) -> tuple[plt.Figure, str]:
    total_samples = signal.size
    start = int(max(1, round(start_blink)))
    end = int(min(total_samples, round(end_blink + 10)))
    if end < start:
        end = start

    x_full = np.arange(1, total_samples + 1, dtype=float)
    window_x = x_full[start - 1 : end]
    window_signal = signal[start - 1 : end]

    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(10, 3),
        gridspec_kw={"width_ratios": [5, 1]},
    )
    legend_ax.axis("off")

    ax.plot(window_x, window_signal, color="C0", lw=1.0, alpha=0.85, label="blink waveform")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Blink {blink_index}")
    ax.grid(alpha=0.25)

    def _scatter_sample(label: str, sample: float | int | None, marker: str, color: str) -> None:
        if not _is_finite(sample):
            return
        sample_val = float(sample)
        if sample_val < 1 or sample_val > total_samples:
            return
        y_val = _value_at_sample(signal, sample_val)
        if y_val is None:
            return
        ax.scatter([sample_val], [y_val], label=label, marker=marker, color=color, s=56, zorder=5)

    _scatter_sample("max_blink", row.get("max_blink"), "*", "C3")
    if _is_finite(row.get("max_value")):
        ax.axhline(
            float(row.get("max_value")),
            color="C3",
            linestyle=":",
            lw=1.0,
            label="max_value",
        )

    sample_markers = {
        "left_zero": ("o", "C1"),
        "right_zero": ("o", "C2"),
        "left_base": ("s", "C4"),
        "right_base": ("s", "C5"),
        "left_base_half_height": ("D", "C6"),
        "right_base_half_height": ("D", "C7"),
        "left_zero_half_height": ("^", "C8"),
        "right_zero_half_height": ("v", "C9"),
    }
    for key, (marker, color) in sample_markers.items():
        _scatter_sample(key, row.get(key), marker, color)

    for key, color in (("left_range", "C1"), ("right_range", "C2")):
        values = _coerce_range(row.get(key))
        if len(values) >= 2 and all(_is_finite(v) for v in values[:2]):
            start_idx, end_idx = values[0], values[1]
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            range_x = np.arange(start_idx, end_idx + 1)
            range_y = _value_at_sample(signal, range_x[0])
            if range_y is not None:
                range_y_vals = np.interp(range_x, x_full, signal)
                ax.plot(
                    range_x,
                    range_y_vals,
                    color=color,
                    lw=2.0,
                    alpha=0.75,
                    label=key,
                )

    x_intersect = row.get("x_intersect")
    y_intersect = row.get("y_intersect")
    if _is_finite(x_intersect) and _is_finite(y_intersect):
        ax.scatter(
            [float(x_intersect)],
            [float(y_intersect)],
            color="C0",
            marker="X",
            s=70,
            label="x_intersect/y_intersect",
            zorder=6,
        )

    left_x_intercept = row.get("left_x_intercept")
    right_x_intercept = row.get("right_x_intercept")
    if _is_finite(left_x_intercept):
        ax.scatter(
            [float(left_x_intercept)],
            [0.0],
            color="C4",
            marker="<",
            s=60,
            label="left_x_intercept",
            zorder=6,
        )
    if _is_finite(right_x_intercept):
        ax.scatter(
            [float(right_x_intercept)],
            [0.0],
            color="C5",
            marker=">",
            s=60,
            label="right_x_intercept",
            zorder=6,
        )

    left_slope = row.get("left_slope")
    right_slope = row.get("right_slope")
    if (
        _is_finite(left_slope)
        and _is_finite(left_x_intercept)
        and _is_finite(x_intersect)
    ):
        left_x = np.linspace(float(left_x_intercept), float(x_intersect), 50)
        left_y = float(left_slope) * (left_x - float(left_x_intercept))
        ax.plot(left_x, left_y, color="C4", lw=1.5, label="left_slope")
    if (
        _is_finite(right_slope)
        and _is_finite(right_x_intercept)
        and _is_finite(x_intersect)
    ):
        right_x = np.linspace(float(x_intersect), float(right_x_intercept), 50)
        right_y = float(right_slope) * (right_x - float(right_x_intercept))
        ax.plot(right_x, right_y, color="C5", lw=1.5, label="right_slope")

    metrics_text = (
        f"left_slope: {_format_numeric(left_slope)}\n"
        f"right_slope: {_format_numeric(right_slope)}\n"
        f"aver_left_velocity: {_format_numeric(row.get('aver_left_velocity'))}\n"
        f"aver_right_velocity: {_format_numeric(row.get('aver_right_velocity'))}\n"
        f"leftR2: {_format_numeric(row.get('leftR2'))}\n"
        f"rightR2: {_format_numeric(row.get('rightR2'))}"
    )
    ax.text(
        0.99,
        0.01,
        metrics_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        seen = set()
        unique_handles = []
        unique_labels = []
        for handle, label in zip(handles, labels):
            if label in seen:
                continue
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
        legend_ax.legend(
            unique_handles,
            unique_labels,
            loc="upper left",
            fontsize=8,
            frameon=True,
            borderpad=0.6,
            labelspacing=0.3,
        )

    caption = _build_caption(
        signal=signal,
        row=row,
        blink_index=blink_index,
        start_blink=start_blink,
        end_blink=end_blink,
        plot_end=end,
    )
    return fig, caption


def build_fitblink_report(
    *,
    title: str,
    signal: np.ndarray,
    blink_positions: pd.DataFrame,
    fit_results: pd.DataFrame,
    section_label: str,
) -> mne.Report:
    report = mne.Report(title=title)
    total_blinks = min(len(blink_positions), len(fit_results))
    for idx in range(total_blinks):
        row = fit_results.iloc[idx]
        position_row = blink_positions.iloc[idx]
        start_blink = float(position_row["start_blink"])
        end_blink = float(position_row["end_blink"])
        fig, caption = _plot_blink_landmarks(
            signal=signal,
            row=row,
            blink_index=idx + 1,
            start_blink=start_blink,
            end_blink=end_blink,
        )
        report.add_figure(
            fig,
            title=f"Blink {idx + 1}",
            caption=caption,
            section=section_label,
            tags=("blink", "fitblinks"),
        )
        plt.close(fig)
    return report


def create_matlab_fitblink_reports(
    *,
    matlab_signal_path: Path,
    matlab_positions_path: Path,
    matlab_output_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    signal, sfreq, std_threshold, channel_name = _load_matlab_signal(matlab_signal_path)
    matlab_positions = _load_matlab_blink_positions(matlab_positions_path)
    matlab_fits = _load_matlab_fitblinks(matlab_output_path)

    python_positions, python_fits = _compute_python_fitblinks(
        signal,
        sfreq=sfreq,
        std_threshold=std_threshold,
    )

    label = channel_name or matlab_signal_path.stem
    matlab_report = build_fitblink_report(
        title="MATLAB output report",
        signal=signal,
        blink_positions=matlab_positions,
        fit_results=matlab_fits,
        section_label=label,
    )
    python_report = build_fitblink_report(
        title="Python-derived from MATLAB data report",
        signal=signal,
        blink_positions=python_positions,
        fit_results=python_fits,
        section_label=label,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    matlab_report_path = output_dir / "matlab_output_report.html"
    python_report_path = output_dir / "python_derived_from_matlab_report.html"
    matlab_report.save(matlab_report_path, overwrite=True, open_browser=False)
    python_report.save(python_report_path, overwrite=True, open_browser=False)
    return matlab_report_path, python_report_path


__all__ = [
    "build_fitblink_report",
    "create_matlab_fitblink_reports",
]
