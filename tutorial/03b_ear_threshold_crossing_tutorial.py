"""Executable demo for EAR threshold crossings and slope features.

This script mirrors the math in ``tutorial/03b_ear_threshold_crossing_tutorial.md``:
it generates a mock EAR blink, detects threshold crossings using
``find_threshold_crossing_triplet``, computes closing/opening slopes, and
optionally plots the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from pyblinker.fitutils.ear_crossing import (
    compute_threshold_slopes,
    find_threshold_crossing_triplet,
)


def generate_mock_ear(
    fs: float = 250.0,
    duration: float = 2.0,
    center: float = 1.0,
    sigma: float = 0.07,
    depth: float = 0.19,
    baseline: float = 0.30,
    noise_std: float = 0.004,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a synthetic EAR signal with a Gaussian-shaped blink."""

    t = np.arange(0, duration, 1 / fs)
    rng = np.random.default_rng(seed)
    noise = noise_std * rng.standard_normal(len(t))
    dip = depth * np.exp(-0.5 * ((t - center) / sigma) ** 2)
    ear = baseline - dip + noise
    return t, ear


def plot_crossings(
    t: np.ndarray,
    ear: np.ndarray,
    theta: float,
    closing_slope: float,
    opening_slope: float,
    output: Path,
    left_time: float,
    min_time: float,
    min_value: float,
    right_time: float,
) -> None:
    """Plot EAR samples with threshold crossings and slope annotations."""

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(t, ear, s=18, color="#1f77b4", alpha=0.75, label="EAR samples")
    ax.plot(t, ear, lw=2.5, color="#1f77b4", alpha=0.35, label="EAR (line)")

    ax.axhline(theta, color="crimson", lw=2, ls="--", label=f"Threshold θ = {theta:.3f}")
    ax.scatter(
        [left_time, min_time, right_time],
        [theta, min_value, theta],
        s=120,
        color=["crimson", "black", "crimson"],
        alpha=0.5,
        zorder=5,
    )

    ax.plot(
        [left_time, min_time],
        [theta, min_value],
        color="black",
        lw=2.5,
        alpha=0.4,
        label=f"Closing slope = {closing_slope:.2f}",
    )
    ax.plot(
        [min_time, right_time],
        [min_value, theta],
        color="green",
        lw=2.5,
        alpha=0.4,
        label=f"Opening slope = {opening_slope:.2f}",
    )

    # Vertical guides
    ax.axvline(left_time, color="crimson", alpha=0.4)
    ax.axvline(right_time, color="crimson", alpha=0.4)
    ax.axvline(min_time, color="black", alpha=0.3)

    # Annotations (matching the instructional snippet)
    ax.annotate(
        "Left threshold crossing",
        xy=(left_time, theta),
        xytext=(left_time - 0.35, theta + 0.06),
        arrowprops=dict(arrowstyle="->", lw=2, color="crimson"),
        fontsize=11,
        color="crimson",
    )

    ax.annotate(
        "Minimum EAR",
        xy=(min_time, min_value),
        xytext=(min_time + 0.10, min_value - 0.08),
        arrowprops=dict(arrowstyle="->", lw=2),
        fontsize=11,
    )

    ax.annotate(
        "Right threshold crossing",
        xy=(right_time, theta),
        xytext=(right_time + 0.05, theta + 0.06),
        arrowprops=dict(arrowstyle="->", lw=2, color="crimson"),
        fontsize=11,
        color="crimson",
    )

    # Slope labels
    ax.text(
        (left_time + min_time) / 2,
        (theta + min_value) / 2 + 0.01,
        f"closing slope = {closing_slope:.2f} EAR/s",
        bbox=dict(boxstyle="round", fc="white", ec="black"),
        fontsize=11,
    )

    ax.text(
        (min_time + right_time) / 2,
        (theta + min_value) / 2 - 0.04,
        f"opening slope = {opening_slope:.2f} EAR/s",
        bbox=dict(boxstyle="round", fc="white", ec="green"),
        fontsize=11,
        color="green",
    )

    ax.set_title("EAR threshold crossings with slopes")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EAR")
    ax.set_ylim(0.05, 0.36)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Plot saved to {output}")  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--theta",
        type=float,
        default=0.185,
        help="Threshold for EAR crossings (default: 0.185).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ear_threshold_crossing_demo.png"),
        help="Path to save the plot (default: ear_threshold_crossing_demo.png).",
    )
    args = parser.parse_args()

    t, ear = generate_mock_ear()

    triplet = find_threshold_crossing_triplet(
        ear,
        theta=args.theta,
        t=t,
        window=(0, len(ear) - 1),
        max_expansion=int(0.05 * len(ear)),  # allow up to ~5% outward search
        expansion_step=max(1, int(0.01 * len(ear))),
        plateau_policy="midpoint",
    )
    closing_slope, opening_slope = compute_threshold_slopes(triplet, args.theta)

    print(f"Left crossing: {triplet.left.time:.4f}s")  # noqa: T201
    print(f"Minimum: {triplet.minimum_time:.4f}s @ {triplet.minimum_value:.4f}")  # noqa: T201
    print(f"Right crossing: {triplet.right.time:.4f}s")  # noqa: T201
    print(f"Closing slope: {closing_slope:.4f}")  # noqa: T201
    print(f"Opening slope: {opening_slope:.4f}")  # noqa: T201

    plot_crossings(
        t,
        ear,
        theta=args.theta,
        closing_slope=closing_slope,
        opening_slope=opening_slope,
        output=args.output,
        left_time=triplet.left.time,
        min_time=triplet.minimum_time,
        min_value=triplet.minimum_value,
        right_time=triplet.right.time,
    )


if __name__ == "__main__":
    main()
