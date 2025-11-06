"""CLI helpers for ``tutorial/using_mat.py``."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the MAT tutorial."""

    parser = argparse.ArgumentParser(description="Run BlinkDetector on a MAT recording")
    parser.add_argument("mat_path", type=Path, help="Path to the MATLAB file containing EEG data")
    parser.add_argument(
        "--channel-prefix",
        default="CH",
        help="Prefix used for channel names when selecting a subset (default: %(default)s)",
    )
    parser.add_argument(
        "--channel-count",
        type=int,
        default=4,
        help="Number of channels with the given prefix to retain before running BlinkDetector",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the resulting Raw object with blink annotations",
    )
    return parser.parse_args()
