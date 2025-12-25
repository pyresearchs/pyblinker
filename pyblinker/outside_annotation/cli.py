"""CLI entry point for the refined blink flow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .refined_blink_flow import BlinkRegionRefinementFlow, RefinementConfig


def parse_args() -> RefinementConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("manual_annotation_feature_calculation_data/ear_eog.csv"),
        help="Path to blink region annotations CSV.",
    )
    parser.add_argument(
        "--fif",
        type=Path,
        default=Path("manual_annotation_feature_calculation_data/ear_eog.fif"),
        help="Path to raw FIF file containing the blink channel.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="EEG-E8",
        help="Channel name to analyze.",
    )
    parser.add_argument(
        "--buffer-seconds",
        type=float,
        default=0.25,
        help="Padding (seconds) around annotated regions for zero-crossing search.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manual_annotation_feature_calculation_data/refined_blink_metrics.csv"),
        help="Destination for the refined blink metrics (CSV).",
    )
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help="Skip FitBlinks fitting while still computing baselines.",
    )

    args = parser.parse_args()

    return RefinementConfig(
        annotation_csv=args.annotations,
        fif_path=args.fif,
        channel=args.channel,
        buffer_seconds=args.buffer_seconds,
        output_path=args.output,
        run_fit=not args.no_fit,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = parse_args()
    flow = BlinkRegionRefinementFlow(config)
    flow.run()


if __name__ == "__main__":
    main()
