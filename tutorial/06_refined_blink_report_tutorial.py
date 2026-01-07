"""Tutorial: run the refined blink flow, compute metrics, and export an HTML report.

What this tutorial does (high level)
------------------------------------
1. Loads manual blink annotations from ``test/test_files/ear_eog.csv``.
2. Loads the corresponding MNE FIF recording at ``test/test_files/ear_eog_raw.fif``.
3. Runs the blink refinement pipeline (zero-crossings, optional fitting, and properties) on the
   ``EEG-E8`` channel.
4. Writes the merged metrics to ``tutorial_outputs/refined_blink_metrics.csv`` for inspection.
5. Generates an ``tutorial_outputs/refined_blink_report.html`` MNE report that overlays:
   - zero-crossing markers and labels,
   - the maximum-amplitude marker,
   - key blink metrics,
   - and, optionally, an EAR overlay (``EAR-avg_ear``) on a secondary y-axis.

Why this exists
---------------
This tutorial is designed to remain understandable years later: it shows the full path from
raw annotations + FIF to visual validation of refined blinks. By default it mirrors the
original behavior (EEG-only plot). Turning on the EAR overlay is a single-flag change to help
compare eye-aspect-ratio traces to EEG-derived blink regions without altering the core
processing code.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

from pyblinker.outside_annotation import (
    BlinkRegionRefinementFlow,
    RefinementConfig,
    build_refined_blink_report,
)


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "test" / "test_files"
    annotations = data_dir / "ear_eog.csv"
    fif_path = data_dir / "ear_eog_raw.fif"

    output_dir = Path(__file__).resolve().parents[1] / "tutorial_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Toggle EAR overlay for plots. Default is False to preserve existing behavior.
    enable_ear_overlay = False

    config = RefinementConfig(
        annotation_csv=annotations,
        fif_path=fif_path,
        channel="EEG-E8",
        buffer_seconds=0.25,
        output_path=output_dir / "refined_blink_metrics.csv",
        run_fit=True,
    )
    flow = BlinkRegionRefinementFlow(config)
    artifacts = flow.run_with_artifacts()

    # Persist CSV for inspection
    artifacts.results.to_csv(config.output_path, index=False)

    # Build visual report with zero-crossing overlays and key metrics
    report_path = output_dir / "refined_blink_report.html"
    overlay_signal = None
    overlay_sfreq = None
    if enable_ear_overlay:
        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
        overlay_sfreq = float(raw.info["sfreq"])
        try:
            overlay_signal = raw.get_data(picks="EAR-avg_ear")[0]
        except Exception as exc:  # pragma: no cover - defensive channel lookup
            raise ValueError("Channel EAR-avg_ear not found for overlay") from exc

    build_refined_blink_report(
        results=artifacts.results.assign(
            onset__refine__ear=artifacts.results["refined_left_threshold"]
            / artifacts.sfreq,
            duration__refine__ear=(
                artifacts.results["refined_right_threshold"]
                - artifacts.results["refined_left_threshold"]
            )
            / artifacts.sfreq,
            onset__th_interpolation__ear=np.nan,
            duration__th_interpolation__ear=np.nan,
            trough__th_point__ear=np.nan,
        ),
        signal=artifacts.signal,
        sfreq=artifacts.sfreq,
        channel_name=config.channel,
        plot_overlay=enable_ear_overlay,
        overlay_signal=overlay_signal,
        overlay_sfreq=overlay_sfreq,
        output_path=report_path,
    )

    print("Refined blink metrics saved to:", config.output_path)
    print("Blink validation report saved to:", report_path)


if __name__ == "__main__":
    main()
