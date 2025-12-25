"""Tutorial: run the refined blink flow and generate an HTML report.

This tutorial uses the provided manual annotations and FIF recording:

- CSV: manual_annotation_feature_calculation_data/ear_eog.csv
- FIF: manual_annotation_feature_calculation_data/ear_eog.fif

It runs the refinement flow, writes the merged metrics to CSV, and produces
an MNE `report.html` that overlays zero-crossings and key blink metrics for
visual validation.
"""

from __future__ import annotations

from pathlib import Path

import mne

from pyblinker.outside_annotation import (
    BlinkRegionRefinementFlow,
    RefinementConfig,
    build_refined_blink_report,
)


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "manual_annotation_feature_calculation_data"
    annotations = data_dir / "ear_eog.csv"
    fif_path = data_dir / "ear_eog.fif"

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
        results=artifacts.results,
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
