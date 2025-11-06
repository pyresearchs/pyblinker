"""Load a MATLAB file into MNE and run :class:`BlinkDetector` on selected channels."""

from __future__ import annotations

from pyblinker.blinker.pyblinker import BlinkDetector
from pyblinker.utils.mat_edf import load_mat_to_mne

from tutorial.utils.using_mat import parse_args


def main() -> None:
    args = parse_args()
    raw = load_mat_to_mne(str(args.mat_path))
    sfreq = float(raw.info["sfreq"])

    if args.channel_count > 0:
        selected = [f"{args.channel_prefix}{idx}" for idx in range(args.channel_count)]
        available = [ch for ch in selected if ch in raw.ch_names]
        if available:
            to_drop = set(raw.ch_names) - set(available)
            if to_drop:
                raw.drop_channels(sorted(to_drop))

    detector = BlinkDetector(
        raw,
        visualize=False,
        annot_label=None,
        filter_low=0.5,
        filter_high=30.0,
        resample_rate=sfreq,
        n_jobs=2,
        use_multiprocessing=True,
    )

    annot, channel, blink_count, _, _, selected_channel = detector.get_blink()
    raw.set_annotations(annot)

    print(f"Detected {blink_count} eye-closure events on channel {channel} (selected {selected_channel}).")
    if args.plot:
        raw.plot(block=True, title=f"Eye closures based on channel {channel}", scalings=10e-6)


if __name__ == "__main__":
    main()
