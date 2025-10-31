import argparse
import os
import sys
from typing import Optional

import mne


DEFAULT_EDF = os.path.join(
    os.path.dirname(__file__),
    "test",
    "test_files",
    "mne_sample_audvis_raw.edf",
)


def read_and_plot(
    edf_path: str,
    *,
    block: bool = True,
    show: bool = True,
    duration: float = 10.0,
    n_channels: int = 30,
    scalings: str | dict = "auto",
) -> None:
    """
    Read an EDF file with MNE and open the interactive browser.

    Parameters
    - edf_path: path to .edf file
    - block: block execution until the browser is closed
    - show: whether to actually show the browser (set False for headless/CI)
    - duration: time window (s) per view
    - n_channels: number of channels to display
    - scalings: scaling for channels ("auto" or dict)
    """
    if not os.path.isfile(edf_path):
        raise FileNotFoundError(f"EDF not found: {edf_path}")

    # Read EDF; preload=False keeps memory small and works fine for plotting
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

    # Print a short summary so it's useful when running headless
    info = raw.info
    print(
        f"Loaded EDF: {edf_path}\n"
        f"- Duration: {raw.times[-1]:.2f} s\n"
        f"- Sampling freq: {info['sfreq']} Hz\n"
        f"- Channels: {info['nchan']}\n"
        f"- Montage: {info.get('dig') is not None}"
    )

    # Only attempt to open the browser when requested
    if show:
        raw.plot(
            block=block,
            duration=duration,
            n_channels=n_channels,
            scalings=scalings,
        )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read and plot an EDF file with MNE")
    p.add_argument(
        "edf",
        nargs="?",
        default=DEFAULT_EDF,
        help="Path to EDF file (default: tests sample)",
    )
    p.add_argument("--no-block", dest="block", action="store_false", help="Don't block on the viewer")
    p.add_argument("--no-show", dest="show", action="store_false", help="Don't show the viewer (headless)")
    p.add_argument("--duration", type=float, default=10.0, help="Time window per view in seconds")
    p.add_argument("--n-ch", type=int, default=30, help="Number of channels to display")
    return p.parse_args(argv)


if __name__ == "__main__":
    ns = _parse_args(sys.argv[1:])
    try:
        read_and_plot(
            ns.edf,
            block=ns.block,
            show=ns.show,
            duration=ns.duration,
            n_channels=ns.n_ch,
        )
    except Exception as e:
        # Provide a clear error and exit non-zero for easy debugging/CI
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

