"""Manual EEG annotation viewer for MAT recordings."""

from __future__ import annotations

from pathlib import Path

from tutorial.utils.mat_data import (
    ensure_mat_file,
    load_raw_from_mat,
    parse_channel_spec,
    read_annotations_as_mne,
)
from tutorial.utils.pathing import ensure_repo_on_path

SAVE_CSV = True
CHANNELS = "1-3"
SFREQ = 200.0
URL = "https://figshare.com/ndownloader/files/12400409"
MAT_NAME = "CLA-SubjectJ-170510-3St-LRHand-Inter.mat"
DATA_DIR = Path("..")
MAT_PATH = DATA_DIR / MAT_NAME
CSV_PATH = Path(r"/tutorial/CLA-SubjectJ-170510-3St-LRHand-Inter_annotations.csv")


def main() -> None:
    ensure_repo_on_path()

    ensure_mat_file(MAT_PATH, URL)
    print("[mne] Loading MAT → Raw ...")
    raw = load_raw_from_mat(MAT_PATH, SFREQ)

    keep = parse_channel_spec(CHANNELS)
    to_drop = [ch for ch in raw.ch_names if ch not in keep]
    if to_drop:
        raw = raw.drop_channels(to_drop)
    print(f"[info] Kept channels: {raw.ch_names}")

    if CSV_PATH.exists():
        print(f"[csv] Loading manual annotations from: {CSV_PATH}")
        manual = read_annotations_as_mne(CSV_PATH)
        raw.set_annotations(manual)
        print(f"[csv] Loaded {len(manual)} annotations.")
    else:
        print(f"[csv] File not found: {CSV_PATH}")
        return

    print(f"[plot] Showing {len(raw.annotations)} annotations — close window to continue.")
    raw.plot(block=True)

    # if SAVE_CSV:
    #     save_annotations_csv(raw.annotations, CSV_PATH)
    #     print(f"[save] Updated annotations saved to: {CSV_PATH.resolve()}")

    print("[done] Visual inspection complete.")


if __name__ == "__main__":
    main()
