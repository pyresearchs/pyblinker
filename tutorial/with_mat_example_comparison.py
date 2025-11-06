"""
Simple Blink Detection Tutorial
-------------------------------
1. Downloads EEG .mat file if not found
2. Converts it to MNE Raw
3. Saves EDF and FIF (cached)
4. Runs blink detection on selected channels
5. Plots annotated EEG with blink markers

Requirements:
    pip install mne scipy numpy pyedflib pyblinker
"""

from pathlib import Path

from pyblinker.utils.mat_edf import load_mat_to_mne
from pyblinker.utils.download import download_once
# def read_annotations_csv(csv_path: Path) -> mne.Annotations:
#     """Load annotations from a CSV file."""
#     with open(csv_path, newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         names = {k.lower(): k for k in (reader.fieldnames or [])}
#
#         def pick(*cands):
#             for c in cands:
#                 if c in names:
#                     return names[c]
#             raise KeyError(f"Missing required column in {csv_path}")
#
#         k_onset = pick("onset_sec", "onset")
#         k_dur = pick("duration_sec", "duration")
#         k_desc = pick("description", "label")
#
#         on, du, de = [], [], []
#         for row in reader:
#             on.append(float(row[k_onset]))
#             du.append(float(row[k_dur]))
#             de.append(str(row[k_desc]))
#
#     return mne.Annotations(onset=on, duration=du, description=de)

def main():
    url = "https://figshare.com/ndownloader/files/12400409"
    mat_name = "CLA-SubjectJ-170510-3St-LRHand-Inter.mat"
    edf_name = "CLA-SubjectJ-170510-3St-LRHand-Inter.edf"
    fif_name = "CLA-SubjectJ-170510-3St-LRHand-Inter-raw.fif"
    sfreq = 200.0

    data_dir = Path(".")
    mat_path = data_dir / mat_name
    edf_path = data_dir / edf_name
    fif_path = data_dir / fif_name

    # 1️⃣ Load cached data if available
    if mat_path.exists():
        print(f"Loading cached Raw: {fif_path}")
        raw = load_mat_to_mne(mat_path.as_posix(), sfreq_default=sfreq)
    else:
        # 2️⃣ Otherwise, download + convert
        download_once(url, mat_path)
        raw = load_mat_to_mne(mat_path.as_posix(), sfreq_default=sfreq)
        # from pyblinker.utils.download import save_edf_once
        # save_edf_once(raw, edf_path)

    # -----------------------------------------------------------------
    # 3️⃣ Blink Detection
    # -----------------------------------------------------------------
    from pyblinker.blinker.pyblinker import BlinkDetector

    # Keep only the first few channels for simplicity
    drange = [f"CH{i}" for i in range(1, 4)]  # CH1–CH3
    to_drop = list(set(raw.ch_names) - set(drange))
    raw = raw.drop_channels(to_drop)

    # Run BlinkDetector
    print("Running BlinkDetector ...")
    detector = BlinkDetector(
        raw,
        visualize=False,
        annot_label="eye_blink",  # provide explicit label (str)
        filter_low=0.5,
        filter_high=30.0,
        resample_rate=int(sfreq),  # ensure int
        n_jobs=2,
        use_multiprocessing=True,
    )
    annot, ch, num_good, blink_details, fig_data, ch_selected = detector.get_blink()

    # Add annotations & plot for manual editing
    raw.set_annotations(annot)
    raw.plot(
        block=True,
        title=f"Eye close based on channel {ch} with length {len(raw.annotations)}",
        # scalings=10e-6,  # ±10 µV
    )
    # <<< Fixed manual annotation file >>>
    CSV_PATH = Path(r"tutorial\CLA-SubjectJ-170510-3St-LRHand-Inter_annotations.csv")
    if CSV_PATH.exists():
        print(f"[csv] Loading manual annotations from: {CSV_PATH}")
        manual = read_annotations_csv(CSV_PATH)
        # raw.set_annotations(manual)
        print(f"[csv] Loaded {len(manual)} annotations.")
    else:
        print(f"[csv] File not found: {CSV_PATH}")
        # return
    # Lets compare with the manual annotations
    # By right,the auto-detected annotations should be 357
    # wheras the manual annotations are 382
     # 382- 357 = 25 annotations difference
    print(f"[info] Auto-detected annotations: {len(raw.annotations)}")
    # 357
    from pyblinker.utils.evaluation import reporting, similarity

    # Generate variable detected_df from the blink_details, extract the column start_blink and end_blink
    import pandas as pd
    similarity.validate_event_table(detected_df)
    # For the manual annotations, there are 2 column, blink onset and duration, use this to create the surrogate start_blink and end_blink
    similarity.validate_event_table(manual)
# ---------------------------------------------------------------------
# 5. Run script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
