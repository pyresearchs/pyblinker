"""Tutorial: Segmenting Raw Ear/EOG Data and Verifying Blink Counts.

This tutorial assume we already find the blinks location either via BLINKER (via eeg) approach,
manually (via mne annotations), or other algorithms.
In this example, we are using the ear_eog_raw.fif, which being manually annotated.


This tutorial mirrors the logic in
``test/utils/test_slice_raw_into_mne_epochs.py``. It demonstrates how to
slice a continuous recording into fixed-length epochs, integrate blink
annotations into each epoch's metadata, and validate the resulting per-epoch
blink counts against a ground truth file.

Flowchart:

1. **Load raw data** from ``ear_eog_raw.fif``.
2. **Segment** the continuous signal into 30 s epochs with
   ``slice_raw_into_mne_epochs``.
3. **Inspect metadata** to confirm ``blink_onset`` and ``blink_duration`` fields
   align with the original annotations.
4. **Count blinks per epoch** and compare totals with
   ``ear_eog_blink_count_epoch.csv``.

The output is an :class:`mne.Epochs` object ready for downstream analyses.
"""


# 1) Imports
import logging
from pathlib import Path
import mne
import pandas as pd
from pyblinker.utils import slice_raw_into_mne_epochs

# 2) Basic logging setup (prints helpful progress/info)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 3) Configuration: project paths (adjust if your files live elsewhere)
#    PROJECT_ROOT is two folders above this file (…/project/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 4) Input file paths (raw FIF and ground-truth blink counts CSV)
raw_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"          # <- EEG/EOG recording with manual blink annotations
gt_path = PROJECT_ROOT / "test" / "test_files" / "ear_eog_blink_count_epoch.csv"  # <- Ground-truth per-epoch blink counts

# 5) Load the raw recording
print(f"Loading raw data from: {raw_path}")
raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)

# 6) Slice the continuous recording into fixed-length epochs (30 s)
#    Also integrates blink annotations into the epochs' metadata.
print("Slicing raw into 30 s epochs and attaching blink metadata...")
epochs = slice_raw_into_mne_epochs(
    raw,
    epoch_len=30.0,        # length of each epoch in seconds
    blink_label=None,      # use whatever blink labels exist in raw annotations
    progress_bar=False
    )
logger.info("Created %d epochs", len(epochs))

# 7) Get the resulting metadata (must exist after slicing)
metadata = epochs.metadata

# 8) Inspect one example annotation and show which epoch it maps to
#    (mirrors the tutorial/test logic)
annotation = raw.annotations[2]
epoch_idx = int(annotation["onset"] // 30.0)
logger.info(
    "Annotation at %.2fs mapped to epoch %d", annotation["onset"], epoch_idx
    )
# Print the blink fields for that epoch
print(metadata.loc[epoch_idx, ["blink_onset", "blink_duration"]])

# 9) Count blinks per epoch based on metadata["blink_onset"]
#    Rules:
#      - list -> count = len(list)
#      - NaN  -> count = 0
#      - else -> count = 1
print("Counting blinks per epoch...")
counts = []
for onset in metadata["blink_onset"]:
    if isinstance(onset, list):
        counts.append(len(onset))
    elif pd.isna(onset):
        counts.append(0)
    else:
        counts.append(1)

# 10) Load ground-truth counts and compare
print(f"Loading ground-truth counts from: {gt_path}")
gt_df = pd.read_csv(gt_path).iloc[: len(counts)]

# 11) Validate totals and per-epoch counts against ground truth (assertions)
# Total count match
assert sum(counts) == int(gt_df["blink_count"].sum())
# Per-epoch match
for epoch_id, count in enumerate(counts):
    assert count == int(gt_df.loc[epoch_id, "blink_count"])

# 12) Done
logger.info("Blink counts validated for %d epochs", len(counts))
print("All checks passed. Epochs object is ready for downstream analysis.")
