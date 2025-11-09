"""Tutorial demonstrating epoching and blink validation report generation."""


# 1) Imports
import logging
from pathlib import Path
import mne
import pandas as pd
import numpy as np
from pyblinker.utils.refinement_utils import slice_raw_into_mne_epochs_refine_annot
from pyblinker.utils.report_utils import add_blink_plots_to_report

# 2) Basic logging setup (prints progress info to the console)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 3) Configuration — adjust these if needed
#    Root folder assumed two levels above this file (…/project/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#    Input files (raw FIF with annotations, and CSV with ground-truth per-epoch counts)
RAW_FIF_PATH = PROJECT_ROOT / "test" / "test_files" / "ear_eog_raw.fif"
CSV_GT_PATH = PROJECT_ROOT / "test" / "test_files" / "ear_eog_blink_count_epoch.csv"

#    Epoching and validation parameters
EPOCH_LEN_SECONDS = 30.0                 # length of each epoch
PROGRESS_BAR = True                      # show progress during processing
ALLOWED_EXCEPTION_ROWS = {31, 55}        # epoch indices to ignore in validation

#    Report generation parameters
PAD_PRE_SECONDS = 0.5                    # time before blink to include in report plots
PAD_POST_SECONDS = 0.5                   # time after blink to include in report plots
LIMIT_PER_EPOCH = None                   # limit number of blinks per epoch in report (None = no limit)
DECIM = 2                                # downsampling factor for report figures
INCLUDE_MODALITIES = ("eeg", "eog", "ear")
REPORT_OUT_PATH = Path("blink_validation_report.html")

# 4) Load the raw recording
logger.info("Loading raw data from: %s", RAW_FIF_PATH)
raw = mne.io.read_raw_fif(RAW_FIF_PATH, preload=True, verbose=False)

# 5) Create epochs and refine blink annotations
#    Treat all annotations as blink candidates (demo file doesn't label them "blink").
logger.info("Slicing into %.1f s epochs and refining annotations...", EPOCH_LEN_SECONDS)
epochs = slice_raw_into_mne_epochs_refine_annot(
    raw,
    epoch_len=EPOCH_LEN_SECONDS,
    blink_label=None,
    progress_bar=PROGRESS_BAR,
    )
logger.info("Created %d epochs", len(epochs))

# 6) Load ground-truth blink counts CSV and prepare metadata for comparison
logger.info("Loading ground-truth counts from: %s", CSV_GT_PATH)
blink_counts_df = pd.read_csv(CSV_GT_PATH)
metadata = epochs.metadata.copy()
metadata["epoch_id"] = metadata.index
merged = metadata.merge(blink_counts_df, on="epoch_id", how="left")

# 7) Validate blink counts per epoch (allowing specified exceptions)
#    If 'blink_onset_extremum_ear' is NaN -> 0 blinks; otherwise use the list length.
logger.info("Validating per-epoch blink counts (exceptions: %s)", sorted(ALLOWED_EXCEPTION_ROWS))
for idx, row in merged.iterrows():
    if idx in ALLOWED_EXCEPTION_ROWS:
        continue

    blink_count = row["blink_count"]
    values = row["blink_onset_extremum_ear"]

    # Convert metadata value to a count: NaN -> 0, list -> len(list)
    length = 0 if (isinstance(values, float) and np.isnan(values)) else len(values)
    assert blink_count == length, (
            f"Mismatch at row {idx}: blink_count={blink_count}, length={length}"
    )

logger.info("Blink counts in metadata align with CSV")

# 8) Build an HTML report with blink-centered plots
logger.info("Generating blink report...")
report = add_blink_plots_to_report(
    epochs,
    pad_pre=PAD_PRE_SECONDS,
    pad_post=PAD_POST_SECONDS,
    limit_per_epoch=LIMIT_PER_EPOCH,
    decim=DECIM,
    include_modalities=INCLUDE_MODALITIES,
    progress_bar=PROGRESS_BAR,
    )

# 9) Save the report to disk
report.save(REPORT_OUT_PATH, overwrite=True)
logger.info("Saved blink report to %s", REPORT_OUT_PATH)

# 10) Done
print("Processing complete. Report saved to:", REPORT_OUT_PATH)
