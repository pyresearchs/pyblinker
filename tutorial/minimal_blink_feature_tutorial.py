#!/usr/bin/env python3
"""
Minimal tutorial: load raw FIF, epoch, extract blink features, save as Excel (or CSV).

This example mirrors the workflow exercised in the aggregate blink feature tests.
"""

from datetime import datetime
from pathlib import Path

import mne

from pyblinker.blink_features.blink_events.event_features.aggregate import (
    aggregate_blink_features,
)
from pyblinker.utils.refinement_utils import slice_raw_into_mne_epochs_refine_annot


# --- User settings ---
# Resolve project directories relative to this script so it can be executed from
# any working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEST_FILES_DIR = PROJECT_ROOT / "test" / "test_files"

RAW_PATH = TEST_FILES_DIR / "ear_eog_raw.fif"   # input FIF file
EPOCH_LEN = 30.0                                # seconds per epoch
CSV_META = TEST_FILES_DIR / "ear_eog_blink_count_epoch.csv"  # optional metadata
OUT_FILE = Path("blink_features.xlsx")          # base output Excel file name
INCLUDE_MODALITIES = ("EEG", "EOG", "EAR")           # which modalities
FEATURE_FAMILIES = (
    "events",
    "energy",
    "freq",
    "kin",
    "morph",
    "wave",
)  # feature families


# --- 1) Load Raw recording ---
print(f"Reading raw FIF from: {RAW_PATH}")
raw = mne.io.read_raw_fif(RAW_PATH, preload=True, verbose=False)

# --- 2) Slice into epochs ---
print(f"Slicing into epochs of {EPOCH_LEN:.1f} s...")
epochs = slice_raw_into_mne_epochs_refine_annot(
    raw,
    epoch_len=EPOCH_LEN,
    blink_label=None,
    progress_bar=False,
)
print(f"Created {len(epochs)} epochs.")

# --- 3) Aggregate blink features ---
print("Computing blink features...")
df = aggregate_blink_features(
    epochs,
    epoch_len=EPOCH_LEN,
    blink_label=None,
    progress_bar=False,
    include_modalities=INCLUDE_MODALITIES,
    feature_families=FEATURE_FAMILIES,
    metadata_csv_path=CSV_META,
)

# --- 4) Save to Excel (or fallback to CSV if needed) ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
timestamped_out = OUT_FILE.with_name(
    f"{OUT_FILE.stem}_{timestamp}{OUT_FILE.suffix}"
)

try:
    df.to_excel(timestamped_out, index=False)
    print(f"Saved features to {timestamped_out.resolve()}")
except ModuleNotFoundError:
    csv_fallback = timestamped_out.with_suffix(".csv")
    df.to_csv(csv_fallback, index=False)
    print(f"openpyxl not found, saved CSV instead: {csv_fallback.resolve()}")

# --- Quick summary ---
print(f"Rows (epochs): {len(df)} | Columns (features): {len(df.columns)}")
print("First few columns:", list(df.columns[:8]))
