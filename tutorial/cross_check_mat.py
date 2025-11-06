"""
Manual EEG Annotation Viewer
----------------------------
Loads EEG .mat file, keeps CH1–CH3,
loads manual annotation CSV,
plots for visual inspection,
and optionally re-saves annotations.

Requirements:
    pip install mne scipy numpy pyedflib pyblinker
"""

from pathlib import Path
import csv
import mne
from pyblinker.utils.mat_edf import load_mat_to_mne
from pyblinker.utils.download import download_once

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
SAVE_CSV = True           # Save annotations after viewing
CHANNELS = "1-3"          # Channels to keep (CH1–CH3)
SFREQ = 200.0

# File paths
URL = "https://figshare.com/ndownloader/files/12400409"
MAT_NAME = "CLA-SubjectJ-170510-3St-LRHand-Inter.mat"
DATA_DIR = Path("..")
MAT_PATH = DATA_DIR / MAT_NAME

# <<< Fixed manual annotation file >>>
CSV_PATH = Path(r"/tutorial/CLA-SubjectJ-170510-3St-LRHand-Inter_annotations.csv")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def parse_channels(spec: str) -> list[str]:
	"""Parse '1-3' or '1,2,3' → ['CH1', 'CH2', 'CH3']"""
	spec = spec.strip()
	if "-" in spec:
		a, b = spec.split("-", 1)
		start, end = int(a), int(b)
		if start > end:
			start, end = end, start
		idxs = range(start, end + 1)
	else:
		idxs = [int(x) for x in spec.replace(" ", "").split(",") if x]
	return [f"CH{i}" for i in idxs]


def read_annotations_csv(csv_path: Path) -> mne.Annotations:
	"""Load annotations from a CSV file."""
	with open(csv_path, newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		names = {k.lower(): k for k in (reader.fieldnames or [])}

		def pick(*cands):
			for c in cands:
				if c in names:
					return names[c]
			raise KeyError(f"Missing required column in {csv_path}")

		k_onset = pick("onset_sec", "onset")
		k_dur = pick("duration_sec", "duration")
		k_desc = pick("description", "label")

		on, du, de = [], [], []
		for row in reader:
			on.append(float(row[k_onset]))
			du.append(float(row[k_dur]))
			de.append(str(row[k_desc]))

	return mne.Annotations(onset=on, duration=du, description=de)


def save_annotations_csv(annotations: mne.Annotations, out_csv: Path) -> None:
	"""Save annotations to a CSV file."""
	with open(out_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["onset_sec", "duration_sec", "description"])
		for onset, duration, desc in zip(
				annotations.onset, annotations.duration, annotations.description
				):
			writer.writerow([onset, duration, desc])


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
	# 1️⃣ Ensure EEG data exists
	if not MAT_PATH.exists():
		print(f"[download] {URL} → {MAT_PATH}")
		download_once(URL, MAT_PATH)

	print("[mne] Loading MAT → Raw ...")
	raw = load_mat_to_mne(MAT_PATH.as_posix(), sfreq_default=SFREQ)

	# 2️⃣ Keep selected channels
	keep = parse_channels(CHANNELS)
	to_drop = [ch for ch in raw.ch_names if ch not in keep]
	if to_drop:
		raw = raw.drop_channels(to_drop)
	print(f"[info] Kept channels: {raw.ch_names}")

	# 3️⃣ Load manual annotation CSV (fixed path)
	if CSV_PATH.exists():
		print(f"[csv] Loading manual annotations from: {CSV_PATH}")
		manual = read_annotations_csv(CSV_PATH)
		raw.set_annotations(manual)
		print(f"[csv] Loaded {len(manual)} annotations.")
	else:
		print(f"[csv] File not found: {CSV_PATH}")
		return
	# 382- 357 = 25 annotations difference
	# 4️⃣ Plot for visual cross-check
	print(f"[plot] Showing {len(raw.annotations)} annotations — close window to continue.")
	raw.plot(block=True)

	# 5️⃣ Optionally save (after manual edits in the viewer)
	# if SAVE_CSV:
	# 	save_annotations_csv(raw.annotations, CSV_PATH)
	# 	print(f"[save] Updated annotations saved to: {CSV_PATH.resolve()}")

	print("[done] Visual inspection complete.")


# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------
if __name__ == "__main__":
	main()
