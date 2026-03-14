from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import pandas as pd

if __package__:
    from .blink_compare import process_recording_comparison
else:
    from blink_compare import process_recording_comparison

if TYPE_CHECKING:
    if __package__:
        from .stat import RecordingComparison
    else:
        from blink_compare import RecordingComparison

CSV_PATH = Path(__file__).with_name("summary_metrics.csv")
DATASET_ROOT = Path("D:/dataset/murat_2018")

TOLERANCE_SAMPLES = 20
OVERWRITE = False


def load_recording_ids(csv_path: Path = CSV_PATH, *,
					   n_rows: int) -> list[str]:
    summary = pd.read_csv(csv_path, dtype={"recording_id": "string"})
    if "recording_id" not in summary.columns:
        raise KeyError(f"Column 'recording_id' not found in {csv_path}")

    recording_ids = (
        summary["recording_id"]
        .dropna()
        .astype("string")
        .str.strip()
        .head(n_rows)
        .tolist()
    )
    return [recording_id for recording_id in recording_ids if recording_id]


def compare_first_rows(
    n_rows: int,
    csv_path: Path = CSV_PATH,
    *,
    dataset_root: Path = DATASET_ROOT,

    tolerance_samples: int = TOLERANCE_SAMPLES,
    overwrite: bool = OVERWRITE,
) -> list[RecordingComparison]:
    comparisons: list[RecordingComparison] = []

    for recording_id in load_recording_ids(csv_path, n_rows=n_rows):
        recording_dir = dataset_root / recording_id
        py_path = recording_dir / "pyblinker_results.pkl"
        blinker_path = recording_dir / "blinker_results.pkl"
        fif_path = recording_dir / f"{recording_id}.fif"

        comparisons.append(
            process_recording_comparison(
                recording_dir,
                py_path,
                blinker_path,
                fif_path,
                recording_id,
                tolerance_samples=tolerance_samples,
                overwrite=overwrite,
            )
        )

    return comparisons


def main() -> int:
    N_ROWS = 9
    res=compare_first_rows(N_ROWS)
    print(res)



if __name__ == "__main__":
    raise SystemExit(main())

