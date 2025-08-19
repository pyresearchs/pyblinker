"""Refine blink events for EOG signal using prepared EEG/EOG segments."""
import logging
from pathlib import Path

import mne

from pyblinker.utils.epochs import slice_raw_into_epochs, EPOCH_LEN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CHANNEL = "EOG-EEG-eog_vert_left"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "unit_test" / "features" / "ear_eog_raw.fif"

def main() -> None:
    logger.info("Preparing segments from: %s", RAW_PATH)



    raw = mne.io.read_raw_fif(RAW_PATH, preload=False, verbose=False)
    if len(raw.annotations) == 0:
        raise ValueError("Raw recording has no annotations to refine")

    segments, _, _, _ = slice_raw_into_epochs(raw, epoch_len=EPOCH_LEN)



if __name__ == "__main__":
    main()
