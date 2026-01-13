"""
Verify consistency of BlinkProperties implementation against a reference snapshot.

This script loads a reference set of blink properties from 'blinkprops_case01.pkl'
and asserts that the current implementation of `BlinkProperties` reproduces
the same values. This ensures that refactoring or changes to `BlinkProperties`
do not inadvertently alter the calculated features, specifically ensuring
compatibility with `pyblinker.blink_features.morphology` and `pyblinker.blink_features.kinematics`.
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_PATH = REPO_ROOT / "tutorial" / "blinkprops_case01.pkl"

from pyblinker.blink_features.waveform_features.extract_blink_properties import (
    BlinkProperties,
)
from pyblinker.utils.blinker_feature import (
    load_blinkprops_pickle,
    replay_and_assert_blinkprops,
)

fx = load_blinkprops_pickle(DATA_PATH)
replay_and_assert_blinkprops(fx, BlinkProperties)
