from __future__ import annotations

import numpy as np
import pandas as pd

from pyblinker.utils.evaluation import blink_comparison


def test_compare_pairs_overlap_even_when_amplitude_differs():
    detected = pd.DataFrame({"start_blink": [100], "end_blink": [120]})
    ground_truth = pd.DataFrame({"start_blink": [101], "end_blink": [121]})

    signal = np.zeros(200, dtype=float)
    signal[120] = 1.0
    signal[121] = 2.0

    result = blink_comparison.compare_detected_vs_ground_truth(
        detected,
        ground_truth,
        sampling_rate_hz=200.0,
        tolerance_samples=20,
        n_preview_rows=1,
        n_diff_rows=10,
        detected_signal=signal,
    )

    assert result.metrics["ground_truth_only"] == 0.0
    assert result.metrics["detected_only"] == 0.0
    assert result.metrics["matches_within_tolerance"] == 2.0
    assert result.metrics["share_within_tolerance"] == 0.0

    diff_row = result.diff_table.iloc[0]
    assert diff_row["match_category"] == "matches_within_tolerance"
    assert bool(diff_row["within_tolerance"]) is True
