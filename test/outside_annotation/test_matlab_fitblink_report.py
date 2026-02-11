from __future__ import annotations

import numpy as np
import pandas as pd

from pyblinker.outside_annotation.matlab_fitblink_report import build_fitblink_report


def test_build_fitblink_report_adds_figures(tmp_path):
    signal = np.linspace(-1.0, 1.0, 50)
    blink_positions = pd.DataFrame(
        {
            "start_blink": [10.0, 30.0],
            "end_blink": [20.0, 40.0],
        }
    )
    fit_results = pd.DataFrame(
        [
            {
                "max_blink": 15.0,
                "max_value": 0.25,
                "left_zero": 12.0,
                "right_zero": 18.0,
                "left_base": 11.0,
                "right_base": 19.0,
                "left_base_half_height": 13.0,
                "right_base_half_height": 17.0,
                "left_zero_half_height": 14.0,
                "right_zero_half_height": 16.0,
                "left_range": [12.0, 15.0],
                "right_range": [15.0, 18.0],
                "left_slope": 0.1,
                "right_slope": -0.1,
                "aver_left_velocity": 0.2,
                "aver_right_velocity": -0.2,
                "leftR2": 0.9,
                "rightR2": 0.8,
                "x_intersect": 15.0,
                "y_intersect": 0.3,
                "left_x_intercept": 9.0,
                "right_x_intercept": 21.0,
            },
            {
                "max_blink": 35.0,
                "max_value": 0.75,
                "left_zero": 32.0,
                "right_zero": 38.0,
                "left_base": 31.0,
                "right_base": 39.0,
                "left_base_half_height": 33.0,
                "right_base_half_height": 37.0,
                "left_zero_half_height": 34.0,
                "right_zero_half_height": 36.0,
                "left_range": [32.0, 35.0],
                "right_range": [35.0, 38.0],
                "left_slope": 0.12,
                "right_slope": -0.12,
                "aver_left_velocity": 0.22,
                "aver_right_velocity": -0.22,
                "leftR2": 0.92,
                "rightR2": 0.82,
                "x_intersect": 35.0,
                "y_intersect": 0.76,
                "left_x_intercept": 29.0,
                "right_x_intercept": 41.0,
            },
        ]
    )
    report = build_fitblink_report(
        title="MATLAB output report",
        signal=signal,
        blink_positions=blink_positions,
        fit_results=fit_results,
        section_label="unit-test",
    )

    output_path = tmp_path / "report.html"
    report.save(output_path, overwrite=True, open_browser=False)
    assert output_path.exists()
