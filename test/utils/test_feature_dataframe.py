"""Tests for :mod:`pyblinker.utils.feature_dataframe`."""

import pandas as pd
import pytest

from pyblinker.utils.feature_dataframe import to_epoch_indexed


def test_to_epoch_indexed_with_subject_column() -> None:
    df = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2"],
            "epoch": [1, 0, 2],
            "value": [0.1, 0.2, 0.3],
        }
    )

    result = to_epoch_indexed(df)

    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["subject", "epoch"]
    assert list(result.index) == [("S1", 0), ("S1", 1), ("S2", 2)]
    assert result.loc[("S1", 0), "value"] == pytest.approx(0.2)


def test_to_epoch_indexed_without_subject_column() -> None:
    df = pd.DataFrame({"epoch": [2, 0, 1], "value": [3.0, 1.0, 2.0]})

    result = to_epoch_indexed(df, subject_column=None)

    assert isinstance(result.index, pd.Index)
    assert result.index.name == "epoch"
    assert list(result.index) == [0, 1, 2]
    assert result.loc[2, "value"] == pytest.approx(3.0)


def test_to_epoch_indexed_raises_when_missing_epoch() -> None:
    df = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(KeyError):
        to_epoch_indexed(df)
