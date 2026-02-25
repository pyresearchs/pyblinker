"""Helper utilities for building expected feature column names."""

from __future__ import annotations

from typing import Iterable, Dict, List


def build_expected_metrics(
    *,
    landmark: str | Iterable[str],
    metrics: Iterable[str],
    stats: Iterable[str],
    modality: str,
    feature: str,
    channel: str,
) -> Dict[str, Dict[str, List[str]]]:
    """Build expected metric column names in the feature schema."""
    if isinstance(landmark, str):
        landmarks = [landmark]
    else:
        landmarks = list(landmark)

    return {
        lm: {
            metric: [
                f"{modality}__{lm}__{feature}__{metric}_{stat}__{channel}"
                for stat in stats
            ]
            for metric in metrics
        }
        for lm in landmarks
    }
