"""Deprecated shim for grouping refined blink metadata."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

from .dict_utils import group_by_key

warnings.warn(
    "pyblinker.utils.blink_refinement_helpers is deprecated; use pyblinker.utils.dict_utils.group_by_key instead.",
    DeprecationWarning,
    stacklevel=2,
)


def group_refined_by_epoch(refined: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Return ``refined`` grouped by ``epoch_index``."""

    grouped = group_by_key(refined, "epoch_index")
    return {int(epoch): [dict(item) for item in items] for epoch, items in grouped.items()}


__all__ = ["group_refined_by_epoch"]
