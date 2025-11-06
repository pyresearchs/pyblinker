"""Helpers for locating the repository root from tutorial scripts."""

from __future__ import annotations

from pathlib import Path
import sys


def find_repo_root() -> Path:
    """Return the repository root directory by locating ``pyproject.toml``."""

    current = Path(__file__).resolve()
    for candidate in (current,) + tuple(current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate repository root relative to this file")


def ensure_repo_on_path() -> Path:
    """Insert the repository root into ``sys.path`` for tutorial imports."""

    repo_root = find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root
