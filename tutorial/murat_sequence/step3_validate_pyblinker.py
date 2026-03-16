"""Run the canonical Murat validation flow with live status output."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation.run_murat_full_with_status import main


if __name__ == "__main__":
    raise SystemExit(main())
