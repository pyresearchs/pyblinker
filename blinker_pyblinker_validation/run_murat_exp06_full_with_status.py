from __future__ import annotations

import json
import logging
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from blinker_pyblinker_validation.fresh_compare_from_csv import (
    CSV_PATH,
    DATASET_ROOT,
    RESULTS_DIR,
    TOLERANCE_SAMPLES,
    _order_summary,
    _write_experiment_outputs,
    load_selected_recording_ids,
    process_recording,
)
from blinker_pyblinker_validation.stat import build_overall_summary, build_summary_frame


PREFIX = "exp06"
SELECTION = "top"
RECORDING_COUNT = 74
HEARTBEAT_SECONDS = 10
MAX_WORKERS = max(1, min(6, os.cpu_count() or 1))

RUN_LABEL = f"{SELECTION}{RECORDING_COUNT}"
STATUS_JSON_PATH = RESULTS_DIR / f"{PREFIX}_{RUN_LABEL}_live_status.json"
STATUS_MD_PATH = RESULTS_DIR / f"{PREFIX}_{RUN_LABEL}_live_status.md"
LOG_PATH = RESULTS_DIR / f"{PREFIX}_{RUN_LABEL}_live_log.txt"


def _utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _configure_logger() -> logging.Logger:
    RESULTS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("exp06_full_sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    return logger


def _write_status(
    *,
    started_at: str,
    recording_ids: list[str],
    completed: list[dict[str, object]],
    in_progress: list[str],
    failed: list[dict[str, object]],
    logger: logging.Logger,
    finished: bool,
) -> None:
    total = len(recording_ids)
    completed_count = len(completed)
    failed_count = len(failed)
    pending_ids = [
        recording_id
        for recording_id in recording_ids
        if recording_id not in {entry["recording_id"] for entry in completed}
        and recording_id not in set(in_progress)
    ]
    payload = {
        "prefix": PREFIX,
        "dataset": "murat_2018",
        "selection": SELECTION,
        "recording_count": total,
        "started_at": started_at,
        "last_heartbeat": _utc_now(),
        "finished": finished,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "in_progress_count": len(in_progress),
        "pending_count": len(pending_ids),
        "max_workers": MAX_WORKERS,
        "completed": completed,
        "in_progress": in_progress,
        "pending_preview": pending_ids[:10],
        "failed": failed,
        "log_path": str(LOG_PATH),
    }
    STATUS_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf8")

    lines = [
        f"# exp06 top74 live status",
        "",
        f"- started_at: {started_at}",
        f"- last_heartbeat: {payload['last_heartbeat']}",
        f"- finished: {finished}",
        f"- total: {total}",
        f"- completed: {completed_count}",
        f"- in_progress: {len(in_progress)}",
        f"- pending: {len(pending_ids)}",
        f"- failed: {failed_count}",
        f"- max_workers: {MAX_WORKERS}",
        f"- log: {LOG_PATH}",
        "",
        "## In Progress",
    ]
    if in_progress:
        lines.extend(f"- {recording_id}" for recording_id in in_progress)
    else:
        lines.append("- none")

    lines.extend(["", "## Recent Completed"])
    recent_completed = completed[-10:]
    if recent_completed:
        lines.extend(
            f"- {entry['recording_id']}: share={entry['share_within_tolerance_percent']}, status={entry['artifact_status']}"
            for entry in recent_completed
        )
    else:
        lines.append("- none")

    lines.extend(["", "## Failed"])
    if failed:
        lines.extend(
            f"- {entry['recording_id']}: {entry['error']}"
            for entry in failed[-10:]
        )
    else:
        lines.append("- none")

    STATUS_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf8")
    logger.info(
        "heartbeat finished=%s completed=%s in_progress=%s pending=%s failed=%s",
        finished,
        completed_count,
        len(in_progress),
        len(pending_ids),
        failed_count,
    )


def main() -> int:
    logger = _configure_logger()
    recording_ids = load_selected_recording_ids(
        CSV_PATH,
        n_rows=RECORDING_COUNT,
        selection=SELECTION,
    )
    started_at = _utc_now()
    completed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []

    logger.info(
        "starting full sweep prefix=%s dataset_root=%s recordings=%s max_workers=%s force_rerun=%s",
        PREFIX,
        DATASET_ROOT,
        len(recording_ids),
        MAX_WORKERS,
        True,
    )
    _write_status(
        started_at=started_at,
        recording_ids=recording_ids,
        completed=completed,
        in_progress=[],
        failed=failed,
        logger=logger,
        finished=False,
    )

    future_to_recording: dict[object, str] = {}
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for recording_id in recording_ids:
            future = executor.submit(
                process_recording,
                recording_id,
                dataset_root=DATASET_ROOT,
                prefix=PREFIX,
                tolerance_samples=TOLERANCE_SAMPLES,
                plot=False,
                target_share_percent=100.0,
                force_rerun=True,
            )
            future_to_recording[future] = recording_id

        pending_futures = set(future_to_recording)
        while pending_futures:
            done, pending_futures = wait(
                pending_futures,
                timeout=HEARTBEAT_SECONDS,
                return_when=FIRST_COMPLETED,
            )

            in_progress = [future_to_recording[future] for future in pending_futures]

            if not done:
                _write_status(
                    started_at=started_at,
                    recording_ids=recording_ids,
                    completed=completed,
                    in_progress=in_progress,
                    failed=failed,
                    logger=logger,
                    finished=False,
                )
                continue

            for future in done:
                recording_id = future_to_recording[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - operational path
                    failed_entry = {"recording_id": recording_id, "error": repr(exc)}
                    failed.append(failed_entry)
                    logger.exception("recording failed: %s", recording_id)
                else:
                    results.append(result)
                    share = result.comparison.metrics.get("share_within_tolerance_percent")
                    entry = {
                        "recording_id": recording_id,
                        "share_within_tolerance_percent": float(share),
                        "artifact_status": result.artifact_status,
                    }
                    completed.append(entry)
                    logger.info(
                        "completed recording=%s share=%s status=%s",
                        recording_id,
                        share,
                        result.artifact_status,
                    )

            _write_status(
                started_at=started_at,
                recording_ids=recording_ids,
                completed=completed,
                in_progress=in_progress,
                failed=failed,
                logger=logger,
                finished=False,
            )

    ordered_summary = _order_summary(
        build_summary_frame([result.comparison for result in results]),
        recording_ids,
    )
    ordered_summary["artifact_status"] = ordered_summary["recording_id"].map(
        {result.recording_id: result.artifact_status for result in results}
    )
    ordered_summary["result_file"] = f"{PREFIX}_pyblinker_results.pkl"
    ordered_summary["selection"] = SELECTION

    overall = build_overall_summary(ordered_summary)
    summary_path, overall_path, selection_path = _write_experiment_outputs(
        summary=ordered_summary,
        overall=overall,
        prefix=PREFIX,
        n_subjects=len(recording_ids),
        selection=SELECTION,
        recording_ids=recording_ids,
    )
    logger.info("summary csv=%s", summary_path)
    logger.info("summary json=%s", overall_path)
    logger.info("selection csv=%s", selection_path)

    _write_status(
        started_at=started_at,
        recording_ids=recording_ids,
        completed=completed,
        in_progress=[],
        failed=failed,
        logger=logger,
        finished=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
