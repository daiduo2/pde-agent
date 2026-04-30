"""Merge single-task Burgers submission directories into one package."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from pde_agent.core.competition_submission import PROBLEM_ID


STATE_HINT = "COMPETITION_SUBMISSIONS_MERGED"


class CompetitionSubmissionMergeError(RuntimeError):
    pass


def merge_competition_submissions(
    *,
    output_dir: str | Path,
    submission_id: str,
    task_submission_dirs: list[str | Path],
    force: bool = False,
) -> dict[str, Any]:
    if not task_submission_dirs:
        raise CompetitionSubmissionMergeError("at least one task submission dir is required")

    output_root = Path(output_dir)
    submission_dir = output_root / "submission"
    zip_path = output_root / "submission.zip"
    report_path = output_root / "competition_submission_merge_report.json"
    if any(path.exists() for path in (submission_dir, zip_path, report_path)) and not force:
        raise CompetitionSubmissionMergeError(
            f"merged submission artifacts already exist under {output_root}"
        )
    if force and submission_dir.exists():
        shutil.rmtree(submission_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    task_ids: list[int] = []
    source_dirs = [Path(path) for path in task_submission_dirs]
    for source_dir in source_dirs:
        task_id = _detect_task_id(source_dir)
        if task_id in task_ids:
            raise CompetitionSubmissionMergeError(f"duplicate task submission: task{task_id}")
        _copy_task_files(source_dir=source_dir, submission_dir=submission_dir, task_id=task_id)
        task_ids.append(task_id)

    _copy_code_dir(source_dirs=source_dirs, submission_dir=submission_dir)
    _write_submission_json(submission_dir=submission_dir, submission_id=submission_id)
    _write_zip(submission_dir=submission_dir, zip_path=zip_path)

    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "submission_id": submission_id,
        "task_ids": sorted(task_ids),
        "source_submission_dirs": [str(path) for path in source_dirs],
        "artifacts": {
            "submission_dir": str(submission_dir),
            "submission_zip_path": str(zip_path),
            "merge_report_path": str(report_path),
        },
        "next_command": "judge-hermes-submission",
    }
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _detect_task_id(source_dir: Path) -> int:
    if not source_dir.is_dir():
        raise CompetitionSubmissionMergeError(f"missing submission dir: {source_dir}")
    matches = [
        task_id
        for task_id in (1, 2)
        if (source_dir / f"task{task_id}_pred.hdf5").is_file()
        or (source_dir / f"task{task_id}_time.csv").is_file()
        or (source_dir / f"task{task_id}_logs.log").is_file()
    ]
    if len(matches) != 1:
        raise CompetitionSubmissionMergeError(
            f"expected exactly one task in {source_dir}, found {matches}"
        )
    task_id = matches[0]
    required = [
        source_dir / f"task{task_id}_pred.hdf5",
        source_dir / f"task{task_id}_time.csv",
        source_dir / f"task{task_id}_logs.log",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise CompetitionSubmissionMergeError(
            f"task{task_id} submission missing files: {', '.join(missing)}"
        )
    return task_id


def _copy_task_files(*, source_dir: Path, submission_dir: Path, task_id: int) -> None:
    for suffix in ("pred.hdf5", "time.csv", "logs.log"):
        source = source_dir / f"task{task_id}_{suffix}"
        destination = submission_dir / source.name
        shutil.copy2(source, destination)


def _copy_code_dir(*, source_dirs: list[Path], submission_dir: Path) -> None:
    destination = submission_dir / "code"
    for source_dir in source_dirs:
        source = source_dir / "code"
        if source.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
            return
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "train.py").write_text("## PUT YOUR CODE HERE\n", encoding="utf-8")
    (destination / "model.py").write_text("## PUT YOUR CODE HERE\n", encoding="utf-8")


def _write_submission_json(*, submission_dir: Path, submission_id: str) -> None:
    payload = {
        "submission_id": submission_id,
        "problem_id": PROBLEM_ID,
        "code_path": "code",
    }
    (submission_dir / "submission.json").write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_zip(*, submission_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(submission_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(submission_dir.parent))
