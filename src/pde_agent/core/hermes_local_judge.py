"""Local judge for Hermes AI4S Burgers competition submissions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pde_agent.core.competition_scoring import score_competition_prediction
from pde_agent.core.competition_submission_audit import audit_competition_submission


STATE_HINT = "HERMES_LOCAL_JUDGE_SCORED"
INFERENCE_TIMEOUT_SECONDS = 120.0
TASK2_TRAIN_BUDGET_SECONDS = 12.0 * 60.0 * 60.0


class HermesLocalJudgeError(RuntimeError):
    pass


def judge_hermes_submission(
    *,
    submission_dir: str | Path,
    output_path: str | Path | None,
    task1_test_path: str | Path | None = None,
    task1_target_path: str | Path | None = None,
    task2_test_path: str | Path | None = None,
    task2_target_path: str | Path | None = None,
    dataset_key: str = "tensor",
    prediction_dataset_key: str = "tensor",
) -> dict[str, Any]:
    root = Path(submission_dir)
    tasks: dict[str, Any] = {}
    for task_id, test_path, target_path in (
        (1, task1_test_path, task1_target_path),
        (2, task2_test_path, task2_target_path),
    ):
        if test_path is None and target_path is None and not _task_files_exist(root, task_id):
            continue
        if test_path is None:
            raise HermesLocalJudgeError(f"task{task_id}_test_path is required")
        tasks[str(task_id)] = _judge_task(
            task_id=task_id,
            submission_dir=root,
            test_path=Path(test_path),
            target_path=Path(target_path) if target_path is not None else None,
            dataset_key=dataset_key,
            prediction_dataset_key=prediction_dataset_key,
        )
    if not tasks:
        raise HermesLocalJudgeError("no task artifacts or task paths provided")

    rules_passed = all(task["rules_passed"] for task in tasks.values())
    total_score = float(sum(task["score"]["task_total_score"] for task in tasks.values()))
    decision = _decision(tasks=tasks, rules_passed=rules_passed)
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "submission_dir": str(root),
        "rule_interpretation": {
            "task1_runtime_scoring": (
                "Uses the detailed threshold table: train_time max 35 and "
                "inference_time max 40. This resolves the public text conflict "
                "where the component summary says train max 45 and inference max 30."
            ),
            "late_frechet": (
                "Local proxy uses the existing Lorentzian-only scorer unless a "
                "future Frechet FD^2 implementation is supplied."
            ),
        },
        "rules_passed": rules_passed,
        "total_score_proxy": total_score if rules_passed else 0.0,
        "tasks": tasks,
        "decision": decision,
        "next_command": decision,
    }
    if output_path is not None:
        output_file = Path(output_path)
        report["judge_report_path"] = str(output_file)
        _write_json(output_file, report)
    return report


def _judge_task(
    *,
    task_id: int,
    submission_dir: Path,
    test_path: Path,
    target_path: Path | None,
    dataset_key: str,
    prediction_dataset_key: str,
) -> dict[str, Any]:
    audit = audit_competition_submission(
        task_id=task_id,
        submission_dir=submission_dir,
        test_path=test_path,
        output_path=None,
        dataset_key=dataset_key,
    )
    log_review = _review_task_log(submission_dir / f"task{task_id}_logs.log")
    timing = _read_task_timing(submission_dir / f"task{task_id}_time.csv")
    rules_passed = audit["decision"] == "passed" and log_review["decision"] == "passed"
    score_report = None
    if target_path is not None:
        score_report = score_competition_prediction(
            task_id=task_id,
            prediction_path=submission_dir / f"task{task_id}_pred.hdf5",
            target_path=target_path,
            output_path=None,
            prediction_dataset_key=prediction_dataset_key,
            target_dataset_key=dataset_key,
        )
    task_score = _score_task(
        task_id=task_id,
        timing=timing,
        score_report=score_report,
        rules_passed=rules_passed,
    )
    return {
        "task_id": task_id,
        "rules_passed": rules_passed,
        "audit": audit,
        "log_review": log_review,
        "timing": timing,
        "validation_score": score_report,
        "score": task_score,
    }


def _score_task(
    *,
    task_id: int,
    timing: dict[str, float],
    score_report: dict[str, Any] | None,
    rules_passed: bool,
) -> dict[str, Any]:
    if not rules_passed:
        return _zero_score(reason="rule_gate_failed")
    if score_report is None:
        return _zero_score(reason="missing_local_ground_truth")
    inference_time = timing["inference_time"]
    if inference_time > INFERENCE_TIMEOUT_SECONDS:
        return _zero_score(reason="inference_over_two_minutes", timeout_zeroed=True)

    segment_score = float(score_report["segment_score"])
    if task_id == 1:
        prediction_score = segment_score * 0.75
        train_score = _task1_train_time_score(timing["train_time"])
        inference_score = _linear_runtime_score(
            elapsed_seconds=inference_time,
            max_score=40.0,
            timeout_seconds=INFERENCE_TIMEOUT_SECONDS,
        )
        total = prediction_score + train_score + inference_score
        return {
            "score_basis": "task1_threshold_table_interpretation",
            "segment_score": segment_score,
            "prediction_accuracy_score": prediction_score,
            "train_time_score": train_score,
            "inference_time_score": inference_score,
            "task_total_score": float(min(150.0, total)),
            "timeout_zeroed": False,
        }
    if task_id == 2:
        prediction_score = segment_score * 1.5
        train_budget_passed = timing["train_time"] <= TASK2_TRAIN_BUDGET_SECONDS
        return {
            "score_basis": "task2_prediction_only_with_inference_timeout",
            "segment_score": segment_score,
            "prediction_accuracy_score": prediction_score,
            "train_time_score": None,
            "inference_time_score": None,
            "train_budget_passed": train_budget_passed,
            "task_total_score": float(min(150.0, prediction_score)),
            "timeout_zeroed": False,
        }
    raise HermesLocalJudgeError(f"unsupported task_id: {task_id}")


def _zero_score(*, reason: str, timeout_zeroed: bool = False) -> dict[str, Any]:
    return {
        "score_basis": reason,
        "segment_score": None,
        "prediction_accuracy_score": 0.0,
        "train_time_score": 0.0,
        "inference_time_score": 0.0,
        "task_total_score": 0.0,
        "timeout_zeroed": timeout_zeroed,
    }


def _task1_train_time_score(train_time: float) -> float:
    if train_time <= 60.0 * 60.0:
        return 35.0
    if train_time <= 120.0 * 60.0:
        return 25.0
    if train_time <= 300.0 * 60.0:
        return 20.0
    if train_time <= 500.0 * 60.0:
        return 10.0
    return 0.0


def _linear_runtime_score(
    *,
    elapsed_seconds: float,
    max_score: float,
    timeout_seconds: float,
) -> float:
    if elapsed_seconds <= 0.0:
        return max_score
    if elapsed_seconds >= timeout_seconds:
        return 0.0
    return float(max_score * (1.0 - elapsed_seconds / timeout_seconds))


def _review_task_log(path: Path) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    if not path.exists():
        issues.append({"code": "missing_log", "detail": str(path)})
        text = ""
    else:
        text = path.read_text(encoding="utf-8")
    _require_phrase(text, "Trace", "missing_trace", path, issues)
    _require_phrase(
        text,
        "Experiment Tracking",
        "missing_experiment_tracking",
        path,
        issues,
    )
    rubric = {
        "trace": 35 if "Trace" in text else 0,
        "experiment_tracking": 35 if "Experiment Tracking" in text else 0,
        "conclusion_or_decision": 15
        if ("Conclusion" in text or "decision" in text.lower())
        else 0,
        "artifact_or_command_detail": 15
        if ("artifact" in text.lower() or "command" in text.lower() or "version:" in text)
        else 0,
    }
    return {
        "decision": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "issue_codes": [issue["code"] for issue in issues],
        "issues": issues,
        "agent_log_score_proxy": int(sum(rubric.values())),
        "rubric": rubric,
    }


def _require_phrase(
    text: str,
    phrase: str,
    code: str,
    path: Path,
    issues: list[dict[str, str]],
) -> None:
    if phrase not in text:
        issues.append({"code": code, "detail": str(path)})


def _read_task_timing(path: Path) -> dict[str, float]:
    if not path.exists():
        return {"train_time": 0.0, "inference_time": float("inf")}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if len(rows) != 1:
        raise HermesLocalJudgeError(f"invalid time csv rows: {path}")
    try:
        return {
            "train_time": float(rows[0]["train_time"]),
            "inference_time": float(rows[0]["inference_time"]),
        }
    except (KeyError, ValueError) as exc:
        raise HermesLocalJudgeError(f"invalid time csv values: {path}") from exc


def _task_files_exist(root: Path, task_id: int) -> bool:
    return any(
        (root / f"task{task_id}_{suffix}").exists()
        for suffix in ("pred.hdf5", "time.csv", "logs.log")
    )


def _decision(*, tasks: dict[str, Any], rules_passed: bool) -> str:
    if not rules_passed:
        return "hold_rule_gate_failed"
    if any(task["score"].get("timeout_zeroed") for task in tasks.values()):
        return "iterate_before_submit"
    if any(task["score"].get("score_basis") == "missing_local_ground_truth" for task in tasks.values()):
        return "review_without_score"
    if any(
        task["score"].get("train_budget_passed") is False for task in tasks.values()
    ):
        return "hold_training_budget_risk"
    return "submit_candidate"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
