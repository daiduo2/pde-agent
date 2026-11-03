"""Task 2 FNO training optimization loop.

Runs multiple candidate trainings and selects the best.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.json_output import success_payload
from pde_agent.core.task2_fno_training_core import (
    Task2FnoTrainingCoreError,
    train_task2_fno_model,
)

STATE_HINT = "TASK2_FNO_OPTIMIZATION_LOOP_RAN"


class Task2FnoTrainingLoopError(RuntimeError):
    pass


def run_task2_fno_optimization_loop(
    *,
    candidate_plan_path: str | Path,
    output_dir: str | Path = "/tmp/task2_fno_optimization",
    device: str = "mps",
    force: bool = False,
    memory: Any | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run training for all candidates in a plan and select best.

    Args:
        candidate_plan_path: Path to candidate plan JSON.
        output_dir: Directory for all candidate outputs.
        device: Torch device.
        force: Overwrite existing outputs.
        memory: Optional HypothesisLabMemory.
        session_id: Session ID for memory.

    Returns:
        Optimization report with candidate_records and selected_candidate.
    """
    plan_path = Path(candidate_plan_path)
    if not plan_path.is_file():
        raise Task2FnoTrainingLoopError(f"missing plan: {plan_path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    configs = plan.get("candidate_configs", [])
    if not configs:
        raise Task2FnoTrainingLoopError("plan has no candidate_configs")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    candidate_records: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    best_score = float("-inf")

    for i, candidate in enumerate(configs):
        candidate_id = candidate.get("candidate_id", f"candidate_{i}")
        config = candidate.get("config", {})
        rationale = candidate.get("rationale", [])

        candidate_dir = root / candidate_id
        print(f"\n{'='*60}")
        print(f"Candidate {i+1}/{len(configs)}: {candidate_id}")
        print(f"Config: {config}")
        print(f"{'='*60}")

        try:
            train_kwargs = dict(config)
            train_kwargs.pop("device", None)
            report = train_task2_fno_model(
                **train_kwargs,
                device=device,
                output_dir=candidate_dir,
                force=force,
                memory=memory,
                session_id=session_id,
            )
            score = report.get("task2_proxy", 0.0)
            segment_score = report.get("segment_score", 0.0)
            record = {
                "candidate_id": candidate_id,
                "candidate_family": "fno_lite_task2",
                "config": config,
                "score": score,
                "segment_score": segment_score,
                "best_ext_val_loss": report.get("best_ext_val_loss"),
                "runtime_seconds": report.get("runtime_seconds"),
                "parameters": report.get("parameters"),
                "status": "completed",
                "rationale": rationale,
                "artifacts": report.get("artifacts", {}),
            }
            candidate_records.append(record)

            if score > best_score:
                best_score = score
                best_record = record

        except (Task2FnoTrainingCoreError, RuntimeError, OSError) as exc:
            record = {
                "candidate_id": candidate_id,
                "candidate_family": "fno_lite_task2",
                "config": config,
                "score": None,
                "segment_score": None,
                "status": "failed",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "rationale": rationale,
            }
            candidate_records.append(record)

    # Write optimization report
    opt_report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "candidate_count": len(configs),
        "completed_count": sum(1 for r in candidate_records if r["status"] == "completed"),
        "failed_count": sum(1 for r in candidate_records if r["status"] == "failed"),
        "candidate_records": candidate_records,
        "selected_candidate": best_record,
        "best_score": best_score if best_record else None,
    }

    report_path = root / "task2_fno_optimization_report.json"
    report_path.write_text(json.dumps(opt_report, indent=2))

    return opt_report
