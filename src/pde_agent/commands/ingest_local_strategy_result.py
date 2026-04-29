from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.local_strategy_result_ingestion import (
    RESULT_STATE_HINT,
    ingest_local_strategy_result,
)


def ingest_local_strategy_result_command(
    *,
    source_path: str,
    output_path: str,
    baseline_score: float | None,
    result_status: str,
    failure_reason: str | None,
) -> dict[str, object]:
    result = ingest_local_strategy_result(
        source_path=Path(source_path),
        output_path=Path(output_path),
        baseline_score=baseline_score,
        result_status=result_status,
        failure_reason=failure_reason,
    )
    return success_payload(
        run_id="local",
        command="ingest-local-strategy-result",
        state_hint=RESULT_STATE_HINT,
        artifacts={"local_strategy_result_path": result["result_path"]},
        metrics={
            "source_state_hint": result["source_state_hint"],
            "result_status": result["result_status"],
            "score": result["score"],
            "baseline_score": result["baseline_score"],
            "score_delta": result["score_delta"],
            "inference_time_seconds": result["inference_time_seconds"],
        },
        next_hint="record-local-strategy-result",
    )
