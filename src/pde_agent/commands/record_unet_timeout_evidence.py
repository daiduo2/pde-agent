from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.unet_timeout_evidence import (
    STATE_HINT,
    record_unet_timeout_evidence,
)


def record_unet_timeout_evidence_command(
    *,
    output_path: str,
    elapsed_seconds: float,
    timeout_seconds: float,
    attempted_command: str,
    checkpoint_path: str,
) -> dict[str, object]:
    result = record_unet_timeout_evidence(
        output_path=Path(output_path),
        elapsed_seconds=elapsed_seconds,
        timeout_seconds=timeout_seconds,
        attempted_command=attempted_command,
        checkpoint_path=checkpoint_path,
    )
    return success_payload(
        run_id="local",
        command="record-unet-timeout-evidence",
        state_hint=STATE_HINT,
        artifacts={"local_strategy_result_path": result["result_path"]},
        metrics={
            "failure_reason": result["failure_reason"],
            "elapsed_seconds": result["inference_time_seconds"],
            "optimization_strategy_id": result["source"]["optimization_plan"][
                "strategy_id"
            ],
        },
        next_hint="record-local-strategy-result",
    )
