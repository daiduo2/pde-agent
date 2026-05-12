"""Record timeout-aware UNet evidence without closing the UNet line."""


import json
from pathlib import Path
from typing import Any


from pde_agent.core.state_hint_registry import LOCAL_EXPERIMENT_RESULT as STATE_HINT


class UNetTimeoutEvidenceError(RuntimeError):
    pass


def record_unet_timeout_evidence(
    *,
    output_path: str | Path,
    elapsed_seconds: float,
    timeout_seconds: float,
    attempted_command: str,
    checkpoint_path: str,
) -> dict[str, Any]:
    if elapsed_seconds < 0.0 or timeout_seconds <= 0.0:
        raise UNetTimeoutEvidenceError("elapsed_seconds and timeout_seconds are invalid")
    result = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "result_path": str(output_path),
        "result_status": "failed",
        "score": None,
        "baseline_score": None,
        "score_delta": None,
        "inference_time_seconds": float(elapsed_seconds),
        "failure_reason": "unet_inference_timeout",
        "result_summary": (
            "UNet competition inference exceeded the local timeout budget; "
            "recorded as negative evidence with an optimization path."
        ),
        "source": {
            "state_hint": "UNET_TIMEOUT_EVIDENCE_RECORDED",
            "attempted_command": attempted_command,
            "checkpoint_path": checkpoint_path,
            "elapsed_seconds": float(elapsed_seconds),
            "timeout_seconds": float(timeout_seconds),
            "optimization_plan": {
                "strategy_id": "unet_timeout_optimization_v0",
                "hypothesis": (
                    "UNet may remain useful if inference memory and rollout throughput "
                    "are controlled before full 1000-sample submission runs."
                ),
                "candidate_actions": [
                    "reduce rollout sample batch size",
                    "profile autoregressive loop hotspots",
                    "cache static tensors and model buffers",
                    "evaluate smaller validation shards before full submission",
                ],
                "acceptance_criteria": [
                    "full Task inference under 120 seconds",
                    "submission audit remains passed",
                    "score does not regress against FNO validation baseline",
                ],
            },
        },
    }
    _write_json(Path(output_path), result)
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
