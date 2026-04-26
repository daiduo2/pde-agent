from __future__ import annotations

from pathlib import Path

from pde_agent.core.fno_backend import inspect_fno_checkpoint as inspect_core
from pde_agent.core.json_output import success_payload


def inspect_fno_checkpoint(
    *,
    checkpoint_path: str,
    output_path: str,
) -> dict[str, object]:
    metadata = inspect_core(
        checkpoint_path=Path(checkpoint_path),
        output_path=Path(output_path),
    )
    return success_payload(
        run_id="local",
        command="inspect-fno-checkpoint",
        state_hint="FNO_CHECKPOINT_INSPECTED",
        artifacts={"fno_checkpoint_metadata_path": metadata["metadata_path"]},
        metrics={
            "checkpoint_format": metadata["checkpoint_format"],
            "top_level_key_count": len(metadata["top_level_keys"]),
            "has_model_state": metadata["has_model_state"],
            "has_config": metadata["has_config"],
        },
        next_hint="predict-fno-local",
    )
