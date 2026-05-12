
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.unet_backend import inspect_unet_checkpoint as inspect_core


def inspect_unet_checkpoint(
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
        command="inspect-unet-checkpoint",
        state_hint="UNET_CHECKPOINT_INSPECTED",
        artifacts={"unet_checkpoint_metadata_path": metadata["metadata_path"]},
        metrics={
            "checkpoint_format": metadata["checkpoint_format"],
            "top_level_key_count": len(metadata["top_level_keys"]),
            "has_model_state": metadata["has_model_state"],
            "has_config": metadata["has_config"],
            "official_unet1d_detected": metadata["official_unet1d_detected"],
        },
        next_hint="predict-unet-local",
    )
