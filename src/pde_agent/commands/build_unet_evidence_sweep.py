from __future__ import annotations

from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.unet_evidence_sweep import build_unet_evidence_sweep


def build_unet_evidence_sweep_command(
    *,
    evaluation_report_paths: list[str],
    output_dir: str,
    force: bool,
) -> dict[str, object]:
    summary = build_unet_evidence_sweep(
        evaluation_report_paths=[Path(path) for path in evaluation_report_paths],
        output_dir=Path(output_dir),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-unet-evidence-sweep",
        state_hint="UNET_EVIDENCE_SWEEP_COMPLETED",
        artifacts={
            "summary_json_path": summary["artifacts"]["summary_json_path"],
            "summary_csv_path": summary["artifacts"]["summary_csv_path"],
        },
        metrics={
            "record_count": summary["record_count"],
            "comparison_count": summary["comparison_count"],
            "model_family": summary["model_family"],
        },
        next_hint=summary["next_command"],
    )
