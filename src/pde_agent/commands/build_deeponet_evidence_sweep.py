from __future__ import annotations

from pde_agent.core.deeponet_evidence_sweep import (
    STATE_HINT,
    build_deeponet_evidence_sweep,
)
from pde_agent.core.json_output import success_payload


def build_deeponet_evidence_sweep_command(
    *,
    evaluation_report_paths: list[str],
    output_dir: str,
    force: bool,
) -> dict[str, object]:
    summary = build_deeponet_evidence_sweep(
        evaluation_report_paths=evaluation_report_paths,
        output_dir=output_dir,
        force=force,
    )
    winner_strategies = {
        winner["winner_training_strategy"] for winner in summary["comparison_winners"]
    }
    return success_payload(
        run_id="local",
        command="build-deeponet-evidence-sweep",
        state_hint=STATE_HINT,
        artifacts={
            "summary_json_path": summary["artifacts"]["summary_json_path"],
            "summary_csv_path": summary["artifacts"]["summary_csv_path"],
        },
        metrics={
            "record_count": summary["record_count"],
            "comparison_count": summary["comparison_count"],
            "winner_strategy_count": len(winner_strategies),
        },
        next_hint=str(summary["next_command"]),
    )
