
from pathlib import Path

from pde_agent.core.meta_research_controller import (
    STATE_HINT,
    build_meta_research_directive,
)
from pde_agent.core.json_output import success_payload


def build_agent_meta_research_directive_command(
    *,
    followthrough_report_path: str,
    chain_report_path: str,
    task1_benchmark_path: str,
    candidate_factory_benchmark_path: str,
    output_path: str,
    force: bool,
) -> dict[str, object]:
    report = build_meta_research_directive(
        followthrough_report_path=Path(followthrough_report_path),
        chain_report_path=Path(chain_report_path),
        task1_benchmark_path=Path(task1_benchmark_path),
        candidate_factory_benchmark_path=Path(candidate_factory_benchmark_path),
        output_path=Path(output_path),
        force=force,
    )
    return success_payload(
        run_id="local",
        command="build-agent-meta-research-directive",
        state_hint=STATE_HINT,
        artifacts={
            "meta_research_directive_json": output_path,
            "research_directive_path": output_path,
        },
        metrics={
            "decision": report["decision"],
            "research_focus": report["directive"]["research_focus"],
            "candidate_generator": report["directive"]["candidate_generator"],
        },
        next_hint=report["next_command"],
    )
