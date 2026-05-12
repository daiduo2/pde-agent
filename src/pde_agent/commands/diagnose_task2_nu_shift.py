
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.task2_nu_shift_diagnostic import (
    STATE_HINT,
    diagnose_task2_nu_shift,
)


def diagnose_task2_nu_shift_command(
    *,
    target_path: str,
    output_path: str,
    reference_path: str | None,
    dataset_key: str,
    nu_key: str,
    context_steps: int,
    sample_limit: int | None,
) -> dict[str, object]:
    report = diagnose_task2_nu_shift(
        target_path=Path(target_path),
        output_path=Path(output_path),
        reference_path=Path(reference_path) if reference_path is not None else None,
        dataset_key=dataset_key,
        nu_key=nu_key,
        context_steps=context_steps,
        sample_limit=sample_limit,
    )
    return success_payload(
        run_id="local",
        command="diagnose-task2-nu-shift",
        state_hint=STATE_HINT,
        artifacts={"task2_nu_shift_diagnostic_path": report["diagnostic_path"]},
        metrics={
            "sample_count": report["target"]["sample_count"],
            "nu_available": report["target"]["nu_available"],
            "shift_score": report["domain_shift"]["shift_score"],
            "strategy_id": report["recommendation"]["strategy_id"],
        },
        next_hint=str(report["next_command"]),
    )
