
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.task1_retrieval_prior import (
    STATE_HINT,
    run_task1_retrieval_prior,
)


def predict_task1_retrieval_prior_command(
    *,
    reference_path: str,
    test_path: str,
    output_dir: str,
    dataset_key: str,
    k: int,
    mode: str,
    leave_one_out: bool,
    force: bool,
) -> dict[str, object]:
    report = run_task1_retrieval_prior(
        reference_path=Path(reference_path),
        test_path=Path(test_path),
        output_dir=Path(output_dir),
        dataset_key=dataset_key,
        k=k,
        mode=mode,
        leave_one_out=leave_one_out,
        force=force,
    )
    return success_payload(
        run_id="local",
        command="predict-task1-retrieval-prior",
        state_hint=STATE_HINT,
        artifacts=report["artifacts"],
        metrics={
            **report["inputs"],
            "k": report["candidate"]["parameters"]["k"],
            "mode": report["candidate"]["parameters"]["mode"],
            "leave_one_out": report["candidate"]["parameters"]["leave_one_out"],
            "inference_time": report["inference"]["elapsed_seconds"],
        },
        next_hint=str(report["next_command"]),
    )
