from pde_agent.core.json_output import error_payload, success_payload


def test_success_payload_has_fixed_top_level_fields() -> None:
    payload = success_payload(
        run_id="run_001",
        command="init-run",
        state_hint="TASK_PARSED",
        artifacts={"run_root": "runs/run_001"},
        metrics={},
    )
    assert payload == {
        "status": "ok",
        "run_id": "run_001",
        "command": "init-run",
        "state_hint": "TASK_PARSED",
        "artifacts": {"run_root": "runs/run_001"},
        "metrics": {},
        "errors": [],
        "next_hint": None,
    }


def test_error_payload_has_machine_readable_error_shape() -> None:
    payload = error_payload(
        run_id="run_001",
        command="eval-rollout",
        error_code="MISSING_CHECKPOINT",
        error_type="artifact_missing",
        message_short="checkpoint file not found",
        recoverable=True,
        retryable=False,
        missing_artifacts=["checkpoint_path"],
    )
    err = payload["errors"][0]
    assert payload["status"] == "error"
    assert err["error_code"] == "MISSING_CHECKPOINT"
    assert err["error_type"] == "artifact_missing"
    assert err["recoverable"] is True
    assert err["retryable"] is False
    assert err["missing_artifacts"] == ["checkpoint_path"]
