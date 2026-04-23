def _base_payload(run_id: str, command: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "command": command,
        "artifacts": {},
        "metrics": {},
        "errors": [],
        "next_hint": None,
    }


def success_payload(
    run_id: str,
    command: str,
    state_hint: str,
    artifacts: dict[str, object],
    metrics: dict[str, object],
    next_hint: str | None = None,
) -> dict[str, object]:
    payload = _base_payload(run_id=run_id, command=command)
    payload.update(
        {
            "status": "ok",
            "state_hint": state_hint,
            "artifacts": artifacts,
            "metrics": metrics,
            "next_hint": next_hint,
        }
    )
    return payload


def error_payload(
    run_id: str,
    command: str,
    error_code: str,
    error_type: str,
    message_short: str,
    recoverable: bool,
    retryable: bool,
    missing_artifacts: list[str] | None = None,
) -> dict[str, object]:
    payload = _base_payload(run_id=run_id, command=command)
    payload.update(
        {
            "status": "error",
            "state_hint": None,
            "errors": [
                {
                    "error_code": error_code,
                    "error_type": error_type,
                    "recoverable": recoverable,
                    "retryable": retryable,
                    "missing_artifacts": missing_artifacts or [],
                    "message_short": message_short,
                }
            ],
        }
    )
    return payload
