from __future__ import annotations

import json
from pathlib import Path


STATE_HINT = "HERMES_AUTONOMY_DECISION_RECORDED"

DECISION_NEXT_COMMANDS = {
    "request_evidence_sweep": "run-fno-evidence-sweep",
    "accept_current_winner": "run-fno-iteration",
}

VALID_REASON_CODES = {
    "single_window_improvement_needs_validation",
    "clear_improvement_accept_for_now",
}


class AutonomyDecisionError(ValueError):
    """Raised when Hermes records a decision outside the allowed protocol."""


def record_autonomy_decision(
    *,
    decision: str,
    reason_code: str,
    evidence_source_path: Path,
    output_path: Path,
) -> dict[str, object]:
    evidence_file = Path(evidence_source_path)
    output_file = Path(output_path)

    if decision not in DECISION_NEXT_COMMANDS:
        raise AutonomyDecisionError(f"unsupported decision: {decision}")
    if reason_code not in VALID_REASON_CODES:
        raise AutonomyDecisionError(f"unsupported reason_code: {reason_code}")
    if not evidence_file.is_file():
        raise AutonomyDecisionError(f"evidence source file not found: {evidence_file}")
    if output_file.exists():
        raise AutonomyDecisionError(f"output file already exists: {output_file}")

    report: dict[str, object] = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "decision": decision,
        "reason_code": reason_code,
        "evidence_source_path": str(evidence_file),
        "autonomy_decision_path": str(output_file),
        "next_command": DECISION_NEXT_COMMANDS[decision],
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
