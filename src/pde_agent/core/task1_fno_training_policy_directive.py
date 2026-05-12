"""Task 1 FNO training policy directive functions."""

from __future__ import annotations

from typing import Any
def _fno_training_directive(
    *, selected: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    action = _fno_directive_action(selected=selected, records=records)
    return {
        "action": action,
        "target_family": str(selected["config"].get("model_family", "fno_lite"))
        if action != "fallback_to_conv1d"
        else "conv1d_residual",
        "selected_candidate_id": selected["candidate_id"],
        "reason_codes": selected["selection_reason_codes"]
        + _fno_directive_reason_codes(action=action, selected=selected),
        "next_command": "plan-fno-training-candidates"
        if action
        in {
            "continue_current_fno_family",
            "change_loss",
            "change_horizon",
            "change_architecture",
        }
        else "run-task1-short-window-family-selection-loop",
    }

def _fno_directive_action(
    *, selected: dict[str, Any], records: list[dict[str, Any]]
) -> str:
    if float(selected["early_rel_mse"]) >= 4.5:
        return "fallback_to_conv1d"
    if selected["config"]["loss_type"] == "mse" and float(selected["early_rel_mse"]) >= 1.0:
        return "change_loss"
    if float(selected["early_rel_mse"]) >= 1.0:
        return "change_horizon"
    if _architecture_search_exhausted(records):
        return "expand_new_model_family"
    if selected.get("decision") == "continue_training":
        return "continue_current_fno_family"
    return "change_architecture"

def _fno_directive_reason_codes(
    *, action: str, selected: dict[str, Any]
) -> list[str]:
    reasons = [action]
    if float(selected["early_rel_mse"]) >= 1.0:
        reasons.append("early_rel_mse_high")
    else:
        reasons.append("early_signal_learned")
    return reasons

def _architecture_search_exhausted(records: list[dict[str, Any]]) -> bool:
    return len({(item["config"]["modes"], item["config"]["width"], item["config"]["depth"]) for item in records}) >= 6
