from __future__ import annotations

from typing import Any


def family_record(report: dict[str, Any]) -> dict[str, Any]:
    loss_curve = report["training"]["loss_curve"]
    middle = report["metrics"]["middle_guardrail"]
    late = report["metrics"]["late_guardrail"]
    return {
        "model_family": report["model"]["model_family"],
        "operator_type": report["model"]["operator_type"],
        "parameter_count": report["model"]["parameter_count"],
        "runtime_seconds": report["training"]["runtime_seconds"],
        "train_steps": report["training"]["train_steps"],
        "loss_curve": loss_curve,
        "final_train_loss": float(loss_curve["train"][-1]),
        "final_val_loss": float(loss_curve["val"][-1]),
        "early_rel_mse": float(report["metrics"]["early"]["rel_mse"]),
        "early_score": float(report["metrics"]["early"]["score"]),
        "middle_guardrail": middle,
        "late_guardrail": late,
        "decision": report["evidence"]["decision"],
        "reason_codes": report["evidence"]["reason_codes"],
        "artifacts": report["artifacts"],
    }


def select_family(records: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        records,
        key=lambda item: (
            -float(item.get("early_score", 0.0)),
            float(item.get("final_val_loss", float("inf"))),
            float(item.get("runtime_seconds", float("inf"))),
        ),
    )
    selected = dict(ranked[0])
    selected["selection_reason_codes"] = selection_reason_codes(
        selected=selected,
        records=records,
    )
    return selected


def selection_reason_codes(
    *, selected: dict[str, Any], records: list[dict[str, Any]]
) -> list[str]:
    reasons = ["best_early_score"]
    early_scores = [float(item.get("early_score", 0.0)) for item in records]
    if early_scores and max(early_scores) - min(early_scores) < 1e-9:
        reasons.append("tie_broken_by_val_loss_or_runtime")
    if selected.get("middle_guardrail", {}).get("evaluated"):
        reasons.append("middle_guardrail_available")
    if selected.get("late_guardrail", {}).get("evaluated"):
        reasons.append("late_guardrail_available")
    if selected.get("decision") == "continue_training":
        reasons.append("selected_family_wants_more_training")
    return reasons


def family_training_directive(
    *, selected: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    action = family_directive_action(selected=selected, records=records)
    return {
        "action": action,
        "target_family": selected["model_family"],
        "reason_codes": selected["selection_reason_codes"],
        "next_command": "run-task1-short-window-training-loop"
        if action in {"continue_training", "change_loss_or_horizon"}
        else "build-hypothesis-algorithm-design",
    }


def family_directive_action(
    *, selected: dict[str, Any], records: list[dict[str, Any]]
) -> str:
    if all(item.get("decision") == "abandon_model_family" for item in records):
        return "abandon_candidate_family"
    if selected.get("decision") == "change_loss_or_horizon":
        return "change_loss_or_horizon"
    if selected.get("decision") == "continue_training":
        return "continue_training"
    return "expand_model_family"


def family_selection_constraints(
    *, directive: dict[str, Any]
) -> dict[str, list[str]]:
    target = str(directive["target_family"])
    constraints = {
        "training_model_families": [],
        "prefer_training_objectives": [],
        "refine_training_horizons": [],
        "avoid_training_model_families": [],
    }
    if directive["action"] == "abandon_candidate_family":
        constraints["avoid_training_model_families"].append(target)
    else:
        constraints["training_model_families"].append(target)
        constraints["prefer_training_objectives"].append("short_window_mse")
    if directive["action"] == "change_loss_or_horizon":
        constraints["prefer_training_objectives"].extend(
            ["early_weighted_mse", "rollout_consistency"]
        )
        constraints["refine_training_horizons"].append("early_window")
    if "middle_guardrail_available" in directive.get("reason_codes", []):
        constraints["refine_training_horizons"].append("middle_guardrail")
    if "late_guardrail_available" in directive.get("reason_codes", []):
        constraints["refine_training_horizons"].append("late_guardrail")
    return {key: _dedupe(value) for key, value in constraints.items()}


def _dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
