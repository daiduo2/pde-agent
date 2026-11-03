"""Task 2 FNO training escalation pipeline.

Mirrors Task 1's fno_training_escalation but simplified for the
single-family fno_lite_task2 regime. When training stagnates or hits a
doom_loop, this module generates strategy suggestions by analyzing
blocked_configs, search_guidance, and convergence_history from memory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATE_HINT = "TASK2_FNO_TRAINING_ESCALATION_RAN"
NO_RECOMMENDATIONS_STATE_HINT = "TASK2_FNO_TRAINING_ESCALATION_NO_RECOMMENDATIONS"
RECOMMENDATIONS_GENERATED_STATE_HINT = "TASK2_FNO_TRAINING_ESCALATION_RECOMMENDATIONS_GENERATED"


class Task2FnoTrainingEscalationError(RuntimeError):
    pass


def run_task2_fno_training_escalation_pipeline(
    *,
    workspace_dir: str | Path,
    memory_path: str | Path,
    output_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run the Task 2 FNO training escalation pipeline.

    Args:
        workspace_dir: Workspace directory for artifacts.
        memory_path: Path to the latest task2_fno_memory_*.json file.
        output_path: Optional explicit output path for the escalation report.
        force: Overwrite existing output file.

    Returns:
        Escalation report dict with failure analysis and strategy recommendations.
    """
    workspace = Path(workspace_dir).expanduser().resolve()
    memory_file = Path(memory_path).expanduser().resolve()
    out_file = (
        Path(output_path)
        if output_path is not None
        else workspace / "task2_fno_training_escalation_report.json"
    )

    memory = _load_json(memory_file)
    if memory is None:
        raise Task2FnoTrainingEscalationError(
            f"could not load memory: {memory_file}"
        )

    failure_analysis = _analyze_training_failure(memory=memory)
    strategy_recommendations = _generate_strategy_recommendations(
        memory=memory,
        failure_analysis=failure_analysis,
    )

    blocked_configs = list(memory.get("blocked_configs", []))
    search_guidance = list(memory.get("search_guidance", []))
    convergence_history = memory.get("convergence_history", {})

    report = _build_escalation_report(
        workspace=workspace,
        memory_file=memory_file,
        failure_analysis=failure_analysis,
        strategy_recommendations=strategy_recommendations,
        blocked_configs=blocked_configs,
        search_guidance=search_guidance,
        convergence_history=convergence_history,
    )

    _write_json(out_file, report, force=force)
    return report


def _analyze_training_failure(
    *,
    memory: dict[str, Any],
) -> dict[str, Any]:
    """Analyze why Task 2 FNO training failed or stagnated."""
    trajectory = memory.get("trajectory_summary", [])
    convergence_history = memory.get("convergence_history", {})
    best_so_far = memory.get("best_so_far", {})
    current_best = memory.get("current_best", {})

    no_improvement_count = _int_or_default(
        convergence_history.get("no_improvement_count"), 0
    )
    best_score_history = convergence_history.get("best_score_history", [])
    best_score = _float_or_none(best_so_far.get("score"))
    current_score = _float_or_none(current_best.get("score"))

    failure_modes: list[dict[str, Any]] = []

    # Check for doom loop: same config repeated without improvement
    if len(trajectory) >= 3:
        repeated = _repeated_non_improving_config(trajectory, min_score_delta=0.5)
        if repeated is not None:
            failure_modes.append({
                "mode": "doom_loop",
                "severity": "high",
                "description": (
                    "Same configuration repeated for 3+ iterations "
                    "without meaningful score improvement."
                ),
                "config_signature": repeated.get("config_signature"),
                "scores": repeated.get("scores"),
            })

    # Check for stagnation: no improvement over multiple iterations
    if no_improvement_count >= 2 and not any(
        fm["mode"] == "doom_loop" for fm in failure_modes
    ):
        failure_modes.append({
            "mode": "stagnation",
            "severity": "medium",
            "description": (
                f"No score improvement for {no_improvement_count} consecutive iterations."
            ),
            "no_improvement_count": no_improvement_count,
            "best_score_history": best_score_history,
        })

    # Check for poor absolute performance
    if best_score is not None and best_score <= 0.0:
        failure_modes.append({
            "mode": "poor_absolute_performance",
            "severity": "high",
            "description": (
                "Best score is zero or negative, indicating fundamental "
                "issues with model capacity or training setup."
            ),
            "best_score": best_score,
        })

    # Check for regression: current best worse than historical best
    if (
        current_score is not None
        and best_score is not None
        and current_score < best_score
    ):
        failure_modes.append({
            "mode": "regression",
            "severity": "medium",
            "description": (
                "Current iteration score is below the historical best, "
                "indicating exploration in the wrong direction."
            ),
            "current_score": current_score,
            "historical_best": best_score,
            "delta": round(current_score - best_score, 6),
        })

    # Check for exhausted search space
    blocked_count = len(memory.get("blocked_configs", []))
    trajectory_count = len(trajectory)
    if blocked_count >= trajectory_count * 0.5 and trajectory_count >= 4:
        failure_modes.append({
            "mode": "search_space_exhaustion",
            "severity": "medium",
            "description": (
                f"{blocked_count} of {trajectory_count} configurations "
                "have been blocked. The effective search space may be exhausted."
            ),
            "blocked_count": blocked_count,
            "trajectory_count": trajectory_count,
        })

    # If no specific failure mode detected but convergence history is short
    if not failure_modes and len(best_score_history) < 2:
        failure_modes.append({
            "mode": "insufficient_data",
            "severity": "low",
            "description": (
                "Too few iterations to determine a clear failure mode. "
                "Consider running more iterations before escalation."
            ),
            "iteration_count": len(best_score_history),
        })

    primary_failure = failure_modes[0] if failure_modes else {
        "mode": "unknown",
        "severity": "low",
        "description": "No clear failure mode detected.",
    }

    return {
        "primary_failure_mode": primary_failure,
        "failure_modes": failure_modes,
        "best_score": best_score,
        "current_score": current_score,
        "no_improvement_count": no_improvement_count,
        "iteration_count": len(trajectory),
        "blocked_config_count": blocked_count,
    }


def _generate_strategy_recommendations(
    *,
    memory: dict[str, Any],
    failure_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate strategy recommendations based on failure analysis."""
    recommendations: list[dict[str, Any]] = []
    primary_mode = failure_analysis.get("primary_failure_mode", {}).get("mode", "unknown")
    best_config = memory.get("best_so_far", {}).get("config", {})

    if primary_mode == "doom_loop":
        recommendations.append({
            "strategy": "escape_doom_loop",
            "priority": 1,
            "description": (
                "Force exploration away from the repeated config by "
                "perturbing key hyperparameters."
            ),
            "actions": [
                "Increase modes by 16-32 or decrease by 8-16",
                "Change depth by +/- 1-2 layers",
                "Adjust physics_weight by +/- 0.005-0.01",
                "Switch learning_rate between 5e-4, 1e-3, 2e-3",
            ],
            "rationale": "Breaking symmetry in the search space to escape local optimum.",
        })
        recommendations.append({
            "strategy": "increase_train_budget",
            "priority": 2,
            "description": (
                "The repeated config may need more training steps to show improvement."
            ),
            "actions": [
                "Double train_steps (e.g., 10000 -> 20000)",
                "Add cosine annealing with longer T_max",
                "Reduce learning_rate and extend training",
            ],
            "rationale": "Under-training can manifest as apparent stagnation.",
        })

    elif primary_mode == "stagnation":
        recommendations.append({
            "strategy": "expand_search_radius",
            "priority": 1,
            "description": (
                "Broaden the hyperparameter search to explore untapped regions."
            ),
            "actions": [
                "Try hidden_channels outside current range (48, 160, 192)",
                "Explore extreme modes values (16, 96, 128)",
                "Test physics_weight at 0.03, 0.05, or 0.0",
                "Vary batch_size between 8 and 64",
            ],
            "rationale": "Stagnation often signals that the current search region is exhausted.",
        })
        recommendations.append({
            "strategy": "fine_tune_best",
            "priority": 2,
            "description": (
                "Perform fine-grained search around the best known config."
            ),
            "actions": [
                "Grid search around best hidden_channels +/- 16",
                "Grid search around best modes +/- 8",
                "Grid search around best depth +/- 1",
                "Grid search around best physics_weight +/- 0.005",
            ],
            "rationale": "The global optimum may be near the current best.",
        })

    elif primary_mode == "poor_absolute_performance":
        recommendations.append({
            "strategy": "increase_model_capacity",
            "priority": 1,
            "description": (
                "Scores near zero suggest the model is too small or "
                "not training long enough."
            ),
            "actions": [
                "Increase hidden_channels to 128-192",
                "Increase depth to 5-6 layers",
                "Increase modes to 64-80",
                "Extend train_steps to 20000+",
            ],
            "rationale": "Insufficient capacity is the most common cause of near-zero scores.",
        })
        recommendations.append({
            "strategy": "debug_training_pipeline",
            "priority": 2,
            "description": (
                "Verify that data loading, normalization, and loss computation are correct."
            ),
            "actions": [
                "Check that train/val data shapes are [N, T, X]",
                "Verify normalization produces reasonable ranges",
                "Inspect a few prediction vs target plots",
                "Confirm physics_loss is computed correctly (dx, dt, nu)",
            ],
            "rationale": "Data or loss bugs can silently produce near-zero scores.",
        })

    elif primary_mode == "regression":
        recommendations.append({
            "strategy": "return_to_best",
            "priority": 1,
            "description": (
                "Resume from the historically best config and explore cautiously."
            ),
            "actions": [
                "Reuse best config as anchor",
                "Perturb only one hyperparameter at a time",
                "Use smaller perturbation magnitudes",
            ],
            "rationale": "Regression means recent exploration moved away from the optimum.",
        })

    elif primary_mode == "search_space_exhaustion":
        recommendations.append({
            "strategy": "reset_search_space",
            "priority": 1,
            "description": (
                "Many configs have been blocked. Reset with new search space boundaries."
            ),
            "actions": [
                "Widen hidden_channels range",
                "Add new physics_weight values",
                "Introduce learning_rate values not yet tried",
                "Consider different optimizer settings",
            ],
            "rationale": "Blocked configs prune the search space; expansion is needed.",
        })

    # Always add family-specific guidance for fno_lite_task2
    recommendations.append({
        "strategy": "fno_lite_task2_specific",
        "priority": 3,
        "description": (
            "Task 2 uses a single FNO family with 190-step rollout. "
            "Focus on rollout stability and long-horizon accuracy."
        ),
        "actions": [
            "Verify forecast_steps matches target (190)",
            "Consider curriculum: train on shorter rollouts first",
            "Increase physics_weight if rollout drifts",
            "Use larger modes for high-frequency Burgers dynamics",
        ],
        "rationale": "Task 2's long rollout makes stability more important than short-horizon accuracy.",
    })

    # If we have a best config, add a recommendation to exploit it
    if best_config and isinstance(best_config, dict):
        recommendations.append({
            "strategy": "exploit_best_config",
            "priority": 4,
            "description": (
                "The best known config is a proven anchor. "
                "Use it as a baseline for further exploration."
            ),
            "actions": [
                f"Anchor: hidden_channels={best_config.get('hidden_channels')}, "
                f"modes={best_config.get('modes')}, depth={best_config.get('depth')}",
                "Train longer with this config",
                "Ensemble multiple checkpoints from this config",
            ],
            "rationale": "Exploitation ensures we don't lose the best known solution.",
        })

    # Sort by priority
    recommendations.sort(key=lambda r: r.get("priority", 99))

    return recommendations


def _build_escalation_report(
    *,
    workspace: Path,
    memory_file: Path,
    failure_analysis: dict[str, Any],
    strategy_recommendations: list[dict[str, Any]],
    blocked_configs: list[dict[str, Any]],
    search_guidance: list[dict[str, Any]],
    convergence_history: dict[str, Any],
) -> dict[str, Any]:
    """Build the final escalation report dict."""
    has_recommendations = len(strategy_recommendations) > 0
    state_hint = (
        RECOMMENDATIONS_GENERATED_STATE_HINT
        if has_recommendations
        else NO_RECOMMENDATIONS_STATE_HINT
    )

    primary_mode = failure_analysis.get("primary_failure_mode", {}).get("mode", "unknown")

    # Determine next command based on primary failure mode
    if primary_mode in ("doom_loop", "stagnation", "search_space_exhaustion"):
        next_command = "run-task2-fno-training-iteration"
    elif primary_mode == "poor_absolute_performance":
        next_command = "run-task2-fno-training-iteration"
    elif primary_mode == "regression":
        next_command = "run-task2-fno-training-iteration"
    else:
        next_command = "run-task2-fno-training-iteration"

    return {
        "schema_version": 1,
        "state_hint": state_hint,
        "workspace_dir": str(workspace),
        "memory_path": str(memory_file),
        "failure_analysis": failure_analysis,
        "strategy_recommendations": strategy_recommendations,
        "blocked_configs": blocked_configs,
        "search_guidance": search_guidance,
        "convergence_history": convergence_history,
        "next_command": next_command,
        "agent_instructions": [
            "Review failure_analysis to understand what went wrong.",
            "Apply strategy_recommendations in priority order.",
            "Respect blocked_configs when planning next candidates.",
            "Use search_guidance to prune the candidate space.",
            "Run the next_command to continue the training loop.",
        ],
    }


# -- Internal helpers --


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise Task2FnoTrainingEscalationError(
            f"refusing to overwrite: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _int_or_default(value: Any, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _config_signature(config: dict[str, Any]) -> str:
    return (
        f"h{config.get('hidden_channels')}"
        f"_m{config.get('modes')}"
        f"_d{config.get('depth')}"
        f"_pw{config.get('physics_weight')}"
        f"_lr{config.get('learning_rate')}"
        f"_s{config.get('train_steps')}"
        f"_b{config.get('batch_size')}"
    )


def _repeated_non_improving_config(
    trajectory: list[dict[str, Any]],
    min_score_delta: float,
) -> dict[str, Any] | None:
    if len(trajectory) < 3:
        return None
    sigs: list[str] = []
    for item in trajectory:
        config = item.get("config")
        if isinstance(config, dict):
            sigs.append(_config_signature(config))
        else:
            sigs.append("")
    if len(sigs) >= 3 and sigs[-1] and sigs[-1] == sigs[-2] == sigs[-3]:
        scores = [_float_or_none(item.get("score")) for item in trajectory[-3:]]
        if all(s is not None for s in scores):
            if max(scores) - min(scores) < min_score_delta:
                return {
                    "trigger": "repeated_non_improving_config",
                    "config_signature": sigs[-1],
                    "reason": "same_config_repeated_without_improvement",
                    "scores": scores,
                }
    return None
