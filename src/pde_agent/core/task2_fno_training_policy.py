"""Task 2 FNO candidate planning policy.

Generates hyperparameter grids and selects next candidates to explore.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Task2FnoTrainingPolicyError(RuntimeError):
    pass


# Default search spaces
DEFAULT_HIDDEN_CHANNELS = [64, 96, 128]
DEFAULT_MODES = [32, 48, 64, 80]
DEFAULT_DEPTHS = [3, 4, 5, 6]
DEFAULT_PHYSICS_WEIGHTS = [0.0, 0.005, 0.01, 0.02]
DEFAULT_LEARNING_RATES = [5e-4, 1e-3, 2e-3]
DEFAULT_TRAIN_STEPS = [5000, 10000, 20000]
DEFAULT_BATCH_SIZES = [16, 32]


def plan_task2_fno_candidates(
    *,
    evidence_paths: list[str | Path] | None = None,
    output_path: str | Path | None = None,
    max_candidates: int = 8,
    hidden_channels_values: list[int] | None = None,
    modes_values: list[int] | None = None,
    depths_values: list[int] | None = None,
    physics_weights_values: list[float] | None = None,
    learning_rates_values: list[float] | None = None,
    train_steps_values: list[int] | None = None,
    batch_sizes_values: list[int] | None = None,
    blocked_configs: list[dict[str, Any]] | None = None,
    search_guidance: list[dict[str, Any]] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Generate Task 2 FNO training candidate configurations.

    Args:
        evidence_paths: Previous evidence JSON paths for learning.
        output_path: Where to write the candidate plan.
        max_candidates: Upper bound on configs to generate.
        hidden_channels_values: Search space for hidden channels.
        modes_values: Search space for Fourier modes.
        depths_values: Search space for model depth.
        physics_weights_values: Search space for physics loss weight.
        learning_rates_values: Search space for learning rate.
        train_steps_values: Search space for training steps.
        batch_sizes_values: Search space for batch size.
        blocked_configs: Configs to exclude.
        search_guidance: Optional guidance dicts from previous iterations.
        force: Overwrite existing output.

    Returns:
        Plan dict with candidate_configs list.
    """
    output_file = Path(output_path) if output_path is not None else None
    if output_file is not None and output_file.exists() and not force:
        raise Task2FnoTrainingPolicyError(f"output already exists: {output_file}")

    evidences = _load_evidences(evidence_paths or [])
    attempted = _attempted_config_signatures(evidences)

    h_vals = hidden_channels_values or DEFAULT_HIDDEN_CHANNELS
    m_vals = modes_values or DEFAULT_MODES
    d_vals = depths_values or DEFAULT_DEPTHS
    p_vals = physics_weights_values or DEFAULT_PHYSICS_WEIGHTS
    lr_vals = learning_rates_values or DEFAULT_LEARNING_RATES
    s_vals = train_steps_values or DEFAULT_TRAIN_STEPS
    b_vals = batch_sizes_values or DEFAULT_BATCH_SIZES

    # Apply search guidance
    if search_guidance:
        for guidance in search_guidance:
            if guidance.get("guidance_type") == "focus_axis":
                axis = guidance.get("axis")
                preferred = guidance.get("preferred_values")
                if axis == "modes" and preferred:
                    m_vals = _merge_preferred(m_vals, preferred)
                elif axis == "depth" and preferred:
                    d_vals = _merge_preferred(d_vals, preferred)
                elif axis == "hidden_channels" and preferred:
                    h_vals = _merge_preferred(h_vals, preferred)
                elif axis == "physics_weight" and preferred:
                    p_vals = _merge_preferred_floats(p_vals, preferred)

    candidates: list[dict[str, Any]] = []
    rng_seed = 42

    # Strategy: grid sweep with adaptive pruning
    # Phase 1: coarse sweep on depth × physics_weight (most impactful)
    for depth in d_vals:
        for pw in p_vals:
            for modes in m_vals:
                for hidden in h_vals:
                    for lr in lr_vals:
                        for steps in s_vals:
                            for bs in b_vals:
                                if len(candidates) >= max_candidates:
                                    break
                                config = {
                                    "hidden_channels": hidden,
                                    "modes": modes,
                                    "depth": depth,
                                    "physics_weight": pw,
                                    "learning_rate": lr,
                                    "train_steps": steps,
                                    "batch_size": bs,
                                    "device": "mps",
                                }
                                sig = _config_signature(config)
                                if sig in attempted:
                                    continue
                                if _is_blocked(config, blocked_configs or []):
                                    continue
                                candidates.append(
                                    {
                                        "candidate_id": f"task2_{sig}",
                                        "candidate_family": "fno_lite_task2",
                                        "config": config,
                                        "rationale": [
                                            f"depth={depth} modes={modes} hidden={hidden}",
                                            f"physics_weight={pw} lr={lr} steps={steps}",
                                        ],
                                    }
                                )
                            if len(candidates) >= max_candidates:
                                break
                        if len(candidates) >= max_candidates:
                            break
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    # If still short, fill with random combinations from full space
    if len(candidates) < max_candidates:
        import random

        random.seed(rng_seed)
        all_combos = [
            {
                "hidden_channels": h,
                "modes": m,
                "depth": d,
                "physics_weight": p,
                "learning_rate": lr,
                "train_steps": s,
                "batch_size": b,
                "device": "mps",
            }
            for h in h_vals
            for m in m_vals
            for d in d_vals
            for p in p_vals
            for lr in lr_vals
            for s in s_vals
            for b in b_vals
        ]
        random.shuffle(all_combos)
        for config in all_combos:
            if len(candidates) >= max_candidates:
                break
            sig = _config_signature(config)
            if sig in attempted or any(
                _config_signature(c["config"]) == sig for c in candidates
            ):
                continue
            if _is_blocked(config, blocked_configs or []):
                continue
            candidates.append(
                {
                    "candidate_id": f"task2_{sig}",
                    "candidate_family": "fno_lite_task2",
                    "config": config,
                    "rationale": ["random_exploration_fill"],
                }
            )

    plan = {
        "schema_version": 1,
        "state_hint": "TASK2_FNO_CANDIDATES_PLANNED",
        "candidate_count": len(candidates),
        "candidate_configs": candidates,
        "search_space": {
            "hidden_channels": h_vals,
            "modes": m_vals,
            "depths": d_vals,
            "physics_weights": p_vals,
            "learning_rates": lr_vals,
            "train_steps": s_vals,
            "batch_sizes": b_vals,
        },
        "evidence_based_pruning": {
            "attempted_signatures": len(attempted),
            "blocked_configs": len(blocked_configs or []),
        },
        "next_hint": "run candidates through task2-fno-training-loop",
    }

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(plan, indent=2))

    return plan


def _config_signature(config: dict[str, Any]) -> str:
    """Stable signature for deduplication."""
    return (
        f"h{config.get('hidden_channels')}"
        f"_m{config.get('modes')}"
        f"_d{config.get('depth')}"
        f"_pw{config.get('physics_weight')}"
        f"_lr{config.get('learning_rate')}"
        f"_s{config.get('train_steps')}"
        f"_b{config.get('batch_size')}"
    )


def _attempted_config_signatures(evidences: list[dict[str, Any]]) -> set[str]:
    signatures: set[str] = set()
    for ev in evidences:
        if not isinstance(ev, dict):
            continue
        config = ev.get("config") or ev.get("best_config")
        if isinstance(config, dict):
            signatures.add(_config_signature(config))
        # Also check candidate records inside optimization reports
        records = ev.get("candidate_records", [])
        for record in records:
            if isinstance(record, dict):
                cfg = record.get("config")
                if isinstance(cfg, dict):
                    signatures.add(_config_signature(cfg))
    return signatures


def _is_blocked(config: dict[str, Any], blocked_configs: list[dict[str, Any]]) -> bool:
    for blocked in blocked_configs:
        if not isinstance(blocked, dict):
            continue
        match = True
        for key, value in blocked.items():
            if config.get(key) != value:
                match = False
                break
        if match:
            return True
    return False


def _load_evidences(paths: list[str | Path]) -> list[dict[str, Any]]:
    evidences: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                evidences.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return evidences


def _merge_preferred(base: list[int], preferred: list[int]) -> list[int]:
    """Move preferred values to front while keeping full diversity."""
    seen = set()
    result: list[int] = []
    for v in preferred:
        if v not in seen:
            seen.add(v)
            result.append(v)
    for v in base:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


def _merge_preferred_floats(base: list[float], preferred: list[float]) -> list[float]:
    seen: set[float] = set()
    result: list[float] = []
    for v in preferred:
        if v not in seen:
            seen.add(v)
            result.append(v)
    for v in base:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result
