"""Candidate generator registry for hypothesis planning.

Extracted from the legacy Hermes heartbeat module so the registry can be used
by the Claude SDK candidate planner without depending on Hermes runtime code.
"""


from typing import Any


CANDIDATE_GENERATOR_REGISTRY: dict[str, dict[str, Any]] = {
    "postprocess": {
        "cost_level": "light",
        "auto_fallback_allowed": True,
        "requires_target": False,
        "requires_reference": False,
        "default_budget": "smoke",
        "targets_error_modes": ["global_score", "late_stability"],
    },
    "physics_residual_nudge": {
        "cost_level": "light",
        "auto_fallback_allowed": True,
        "requires_target": False,
        "requires_reference": False,
        "default_budget": "smoke",
        "targets_error_modes": ["physical_residual", "late_stability"],
    },
    "sem_window_gate_blend_multi_early_smoke": {
        "cost_level": "medium",
        "auto_fallback_allowed": True,
        "requires_target": True,
        "requires_reference": False,
        "default_budget": "smoke",
        "targets_error_modes": ["early_segment", "transition_window"],
        "saturation_blocked_by": ["local_strategy_skill"],
    },
    "hypothesis_lab_front25_dynamics_smoke": {
        "cost_level": "medium",
        "auto_fallback_allowed": True,
        "factory_handoff_required": True,
        "requires_target": True,
        "requires_reference": True,
        "default_budget": "smoke",
        "targets_error_modes": ["early_segment", "transition_window"],
        "saturation_blocked_by": ["front25_dynamics"],
    },
    "hypothesis_lab_early_subsegment_patch_search": {
        "cost_level": "medium",
        "auto_fallback_allowed": True,
        "factory_handoff_required": True,
        "requires_target": True,
        "requires_reference": True,
        "default_budget": "smoke",
        "targets_error_modes": ["early_segment", "transition_window"],
        "saturation_blocked_by": ["residual_corrector", "spectral_physics"],
    },
    "sem_window_gate_blend": {
        "cost_level": "light",
        "auto_fallback_allowed": True,
        "requires_target": False,
        "requires_reference": False,
        "default_budget": "smoke",
        "targets_error_modes": ["early_segment", "transition_window"],
        "saturation_blocked_by": ["local_strategy_skill"],
    },
    "sem_window_blend": {
        "cost_level": "light",
        "auto_fallback_allowed": True,
        "requires_target": False,
        "requires_reference": False,
        "default_budget": "smoke",
        "targets_error_modes": ["early_segment"],
        "saturation_blocked_by": ["local_strategy_skill"],
    },
    "fast_baseline_family": {
        "cost_level": "light",
        "auto_fallback_allowed": True,
        "requires_target": False,
        "requires_reference": False,
        "default_budget": "smoke",
        "targets_error_modes": ["baseline_control", "runtime"],
    },
    "sem_window_gate_blend_multi_search": {
        "cost_level": "heavy",
        "auto_fallback_allowed": False,
        "requires_target": True,
        "requires_reference": False,
        "default_budget": "explicit",
        "targets_error_modes": ["early_segment", "transition_window"],
    },
}


def get_candidate_generator_registry() -> dict[str, dict[str, Any]]:
    return {
        name: dict(metadata)
        for name, metadata in CANDIDATE_GENERATOR_REGISTRY.items()
    }
