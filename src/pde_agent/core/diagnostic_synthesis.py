from __future__ import annotations

import json
from pathlib import Path


DIAGNOSTIC_MODE = "synthetic_pde_diagnostic"


def write_json(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def spectrum_payload(run_id: str, rollout: dict[str, object]) -> dict[str, object]:
    _ = rollout
    return {
        "run_id": run_id,
        "mode": DIAGNOSTIC_MODE,
        "low_frequency_error": 0.024,
        "mid_frequency_error": 0.052,
        "high_frequency_error": 0.141,
        "spectral_drift_score": 0.37,
        "shock_region_proxy": 0.68,
        "dominant_spectral_issue": "excessive_smoothing",
    }


def physics_payload(run_id: str, rollout: dict[str, object]) -> dict[str, object]:
    _ = rollout
    return {
        "run_id": run_id,
        "mode": DIAGNOSTIC_MODE,
        "mass_drift": 0.006,
        "energy_decay_ratio": 0.82,
        "total_variation_ratio": 0.71,
        "nonphysical_oscillation": False,
        "blowup_like_behavior": False,
        "excessive_smoothing": True,
        "physics_issue": "overdissipation",
    }


def interpretation_payload(
    *,
    run_id: str,
    rollout: dict[str, object],
    spectrum: dict[str, object],
    physics: dict[str, object],
) -> dict[str, object]:
    _ = rollout
    return {
        "run_id": run_id,
        "mode": DIAGNOSTIC_MODE,
        "dominant_failure_mode": "excessive_smoothing",
        "evidence": [
            f"high_frequency_error={spectrum['high_frequency_error']}",
            f"dominant_spectral_issue={spectrum['dominant_spectral_issue']}",
            f"physics_issue={physics['physics_issue']}",
            f"total_variation_ratio={physics['total_variation_ratio']}",
        ],
        "confidence": 0.82,
        "rejected_explanations": [
            "blowup_like_behavior",
            "nonphysical_oscillation",
        ],
        "recommended_focus": (
            "restore_high_frequency_structure_without_destabilizing_rollout"
        ),
    }


def hypothesis_payload(run_id: str, interpretation: dict[str, object]) -> dict[str, object]:
    _ = interpretation
    return {
        "run_id": run_id,
        "mode": DIAGNOSTIC_MODE,
        "selected_hypothesis": "add_spectral_consistency_loss",
        "rationale": (
            "The synthetic diagnostic indicates excessive smoothing: high-frequency "
            "error is dominant while physics checks show stable overdissipation."
        ),
        "expected_effect": (
            "reduce high-frequency damping while preserving rollout stability"
        ),
        "risk": "may amplify oscillations near shocks if overweighted",
        "next_experiment_scope": "read_only_proposal_no_training",
    }
