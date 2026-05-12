from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_lab_utils import HypothesisLabError


def _linear_trend_spectral_card(
    report_file: Path,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    params = candidate.get("params") if isinstance(candidate.get("params"), dict) else {}
    return {
        "card_id": f"manual_{candidate.get('candidate_id', 'linear_trend_spectral')}",
        "source_type": "manual_tuning_report",
        "source_path": str(report_file),
        "claim": (
            "Damped negative linear trend extrapolation followed by spectral "
            "smoothing can improve long-horizon 1D Burgers score over persistence."
        ),
        "procedural_heuristic": (
            "Use the last context slope with a small negative gain, exponential "
            "decay, and spectral smoothing on predicted future steps only."
        ),
        "preconditions": [
            "1D Burgers",
            "persistence plateau",
            "late-window RMSE dominates local objective",
        ],
        "implementation_hooks": ["linear_trend_spectral"],
        "expected_observables": [
            "higher segment score",
            "lower late RMSE",
            "context copy remains exact",
        ],
        "known_risks": [
            "can overfit validation viscosity",
            "early Rel-MSE may remain poor",
            "strong smoothing can suppress shocks",
        ],
        "evidence_status": "tested",
        "evidence": {
            "candidate_id": candidate.get("candidate_id"),
            "params": params,
            "segment_score": candidate.get("segment_score"),
            "task1_total_fast_proxy": candidate.get("task1_total_fast_proxy"),
            "late_rmse": candidate.get("segments", {}).get("late", {}).get("rmse")
            if isinstance(candidate.get("segments"), dict)
            else None,
        },
    }


def _external_blend_card(report_file: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "card_id": f"manual_{candidate.get('candidate_id', 'external_blend')}",
        "source_type": "manual_tuning_report",
        "source_path": str(report_file),
        "claim": (
            "A weak neural operator can still provide useful low-weight residual "
            "signal when blended with persistence."
        ),
        "procedural_heuristic": (
            "Blend only a small fraction of the weak model rollout into the fast "
            "baseline and account for full upstream inference time."
        ),
        "preconditions": ["weak model exists", "runtime budget remains feasible"],
        "implementation_hooks": ["external_residual_blend"],
        "expected_observables": ["lower late RMSE", "runtime tradeoff visible"],
        "known_risks": ["upstream model inference can erase score gains"],
        "evidence_status": "seeded",
        "evidence": {
            "candidate_id": candidate.get("candidate_id"),
            "params": candidate.get("params"),
            "segment_score": candidate.get("segment_score"),
            "task1_total_fast_proxy": candidate.get("task1_total_fast_proxy"),
        },
    }


def _checkpoint_card(checkpoint_file: Path) -> dict[str, Any]:
    if not checkpoint_file.exists():
        raise HypothesisLabError(f"checkpoint artifact not found: {checkpoint_file}")
    model_family = _checkpoint_family(checkpoint_file)
    return {
        "card_id": f"checkpoint_{checkpoint_file.stem}",
        "source_type": "official_checkpoint",
        "source_path": str(checkpoint_file),
        "claim": (
            f"Official PDEBench {model_family} checkpoint should be evaluated as "
            "a separate model family when persistence-style postprocessing plateaus."
        ),
        "procedural_heuristic": (
            "Load the checkpoint through the audited backend, run local validation "
            "rollout, then calibrate horizon blending or short fine-tuning only "
            "if runtime remains under the 2 minute inference budget."
        ),
        "preconditions": [
            "official checkpoint file is available",
            "Hermes postprocessing family has plateaued",
            "torch backend is available when executing the checkpoint pipeline",
        ],
        "implementation_hooks": ["official_checkpoint_calibration"],
        "expected_observables": [
            "candidate beats persistence-like postprocessing on early or middle Rel-MSE",
            "inference time remains under 120 seconds",
            "context prefix remains exact",
        ],
        "known_risks": [
            "checkpoint schema may not match local adapter",
            "UNet checkpoint can exceed runtime budget",
            "short fine-tuning can overfit validation windows",
        ],
        "evidence_status": "available",
        "evidence": {
            "checkpoint_path": str(checkpoint_file),
            "model_family": model_family,
        },
    }


def _checkpoint_family(checkpoint_file: Path) -> str:
    name = checkpoint_file.name.lower()
    if "unet" in name:
        return "UNet"
    if "fno" in name:
        return "FNO"
    return "NeuralOperator"


def _literature_note_card(note_file: Path) -> dict[str, Any] | None:
    if not note_file.exists():
        raise HypothesisLabError(f"literature artifact not found: {note_file}")
    text = note_file.read_text(encoding="utf-8")
    expansion_card = _literature_expansion_report_card(note_file, text)
    if expansion_card is not None:
        return expansion_card
    fields = _parse_literature_note(text)
    hook = fields.get("hook")
    claim = fields.get("claim")
    if not hook or not claim:
        return None
    return {
        "card_id": f"literature_{note_file.stem}",
        "source_type": "literature_note",
        "source_path": str(note_file),
        "claim": claim,
        "procedural_heuristic": fields.get("mechanism", claim),
        "preconditions": ["local literature note has an implementation hook"],
        "implementation_hooks": [hook],
        "expected_observables": [fields.get("observable", "sandbox score improves")],
        "known_risks": [fields.get("risk", "literature claim may not transfer")],
        "evidence_status": "seeded",
        "evidence": {
            "title": fields.get("title"),
            "hook": hook,
        },
    }


def _parse_literature_note(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized = key.strip().lower()
        if normalized in {"title", "claim", "mechanism", "hook", "observable", "risk"}:
            fields[normalized] = value.strip()
    return fields


def _literature_expansion_report_card(
    note_file: Path,
    text: str,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("state_hint") != "LITERATURE_STRATEGY_EXPANSION_RAN":
        return None
    skipped = [
        item
        for item in payload.get("skipped_candidates", [])
        if isinstance(item, dict)
    ]
    blocked = [
        item
        for item in payload.get("actions", [])
        if isinstance(item, dict) and item.get("executable") is False
    ]
    if not skipped and not blocked:
        return None
    skipped_ids = [
        str(item.get("algorithm_candidate_id"))
        for item in skipped
        if item.get("algorithm_candidate_id")
    ]
    blocked_ids = [
        str(item.get("algorithm_candidate_id"))
        for item in blocked
        if item.get("algorithm_candidate_id")
    ]
    return {
        "card_id": f"literature_expansion_{note_file.stem}",
        "source_type": "literature_expansion_report",
        "source_path": str(note_file),
        "claim": (
            "Hermes literature expansion should avoid failed literature "
            "candidates and prioritize candidates that can enter a bounded sandbox."
        ),
        "procedural_heuristic": (
            "Convert failed or blocked literature-candidate outcomes into "
            "negative memory before generating another hypothesis batch."
        ),
        "preconditions": [
            "Hermes literature expansion has produced skipped or blocked candidates"
        ],
        "implementation_hooks": ["avoid_failed_literature_candidate"],
        "expected_observables": [
            "future candidate rankings skip remembered failures",
            "Hermes does not repeat the same literature-candidate failure",
        ],
        "known_risks": [
            "skipped="
            + ",".join(skipped_ids or ["none"])
            + "; blocked="
            + ",".join(blocked_ids or ["none"])
        ],
        "evidence_status": "negative_memory",
        "evidence": {
            "skipped_candidate_ids": skipped_ids,
            "blocked_candidate_ids": blocked_ids,
        },
    }
