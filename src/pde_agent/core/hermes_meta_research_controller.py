"""Meta-research controller for Hermes research organization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STATE_HINT = "HERMES_META_RESEARCH_DIRECTIVE_BUILT"


class HermesMetaResearchControllerError(RuntimeError):
    pass


def build_hermes_meta_research_directive(
    *,
    followthrough_report_path: str | Path,
    chain_report_path: str | Path,
    task1_benchmark_path: str | Path,
    candidate_factory_benchmark_path: str | Path,
    output_path: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path)
    _ensure_can_write(output_file, force=force)
    followthrough_file = _require_file(
        Path(followthrough_report_path),
        "research directive followthrough report",
    )
    chain_file = _require_file(Path(chain_report_path), "runtime audit-chain report")
    task1_file = _require_file(Path(task1_benchmark_path), "Task1 benchmark")
    factory_file = _require_file(
        Path(candidate_factory_benchmark_path),
        "candidate factory benchmark",
    )
    followthrough = _read_json(followthrough_file)
    chain = _read_json(chain_file)
    task1 = _read_json(task1_file)
    factory = _read_json(factory_file)
    assessment = _research_assessment(
        followthrough=followthrough,
        chain=chain,
        task1=task1,
        factory=factory,
    )
    decision, directive = _next_directive(assessment)
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "directive_path": str(output_file),
        "decision": decision,
        "inputs": {
            "followthrough_report_path": str(followthrough_file),
            "chain_report_path": str(chain_file),
            "task1_benchmark_path": str(task1_file),
            "candidate_factory_benchmark_path": str(factory_file),
        },
        "research_assessment": assessment,
        "directive": directive,
        "next_command": "run-hermes-runtime-audit-chain",
    }
    _write_json(output_file, report)
    return report


def _research_assessment(
    *,
    followthrough: dict[str, Any],
    chain: dict[str, Any],
    task1: dict[str, Any],
    factory: dict[str, Any],
) -> dict[str, Any]:
    checks = followthrough.get("checks")
    checks = checks if isinstance(checks, dict) else {}
    candidate_followed = _check_passed(checks, "candidate_generator_followed")
    audit_clean = _check_passed(checks, "audit_clean")
    directive_conflict = checks.get("directive_conflict")
    directive_conflict = directive_conflict if isinstance(directive_conflict, dict) else {}
    expected_generator = _text(
        (checks.get("candidate_generator_followed") or {}).get("expected")
        if isinstance(checks.get("candidate_generator_followed"), dict)
        else None
    )
    runtime_reports = [_read_json(path) for path in _runtime_report_paths(chain)]
    blocked_candidate_families = _blocked_candidate_families(runtime_reports)
    heartbeat_reports = [
        _read_json(path)
        for path in [_heartbeat_path(runtime) for runtime in runtime_reports]
        if path is not None and path.exists()
    ]
    batch = _last_batch(heartbeat_reports)
    diagnostics = batch.get("candidate_diagnostics") if isinstance(batch, dict) else {}
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    return {
        "followthrough_decision": _text(followthrough.get("decision")),
        "candidate_generator_followed": candidate_followed,
        "audit_clean": audit_clean,
        "directive_conflict_reason": _text(directive_conflict.get("reason")),
        "expected_candidate_generator": expected_generator,
        "last_candidate_generator": _text(batch.get("candidate_generator"))
        if isinstance(batch, dict)
        else expected_generator,
        "attempted_candidate_count": int(
            diagnostics.get("attempted_candidate_count") or 0
        ),
        "best_attempted_delta": _float(
            diagnostics.get("best_attempted_delta_vs_initial_objective")
        ),
        "chain_decision": _text(chain.get("decision")),
        "chain_score_delta": _float((chain.get("metrics") or {}).get("score_delta"))
        if isinstance(chain.get("metrics"), dict)
        else None,
        "candidate_space_exhausted": _bool_metric(chain, "candidate_space_exhausted_count"),
        "task1_score_delta": _nested_float(task1, ["score_progress", "score_delta"]),
        "task1_stagnation_count": int(
            ((task1.get("score_progress") or {}).get("stagnation_count") or 0)
        )
        if isinstance(task1.get("score_progress"), dict)
        else 0,
        "candidate_factory_ready": bool(
            (factory.get("heartbeat_handoff") or {}).get("ready")
        )
        if isinstance(factory.get("heartbeat_handoff"), dict)
        else False,
        "candidate_factory_mechanism_count": int(
            ((factory.get("candidate_space") or {}).get("mechanism_count") or 0)
        )
        if isinstance(factory.get("candidate_space"), dict)
        else 0,
        "blocked_candidate_families": blocked_candidate_families,
    }


def _next_directive(assessment: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    conflict_reason = assessment.get("directive_conflict_reason")
    expected_generator = _text(assessment.get("expected_candidate_generator"))
    if (
        assessment.get("followthrough_decision") == "failed"
        and conflict_reason == "blocked_by_doom_loop_guard"
        and expected_generator
    ):
        return (
            "resolve_directive_guard_conflict",
            {
                "research_focus": "directive_followthrough_repair",
                "bottleneck": "directive_blocked_by_doom_loop_guard",
                "candidate_generator": expected_generator,
                "candidate_generator_fallback": "auto",
                "learning_objective": "score",
                "hypothesis_candidate_options": {},
                "guard_conflict_policy": {
                    "mode": "evidence_bounded_reprobe",
                    "allowed_candidate_generators": [expected_generator],
                    "max_reprobe_batches": 1,
                    "reason": "meta_controller_guard_conflict_resolution",
                },
                "experiment_design": {
                    "minimum_trials": 1,
                    "primary_metric": "task1_total_score",
                    "compare_against_current_best": True,
                },
                "promotion_gate": {
                    "min_score_delta": 0.05,
                    "require_audit_passed": True,
                    "require_rules_passed": True,
                },
            },
        )
    if (
        assessment.get("candidate_generator_followed") is True
        and (assessment.get("best_attempted_delta") or 0.0) < 0.0
        and assessment.get("candidate_factory_ready") is True
    ):
        return (
            "expand_candidate_space",
            {
                "research_focus": "candidate_factory_expansion",
                "bottleneck": "followed_directive_regressed_without_score_gain",
                "candidate_generator": _expansion_candidate_generator(assessment),
                "candidate_generator_fallback": "auto",
                "learning_objective": "score",
                "hypothesis_candidate_options": {
                    "source_stability_guard": True,
                    "handoff_smoothing_weight": 0.2,
                },
                "experiment_design": {
                    "minimum_trials": 2,
                    "primary_metric": "task1_total_score",
                    "secondary_metrics": ["early_rel_mse", "inference_time"],
                    "compare_against_current_best": True,
                },
                "promotion_gate": {
                    "min_score_delta": 0.05,
                    "require_audit_passed": True,
                    "require_rules_passed": True,
                },
            },
        )
    return (
        "continue_directed_experiment",
        {
            "research_focus": "directed_score_refinement",
            "bottleneck": "insufficient_evidence_to_change_family",
            "candidate_generator": expected_generator or "postprocess",
            "candidate_generator_fallback": "auto",
            "learning_objective": "score",
            "hypothesis_candidate_options": {},
            "experiment_design": {
                "minimum_trials": 1,
                "primary_metric": "task1_total_score",
                "compare_against_current_best": True,
            },
            "promotion_gate": {
                "min_score_delta": 0.05,
                "require_audit_passed": True,
                "require_rules_passed": True,
            },
        },
    )


def _runtime_report_paths(chain: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for key in ("iteration_reports", "iterations"):
        values = chain.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict) and isinstance(item.get("runtime_report_path"), str):
                path = Path(item["runtime_report_path"])
                if path.exists():
                    paths.append(path)
    return paths


def _expansion_candidate_generator(assessment: dict[str, Any]) -> str:
    blocked = {
        item
        for item in assessment.get("blocked_candidate_families", [])
        if isinstance(item, str)
    }
    for candidate in (
        "hypothesis_lab_early_subsegment_patch_search",
        "hypothesis_lab_front25_dynamics_smoke",
    ):
        if candidate not in blocked:
            return candidate
    return "hypothesis_lab_front25_dynamics_smoke"


def _blocked_candidate_families(runtime_reports: list[dict[str, Any]]) -> list[str]:
    blocked: set[str] = set()
    for runtime in runtime_reports:
        guard = runtime.get("guard")
        if isinstance(guard, dict) and isinstance(
            guard.get("blocked_candidate_families"),
            list,
        ):
            blocked.update(
                item
                for item in guard["blocked_candidate_families"]
                if isinstance(item, str) and item
            )
        candidate_plan = runtime.get("candidate_plan")
        if isinstance(candidate_plan, dict) and isinstance(
            candidate_plan.get("blocked_candidate_families"),
            list,
        ):
            blocked.update(
                item
                for item in candidate_plan["blocked_candidate_families"]
                if isinstance(item, str) and item
            )
    return sorted(blocked)


def _heartbeat_path(runtime: dict[str, Any]) -> Path | None:
    artifacts = runtime.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    value = artifacts.get("heartbeat_report_path")
    return Path(value) if isinstance(value, str) else None


def _last_batch(heartbeats: list[dict[str, Any]]) -> dict[str, Any]:
    for heartbeat in reversed(heartbeats):
        batches = heartbeat.get("batches")
        if isinstance(batches, list):
            for item in reversed(batches):
                if isinstance(item, dict):
                    return item
    return {}


def _check_passed(checks: dict[str, Any], name: str) -> bool:
    check = checks.get(name)
    return bool(check.get("passed")) if isinstance(check, dict) else False


def _bool_metric(payload: dict[str, Any], key: str) -> bool:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return False
    value = metrics.get(key)
    return bool(value)


def _nested_float(payload: dict[str, Any], keys: list[str]) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _float(value)


def _float(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HermesMetaResearchControllerError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise HermesMetaResearchControllerError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise HermesMetaResearchControllerError(f"{label} does not exist: {path}")
    return path


def _ensure_can_write(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise HermesMetaResearchControllerError(f"meta directive exists: {path}")
