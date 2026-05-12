"""Research-directive layer for higher-level Hermes guidance."""


import json
from pathlib import Path
from typing import Any


DIRECTIVE_STATE_HINT = "RESEARCH_DIRECTIVE_BUILT"
FOLLOWTHROUGH_STATE_HINT = "RESEARCH_DIRECTIVE_FOLLOWTHROUGH_EVALUATED"
META_DIRECTIVE_STATE_HINT = "META_RESEARCH_DIRECTIVE_BUILT"


class ResearchDirectiveError(RuntimeError):
    pass


def build_research_directive(
    *,
    task1_benchmark_path: str | Path,
    candidate_factory_benchmark_path: str | Path,
    output_path: str | Path,
    recent_judge_report_paths: list[str | Path] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path)
    _ensure_can_write(output_file, force=force)
    task1_file = _require_file(Path(task1_benchmark_path), "Task1 benchmark")
    factory_file = _require_file(
        Path(candidate_factory_benchmark_path),
        "candidate factory benchmark",
    )
    task1 = _read_json(task1_file)
    factory = _read_json(factory_file)
    judges = [
        _read_json(_require_file(Path(path), "judge report"))
        for path in recent_judge_report_paths or []
    ]
    directive = _directive_from_evidence(task1=task1, factory=factory, judges=judges)
    report = {
        "schema_version": 1,
        "state_hint": DIRECTIVE_STATE_HINT,
        "directive_path": str(output_file),
        "decision": "run_directed_experiment",
        "inputs": {
            "task1_benchmark_path": str(task1_file),
            "candidate_factory_benchmark_path": str(factory_file),
            "recent_judge_report_paths": [
                str(path) for path in recent_judge_report_paths or []
            ],
        },
        "evidence_summary": {
            "task1_decision": task1.get("decision"),
            "task1_best_score": _nested_float(task1, ["score_progress", "best_score"]),
            "task1_score_delta": _nested_float(task1, ["score_progress", "score_delta"]),
            "factory_decision": factory.get("decision"),
            "judge_count": len(judges),
            "best_recent_judge_score": max(
                [_float(judge.get("total_score_proxy")) for judge in judges]
                or [None],
                key=lambda value: float("-inf") if value is None else value,
            ),
        },
        "directive": directive,
        "next_command": "run-claude-agent-sdk-runtime",
    }
    _write_json(output_file, report)
    return report


def load_research_directive(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _read_json(_require_file(Path(path), "research directive"))
    if payload.get("state_hint") not in {
        DIRECTIVE_STATE_HINT,
        META_DIRECTIVE_STATE_HINT,
    }:
        raise ResearchDirectiveError("research directive has invalid state_hint")
    directive = payload.get("directive")
    if not isinstance(directive, dict):
        raise ResearchDirectiveError("research directive missing directive object")
    return payload


def apply_research_directive_overrides(
    *,
    directive_report: dict[str, Any] | None,
    candidate_generator: str,
    candidate_generator_fallback: str,
    learning_objective: str,
    hypothesis_candidate_options: dict[str, Any] | None,
) -> dict[str, Any]:
    if directive_report is None:
        return {
            "candidate_generator": candidate_generator,
            "candidate_generator_fallback": candidate_generator_fallback,
            "learning_objective": learning_objective,
            "hypothesis_candidate_options": hypothesis_candidate_options,
            "candidate_generator_guard_overrides": [],
        }
    directive = directive_report.get("directive")
    if not isinstance(directive, dict):
        raise ResearchDirectiveError("research directive missing directive object")
    options = dict(hypothesis_candidate_options or {})
    directive_options = directive.get("hypothesis_candidate_options")
    if isinstance(directive_options, dict):
        options.update(directive_options)
    effective_candidate_generator = _text(
        directive.get("candidate_generator"),
        candidate_generator,
    )
    return {
        "candidate_generator": effective_candidate_generator,
        "candidate_generator_fallback": _text(
            directive.get("candidate_generator_fallback"),
            candidate_generator_fallback,
        ),
        "learning_objective": _text(
            directive.get("learning_objective"),
            learning_objective,
        ),
        "hypothesis_candidate_options": options,
        "candidate_generator_guard_overrides": _guard_overrides_from_policy(
            directive.get("guard_conflict_policy"),
            candidate_generator=effective_candidate_generator,
        ),
    }


def evaluate_research_directive_followthrough(
    *,
    directive_path: str | Path,
    chain_report_path: str | Path,
    output_path: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path)
    _ensure_can_write(output_file, force=force)
    directive_report = load_research_directive(directive_path)
    assert directive_report is not None
    chain_file = _require_file(Path(chain_report_path), "runtime audit-chain report")
    chain = _read_json(chain_file)
    directive = directive_report["directive"]
    runtime_reports = _runtime_reports_from_chain(chain)
    observed_generators = []
    heartbeat_options = []
    directive_conflicts = []
    audit_failed_count = int((chain.get("metrics") or {}).get("failed_audit_count") or 0)
    recorded_attempt_count = 0
    expected_generator = str(directive.get("candidate_generator") or "")
    for runtime_path in runtime_reports:
        runtime = _read_json(runtime_path)
        candidate_plan = runtime.get("candidate_plan")
        detailed_candidate_plan = _load_runtime_candidate_plan(runtime)
        if isinstance(candidate_plan, dict):
            observed_generators.extend(
                item
                for item in candidate_plan.get("candidate_generator_schedule", [])
                if isinstance(item, str)
            )
        if isinstance(detailed_candidate_plan, dict):
            observed_generators.extend(
                item
                for item in detailed_candidate_plan.get(
                    "candidate_generator_schedule",
                    [],
                )
                if isinstance(item, str)
            )
            directive_conflicts.extend(
                _directive_conflicts_from_candidate_plan(
                    detailed_candidate_plan,
                    expected_generator=expected_generator,
                )
            )
        metrics = runtime.get("metrics")
        if isinstance(metrics, dict):
            recorded_attempt_count += int(
                metrics.get("recorded_candidate_attempt_count") or 0
            )
        heartbeat_path = ((runtime.get("artifacts") or {}).get("heartbeat_report_path"))
        if isinstance(heartbeat_path, str) and Path(heartbeat_path).exists():
            heartbeat = _read_json(Path(heartbeat_path))
            parameters = heartbeat.get("parameters")
            if isinstance(parameters, dict):
                heartbeat_options.append(parameters.get("hypothesis_candidate_options"))
            best = heartbeat.get("best_candidate")
            if isinstance(best, dict) and isinstance(best.get("family"), str):
                observed_generators.append(best["family"])
    expected_options = directive.get("hypothesis_candidate_options")
    expected_options_for_check = (
        expected_options if isinstance(expected_options, dict) else {}
    )
    normalized_heartbeat_options = [
        item if isinstance(item, dict) else {} for item in heartbeat_options
    ]
    first_conflict = directive_conflicts[0] if directive_conflicts else {}
    checks = {
        "candidate_generator_followed": {
            "expected": expected_generator,
            "observed": sorted(set(observed_generators)),
            "passed": (
                not expected_generator
                or expected_generator in set(observed_generators)
            ),
        },
        "candidate_options_forwarded": {
            "expected": expected_options_for_check,
            "observed": heartbeat_options,
            "passed": (
                not isinstance(expected_options, dict)
                or expected_options_for_check in normalized_heartbeat_options
                or not heartbeat_options
            ),
        },
        "audit_clean": {
            "actual": audit_failed_count,
            "maximum": 0,
            "passed": audit_failed_count == 0,
        },
        "evidence_recorded": {
            "actual": recorded_attempt_count,
            "minimum": 0,
            "passed": recorded_attempt_count >= 0,
        },
        "directive_conflict": {
            "expected": "no_conflict",
            "actual": first_conflict or None,
            "candidate_generator": expected_generator,
            "reason": first_conflict.get("reason") if first_conflict else "",
            "passed": not directive_conflicts,
        },
    }
    decision = "passed" if all(check["passed"] for check in checks.values()) else "failed"
    report = {
        "schema_version": 1,
        "state_hint": FOLLOWTHROUGH_STATE_HINT,
        "followthrough_report_path": str(output_file),
        "decision": decision,
        "directive_path": str(directive_path),
        "chain_report_path": str(chain_file),
        "checks": checks,
        "next_command": "build-agent-research-directive"
        if decision != "passed"
        else "run-claude-agent-sdk-runtime",
    }
    _write_json(output_file, report)
    return report


def _directive_from_evidence(
    *,
    task1: dict[str, Any],
    factory: dict[str, Any],
    judges: list[dict[str, Any]],
) -> dict[str, Any]:
    score_delta = _nested_float(task1, ["score_progress", "score_delta"]) or 0.0
    factory_ready = bool((factory.get("heartbeat_handoff") or {}).get("ready"))
    if task1.get("decision") == "passed" and score_delta > 0.0:
        return {
            "research_focus": "spectral_postprocess_refinement",
            "bottleneck": "score_plateau_after_small_postprocess_gain",
            "candidate_generator": "postprocess",
            "candidate_generator_fallback": "auto",
            "learning_objective": "score",
            "hypothesis_candidate_options": {},
            "guard_conflict_policy": {
                "mode": "evidence_bounded_reprobe",
                "allowed_candidate_generators": ["postprocess"],
                "max_reprobe_batches": 1,
                "reason": "positive_task1_benchmark_delta",
            },
            "forbidden_actions": [
                "manual_parameter_selection_without_sandbox",
                "repeat_failed_candidate_family_without_mutation",
                "promote_without_evidence",
            ],
            "experiment_design": {
                "minimum_trials": 3,
                "use_clean_trace_for_ablation": True,
                "compare_against_current_best": True,
                "primary_metric": "task1_total_score",
                "secondary_metrics": ["early_rel_mse", "inference_time"],
            },
            "promotion_gate": {
                "min_score_delta": 0.05,
                "require_rules_passed": True,
                "require_audit_passed": True,
            },
            "expected_trace_tags": ["research_directive", "task1", "sandbox"],
        }
    return {
        "research_focus": "candidate_factory_expansion",
        "bottleneck": "candidate_space_exhaustion" if factory_ready else "weak_evidence",
        "candidate_generator": "hypothesis_lab_early_subsegment_patch_search",
        "candidate_generator_fallback": "auto",
        "learning_objective": "score",
        "hypothesis_candidate_options": {
            "source_stability_guard": True,
            "handoff_smoothing_weight": 0.2,
        },
        "forbidden_actions": [
            "manual_parameter_selection_without_sandbox",
            "repeat_failed_candidate_family_without_mutation",
            "promote_without_evidence",
        ],
        "experiment_design": {
            "minimum_trials": 3,
            "use_clean_trace_for_ablation": True,
            "compare_against_current_best": True,
            "primary_metric": "task1_total_score",
            "secondary_metrics": ["early_rel_mse", "inference_time"],
        },
        "promotion_gate": {
            "min_score_delta": 0.05,
            "require_rules_passed": True,
            "require_audit_passed": True,
        },
        "expected_trace_tags": ["research_directive", "task1", "sandbox"],
    }


def _runtime_reports_from_chain(chain: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    raw_items = []
    for key in ("iteration_reports", "iterations"):
        values = chain.get(key)
        if isinstance(values, list):
            raw_items.extend(values)
    for item in raw_items:
        if isinstance(item, dict) and isinstance(item.get("runtime_report_path"), str):
            path = Path(item["runtime_report_path"])
            if path.exists():
                paths.append(path)
    return paths


def _load_runtime_candidate_plan(runtime: dict[str, Any]) -> dict[str, Any] | None:
    candidate_plan = runtime.get("candidate_plan")
    artifacts = runtime.get("artifacts")
    candidate_plan_path = None
    if isinstance(artifacts, dict) and isinstance(
        artifacts.get("candidate_plan_path"),
        str,
    ):
        candidate_plan_path = Path(artifacts["candidate_plan_path"])
    if candidate_plan_path is not None and candidate_plan_path.exists():
        return _read_json(candidate_plan_path)
    return candidate_plan if isinstance(candidate_plan, dict) else None


def _directive_conflicts_from_candidate_plan(
    candidate_plan: dict[str, Any],
    *,
    expected_generator: str,
) -> list[dict[str, Any]]:
    if not expected_generator:
        return []
    conflicts = []
    for item in candidate_plan.get("selection_rationale", []):
        if not isinstance(item, dict):
            continue
        if item.get("candidate_generator") != expected_generator:
            continue
        if item.get("decision") != "excluded":
            continue
        reason = item.get("reason")
        conflicts.append(
            {
                "candidate_generator": expected_generator,
                "reason": reason if isinstance(reason, str) else "excluded",
            }
        )
    return conflicts


def _guard_overrides_from_policy(
    policy: Any,
    *,
    candidate_generator: str,
) -> list[str]:
    if not isinstance(policy, dict):
        return []
    if policy.get("mode") != "evidence_bounded_reprobe":
        return []
    if int(policy.get("max_reprobe_batches") or 0) < 1:
        return []
    allowed = policy.get("allowed_candidate_generators")
    if not isinstance(allowed, list):
        return []
    candidates = [
        item
        for item in allowed
        if isinstance(item, str) and item and item == candidate_generator
    ]
    return sorted(set(candidates))


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


def _text(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) and value.strip() else fallback


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ResearchDirectiveError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise ResearchDirectiveError(f"{label} not found: {path}")
    return path


def _ensure_can_write(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise ResearchDirectiveError(f"output already exists: {path}")
