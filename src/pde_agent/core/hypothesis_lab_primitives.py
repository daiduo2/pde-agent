"""Composable primitives for Hypothesis Lab agent infrastructure.

Each primitive is an independently callable operation that works with a
HypothesisLabSession. Primitives replace the monolithic pipeline with
granular, observable, resumable steps that an agent can compose dynamically.

Usage:
    session = HypothesisLabSession.create(workspace="runs/exp_001")
    build_knowledge_cards_primitive(session, manual_report_paths=[...])
    generate_hypotheses_primitive(session)
    design_algorithm_blueprints_primitive(session)
    run_blueprint_sandbox_primitive(session, context_path="...", target_path="...")
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_lab_events import (
    EVENT_STAGE_COMPLETED,
    EVENT_STAGE_FAILED,
    EVENT_STAGE_STARTED,
)
from pde_agent.core.hypothesis_lab_knowledge import build_knowledge_cards
from pde_agent.core.hypothesis_lab_generation import generate_hypotheses
from pde_agent.core.hypothesis_lab_design import design_algorithm_blueprints
from pde_agent.core.hypothesis_lab_sandbox_runner import run_blueprint_sandbox
from pde_agent.core.hypothesis_lab_pareto import (
    _record_algorithm_candidate_outcome,
    _record_pareto_failure_memory,
    _executed_algorithm_candidate_evidence,
)
from pde_agent.core.hypothesis_lab_memory import (
    HypothesisLabMemory,
    get_memory_path_from_env,
)
from pde_agent.core.hypothesis_lab_session import HypothesisLabSession
from pde_agent.core.hypothesis_lab_utils import HypothesisLabError


class PrimitiveError(RuntimeError):
    pass


# -- Stage 1: Build Knowledge Cards --

def build_knowledge_cards_primitive(
    session: HypothesisLabSession,
    *,
    manual_report_paths: list[str | Path],
    adaptive_report_paths: list[str | Path] | None = None,
    failure_memory_paths: list[str | Path] | None = None,
    log_paths: list[str | Path] | None = None,
    checkpoint_paths: list[str | Path] | None = None,
    literature_paths: list[str | Path] | None = None,
    algorithm_candidate_paths: list[str | Path] | None = None,
    artifact_id: str = "knowledge_cards",
    force: bool = True,
) -> dict[str, Any]:
    """Build knowledge cards from input reports and store in session."""
    session.emit(
        event_type=EVENT_STAGE_STARTED,
        payload={"stage": "build_knowledge_cards", "artifact_id": artifact_id},
        tags=["stage", "cards"],
    )
    try:
        result = build_knowledge_cards(
            manual_report_paths=manual_report_paths,
            adaptive_report_paths=adaptive_report_paths,
            failure_memory_paths=failure_memory_paths,
            log_paths=log_paths,
            checkpoint_paths=checkpoint_paths,
            literature_paths=literature_paths,
            algorithm_candidate_paths=algorithm_candidate_paths,
            output_path=None,
            force=force,
        )
        session.put_artifact(
            artifact_type="knowledge_cards",
            artifact_id=artifact_id,
            payload=result,
        )
        session.emit(
            event_type=EVENT_STAGE_COMPLETED,
            payload={
                "stage": "build_knowledge_cards",
                "artifact_id": artifact_id,
                "card_count": result.get("card_count", 0),
            },
            tags=["stage", "cards"],
        )
        return result
    except Exception as exc:
        session.emit(
            event_type=EVENT_STAGE_FAILED,
            payload={
                "stage": "build_knowledge_cards",
                "artifact_id": artifact_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            tags=["stage", "cards", "error"],
        )
        raise PrimitiveError(f"build_knowledge_cards failed: {exc}") from exc


# -- Stage 2: Generate Hypotheses --

def generate_hypotheses_primitive(
    session: HypothesisLabSession,
    *,
    cards_artifact_id: str = "knowledge_cards",
    artifact_id: str = "hypotheses",
    force: bool = True,
) -> dict[str, Any]:
    """Generate hypotheses from knowledge cards stored in session."""
    session.emit(
        event_type=EVENT_STAGE_STARTED,
        payload={"stage": "generate_hypotheses", "artifact_id": artifact_id},
        tags=["stage", "hypotheses"],
    )
    try:
        cards = session.get_artifact("knowledge_cards", cards_artifact_id)
        if cards is None:
            raise PrimitiveError(
                f"knowledge_cards artifact '{cards_artifact_id}' not found in session"
            )
        result = generate_hypotheses(
            cards=cards,
            output_path=None,
            force=force,
        )
        session.put_artifact(
            artifact_type="hypotheses",
            artifact_id=artifact_id,
            payload=result,
        )
        session.emit(
            event_type=EVENT_STAGE_COMPLETED,
            payload={
                "stage": "generate_hypotheses",
                "artifact_id": artifact_id,
                "hypothesis_count": result.get("hypothesis_count", 0),
            },
            tags=["stage", "hypotheses"],
        )
        return result
    except Exception as exc:
        session.emit(
            event_type=EVENT_STAGE_FAILED,
            payload={
                "stage": "generate_hypotheses",
                "artifact_id": artifact_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            tags=["stage", "hypotheses", "error"],
        )
        raise PrimitiveError(f"generate_hypotheses failed: {exc}") from exc


# -- Stage 3: Design Algorithm Blueprints --

def design_algorithm_blueprints_primitive(
    session: HypothesisLabSession,
    *,
    hypotheses_artifact_id: str = "hypotheses",
    artifact_id: str = "blueprints",
    force: bool = True,
) -> dict[str, Any]:
    """Design blueprints from hypotheses stored in session."""
    session.emit(
        event_type=EVENT_STAGE_STARTED,
        payload={"stage": "design_algorithm_blueprints", "artifact_id": artifact_id},
        tags=["stage", "blueprints"],
    )
    try:
        hypotheses = session.get_artifact("hypotheses", hypotheses_artifact_id)
        if hypotheses is None:
            raise PrimitiveError(
                f"hypotheses artifact '{hypotheses_artifact_id}' not found in session"
            )
        result = design_algorithm_blueprints(
            hypotheses=hypotheses,
            output_path=None,
            force=force,
        )
        session.put_artifact(
            artifact_type="blueprints",
            artifact_id=artifact_id,
            payload=result,
        )
        session.emit(
            event_type=EVENT_STAGE_COMPLETED,
            payload={
                "stage": "design_algorithm_blueprints",
                "artifact_id": artifact_id,
                "blueprint_count": result.get("blueprint_count", 0),
            },
            tags=["stage", "blueprints"],
        )
        return result
    except Exception as exc:
        session.emit(
            event_type=EVENT_STAGE_FAILED,
            payload={
                "stage": "design_algorithm_blueprints",
                "artifact_id": artifact_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            tags=["stage", "blueprints", "error"],
        )
        raise PrimitiveError(f"design_algorithm_blueprints failed: {exc}") from exc


# -- Stage 4: Run Blueprint Sandbox --

def run_blueprint_sandbox_primitive(
    session: HypothesisLabSession,
    *,
    blueprint: dict[str, Any] | None = None,
    blueprints_artifact_id: str = "blueprints",
    blueprint_index: int = 0,
    context_path: str | Path,
    target_path: str | Path,
    reference_path: str | Path | None = None,
    reference_sample_limit: int | None = None,
    artifact_id: str = "sandbox",
    force: bool = True,
) -> dict[str, Any]:
    """Run sandbox for a blueprint.

    If ``blueprint`` is not provided directly, reads from session blueprints
    at the given ``blueprint_index``.
    """
    session.emit(
        event_type=EVENT_STAGE_STARTED,
        payload={
            "stage": "run_blueprint_sandbox",
            "artifact_id": artifact_id,
            "blueprint_index": blueprint_index,
        },
        tags=["stage", "sandbox"],
    )
    try:
        if blueprint is None:
            blueprints = session.get_artifact("blueprints", blueprints_artifact_id)
            if blueprints is None:
                raise PrimitiveError(
                    f"blueprints artifact '{blueprints_artifact_id}' not found in session"
                )
            blueprint_list = blueprints.get("blueprints", [])
            if not blueprint_list:
                raise PrimitiveError("no blueprints found")
            if blueprint_index >= len(blueprint_list):
                raise PrimitiveError(
                    f"blueprint_index {blueprint_index} out of range ({len(blueprint_list)} blueprints)"
                )
            blueprint = blueprint_list[blueprint_index]

        result = run_blueprint_sandbox(
            blueprint=blueprint,
            context_path=Path(context_path),
            target_path=Path(target_path),
            reference_path=Path(reference_path) if reference_path is not None else None,
            reference_sample_limit=reference_sample_limit,
            output_dir=session.workspace / "sandbox_runs",
            submission_id=session.session_id,
            force=force,
        )
        session.put_artifact(
            artifact_type="sandbox",
            artifact_id=artifact_id,
            payload=result,
        )
        best = result.get("best_candidate", {})
        verifier = result.get("verifier", {})
        post_mortem = result.get("post_mortem", {})
        session.emit(
            event_type=EVENT_STAGE_COMPLETED,
            payload={
                "stage": "run_blueprint_sandbox",
                "artifact_id": artifact_id,
                "best_candidate_id": best.get("candidate_id"),
                "best_score": best.get("task1_total_score_proxy"),
                "blueprint_id": result.get("blueprint_id"),
                "candidate_family": result.get("candidate_family") or best.get("family"),
                "algorithm_candidate_id": best.get("candidate_id"),
                "score_delta": verifier.get("score_delta_vs_baseline"),
                "outcome": (
                    "ok"
                    if verifier.get("rules_passed")
                    else "failed"
                ),
                "worst_segment": post_mortem.get("worst_segment"),
                "root_cause_hint": post_mortem.get("root_cause_hint"),
                "context_preserved": post_mortem.get("context_preserved"),
                "inference_time_seconds": post_mortem.get("inference_time_seconds"),
            },
            tags=["stage", "sandbox"],
        )
        return result
    except Exception as exc:
        session.emit(
            event_type=EVENT_STAGE_FAILED,
            payload={
                "stage": "run_blueprint_sandbox",
                "artifact_id": artifact_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            tags=["stage", "sandbox", "error"],
        )
        raise PrimitiveError(f"run_blueprint_sandbox failed: {exc}") from exc


# -- Stage 5: Record Sandbox Outcome --

def record_sandbox_outcome_primitive(
    session: HypothesisLabSession,
    *,
    cards_artifact_id: str = "knowledge_cards",
    hypotheses_artifact_id: str = "hypotheses",
    blueprints_artifact_id: str = "blueprints",
    sandbox_artifact_id: str = "sandbox",
    failure_memory_output_path: str | Path | None = None,
    artifact_id: str = "outcome",
    force: bool = True,
) -> dict[str, Any]:
    """Record candidate outcome and Pareto failure memory from sandbox results."""
    session.emit(
        event_type=EVENT_STAGE_STARTED,
        payload={"stage": "record_sandbox_outcome", "artifact_id": artifact_id},
        tags=["stage", "outcome"],
    )
    try:
        cards = session.get_artifact("knowledge_cards", cards_artifact_id)
        hypotheses = session.get_artifact("hypotheses", hypotheses_artifact_id)
        blueprints = session.get_artifact("blueprints", blueprints_artifact_id)
        sandbox = session.get_artifact("sandbox", sandbox_artifact_id)

        if sandbox is None:
            raise PrimitiveError(
                f"sandbox artifact '{sandbox_artifact_id}' not found in session"
            )

        result: dict[str, Any] = {"sandbox_artifact_id": sandbox_artifact_id}

        # Find the executable blueprint that was sandboxed
        executable_blueprint = None
        if blueprints is not None:
            for bp in blueprints.get("blueprints", []):
                if isinstance(bp, dict) and bp.get("blueprint_id") == sandbox.get("blueprint_id"):
                    executable_blueprint = bp
                    break

        # Record candidate outcome
        if cards is not None and hypotheses is not None and executable_blueprint is not None:
            try:
                candidate_outcome = _record_algorithm_candidate_outcome(
                    cards=cards,
                    hypotheses=hypotheses,
                    executable_blueprint=executable_blueprint,
                    sandbox=sandbox,
                    failure_memory_output_path=failure_memory_output_path,
                    force=force,
                )
                if candidate_outcome is not None:
                    result["candidate_outcome"] = candidate_outcome
            except Exception:
                pass  # Non-fatal: candidate outcome is best-effort

        # Record Pareto failure memory
        try:
            algorithm_candidate_evidence = None
            if cards is not None and hypotheses is not None and executable_blueprint is not None:
                algorithm_candidate_evidence = _executed_algorithm_candidate_evidence(
                    cards=cards,
                    hypotheses=hypotheses,
                    executable_blueprint=executable_blueprint,
                )
            pareto_memory = _record_pareto_failure_memory(
                sandbox=sandbox,
                failure_memory_output_path=failure_memory_output_path,
                algorithm_candidate_evidence=algorithm_candidate_evidence,
                force=force,
            )
            if pareto_memory is not None:
                result["pareto_memory"] = pareto_memory
        except Exception:
            pass  # Non-fatal: pareto memory is best-effort

        session.put_artifact(
            artifact_type="outcome",
            artifact_id=artifact_id,
            payload=result,
        )

        # -- Auto-sync to global memory (best-effort) --
        _sync_outcome_to_memory(session, result, sandbox)

        session.emit(
            event_type=EVENT_STAGE_COMPLETED,
            payload={
                "stage": "record_sandbox_outcome",
                "artifact_id": artifact_id,
                "has_candidate_outcome": "candidate_outcome" in result,
                "has_pareto_memory": "pareto_memory" in result,
            },
            tags=["stage", "outcome"],
        )
        return result
    except Exception as exc:
        session.emit(
            event_type=EVENT_STAGE_FAILED,
            payload={
                "stage": "record_sandbox_outcome",
                "artifact_id": artifact_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            tags=["stage", "outcome", "error"],
        )
        raise PrimitiveError(f"record_sandbox_outcome failed: {exc}") from exc


# -- Orchestrator: run full pipeline on session --

def _sync_outcome_to_memory(
    session: HypothesisLabSession,
    result: dict[str, Any],
    sandbox: dict[str, Any] | None,
) -> None:
    """Best-effort sync of sandbox outcome to global memory."""
    memory_path = get_memory_path_from_env()
    if memory_path is None:
        return
    try:
        memory = HypothesisLabMemory(memory_path)
        outcome_payload = {
            "candidate": sandbox.get("best_candidate") if sandbox else None,
            "sandbox_artifact_id": result.get("sandbox_artifact_id"),
            "has_candidate_outcome": "candidate_outcome" in result,
            "has_pareto_memory": "pareto_memory" in result,
        }
        memory.record_experiment(session.session_id, outcome_payload)

        # Also record failure patterns if present
        pareto_memory = result.get("pareto_memory", {})
        events = pareto_memory.get("pareto_learning_events", [])
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("outcome") in {"failed", "rejected", "blocked"}:
                family = str(event.get("candidate_family", "unknown"))
                memory.record_failure_pattern(
                    session.session_id,
                    family,
                    {
                        "decision": event.get("decision"),
                        "reason_codes": event.get("reason_codes", []),
                        "score_delta": event.get("score_delta_vs_baseline"),
                    },
                )

        # Record candidate if we have one
        best = sandbox.get("best_candidate") if sandbox else None
        if isinstance(best, dict):
            memory.record_candidate(session.session_id, best)
    except Exception:
        pass  # Best-effort: memory failures must not break the primitive


def run_hypothesis_lab_on_session(
    session: HypothesisLabSession,
    *,
    manual_report_paths: list[str | Path],
    adaptive_report_paths: list[str | Path] | None = None,
    log_paths: list[str | Path] | None = None,
    checkpoint_paths: list[str | Path] | None = None,
    literature_paths: list[str | Path] | None = None,
    algorithm_candidate_paths: list[str | Path] | None = None,
    failure_memory_paths: list[str | Path] | None = None,
    context_path: str | Path,
    target_path: str | Path,
    reference_path: str | Path | None = None,
    reference_sample_limit: int | None = None,
    failure_memory_output_path: str | Path | None = None,
    force: bool = True,
) -> dict[str, Any]:
    """Run the full hypothesis-lab pipeline on a session.

    This is the session-aware equivalent of ``run_hypothesis_lab()``.
    Each stage is checkpointed and can be resumed.
    """
    # Check for existing checkpoint
    checkpoint = session.load_checkpoint()
    if checkpoint is not None:
        completed = set(checkpoint.completed_steps)
    else:
        completed = set()

    def _maybe_run(stage_name: str, func, *args, **kwargs):
        if stage_name in completed:
            session.emit(
                event_type="stage_skipped",
                payload={"stage": stage_name, "reason": "already_completed"},
                tags=["stage", "skip"],
            )
            return session.get_latest_artifact(stage_name)["payload"]
        result = func(*args, **kwargs)
        completed.add(stage_name)
        session.save_checkpoint(
            stage=stage_name,
            stage_data={"last_stage": stage_name},
            completed_steps=sorted(completed),
        )
        return result

    cards = _maybe_run(
        "build_knowledge_cards",
        build_knowledge_cards_primitive,
        session,
        manual_report_paths=manual_report_paths,
        adaptive_report_paths=adaptive_report_paths,
        log_paths=log_paths,
        checkpoint_paths=checkpoint_paths,
        literature_paths=literature_paths,
        algorithm_candidate_paths=algorithm_candidate_paths,
        failure_memory_paths=failure_memory_paths,
        force=force,
    )

    hypotheses = _maybe_run(
        "generate_hypotheses",
        generate_hypotheses_primitive,
        session,
        force=force,
    )

    blueprints = _maybe_run(
        "design_algorithm_blueprints",
        design_algorithm_blueprints_primitive,
        session,
        force=force,
    )

    sandbox = _maybe_run(
        "run_blueprint_sandbox",
        run_blueprint_sandbox_primitive,
        session,
        context_path=context_path,
        target_path=target_path,
        reference_path=reference_path,
        reference_sample_limit=reference_sample_limit,
        force=force,
    )

    outcome = _maybe_run(
        "record_sandbox_outcome",
        record_sandbox_outcome_primitive,
        session,
        failure_memory_output_path=failure_memory_output_path,
        force=force,
    )

    report = {
        "schema_version": 1,
        "state_hint": "HYPOTHESIS_LAB_RAN",
        "session_id": session.session_id,
        "cards_count": cards.get("card_count", 0),
        "hypotheses_count": hypotheses.get("hypothesis_count", 0),
        "blueprints_count": blueprints.get("blueprint_count", 0),
        "sandbox": sandbox,
        "outcome": outcome,
    }
    session.put_artifact(
        artifact_type="report",
        artifact_id="hypothesis_lab_report",
        payload=report,
    )
    session.clear_checkpoint()
    return report
