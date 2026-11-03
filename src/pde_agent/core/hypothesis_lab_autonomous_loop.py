"""Autonomous execution engine for the Hypothesis Lab pipeline.

Runs the full hypothesis-lab pipeline without LLM round-trips for
deterministic transitions.  Integrates with existing primitives,
queries, doom-loop detection, and events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pde_agent.core.hypothesis_doom_loop import check_hypothesis_doom_loop
from pde_agent.core.hypothesis_lab_action_dispatcher import (
    ActionDispatchError,
    _dispatch_action,
    is_deterministic_action,
)
from pde_agent.core.hypothesis_lab_creative_escalation import (
    build_creative_escalation_context,
)
from pde_agent.core.hypothesis_lab_events import EVENT_STAGE_COMPLETED
from pde_agent.core.hypothesis_lab_queries import (
    recommend_next_action,
    summarize_recent_events,
)
from pde_agent.core.hypothesis_lab_session import HypothesisLabSession


STATE_HINT = "HYPOTHESIS_LAB_AUTONOMOUS_LOOP_RAN"
EVENT_AUTONOMOUS_STEP_COMPLETED = "autonomous_step_completed"


class AutonomousHypothesisLabError(RuntimeError):
    pass


class AutonomousHypothesisLab:
    """Autonomous execution engine for hypothesis-lab pipelines."""

    def __init__(
        self,
        session: HypothesisLabSession,
        memory: Any | None = None,
        context_path: str | None = None,
        target_path: str | None = None,
        reference_path: str | None = None,
        reference_sample_limit: int | None = None,
        manual_report_paths: list[str] | None = None,
    ):
        self.session = session
        self.memory = memory
        self._context_path = context_path
        self._target_path = target_path
        self._reference_path = reference_path
        self._reference_sample_limit = reference_sample_limit
        self._manual_report_paths = manual_report_paths
        self._iteration = 0
        self._results: list[dict[str, Any]] = []

    def run(
        self,
        *,
        max_iterations: int = 10,
        autonomy_level: str = "deterministic",
    ) -> dict[str, Any]:
        """Run the autonomous loop.

        Args:
            max_iterations: Upper bound on loop iterations.
            autonomy_level: "deterministic" | "guarded" | "full".

        Returns:
            Summary dict with state_hint, iterations, artifacts, metrics.
        """
        for _ in range(max_iterations):
            result = self.step(autonomy_level=autonomy_level)
            self._results.append(result)

            if result.get("action") == "report_results":
                break
            if result.get("approval_required"):
                break
            if result.get("escalation_required"):
                break

        return self._build_summary()

    def step(
        self,
        autonomy_level: str = "deterministic",
    ) -> dict[str, Any]:
        """Single iteration: observe -> decide -> execute -> learn."""
        self._iteration += 1

        # 1. Observe
        observation = self._observe()

        # 2. Decide
        decision = self._decide(observation, autonomy_level=autonomy_level)
        action = decision["action"]

        # 3. Execute
        if action == "report_results":
            execution_result = {
                "action": action,
                "status": "ok",
                "result": self._build_final_summary(),
            }
        else:
            execution_result = self._execute(
                action,
                creative_context=decision.get("creative_context"),
            )

        # 4. Learn
        step_result = {
            "iteration": self._iteration,
            "action": action,
            "observation": observation,
            "decision": decision,
            "execution": execution_result,
        }
        if decision.get("approval_required"):
            step_result["approval_required"] = True
        if decision.get("escalation_required"):
            step_result["escalation_required"] = True

        self.session.emit(
            event_type=EVENT_AUTONOMOUS_STEP_COMPLETED,
            payload={
                "iteration": self._iteration,
                "action": action,
                "status": execution_result.get("status", "unknown"),
            },
            tags=["autonomous", "step"],
        )

        return step_result

    def _observe(self) -> dict[str, Any]:
        """Gather current session state and recent events."""
        state_summary = self.session.get_state_summary()
        recent_events = summarize_recent_events(self.session, n=20)
        return {
            "state_summary": state_summary,
            "recent_events": recent_events,
        }

    def _decide(
        self,
        observation: dict[str, Any],
        autonomy_level: str,
    ) -> dict[str, Any]:
        """Determine next action based on state and autonomy level."""
        recommendation = recommend_next_action(self.session)
        action = recommendation["action"]

        # -- Cross-session memory integration --
        memory_hint = ""
        if self.memory is not None and action in (
            "run_blueprint_sandbox",
            "run_next_blueprint_sandbox",
        ):
            family = self._peek_next_blueprint_family()
            if family and family != "unknown":
                failures = self.memory.get_similar_failures(family)
                history = self.memory.get_family_performance_history(family)
                if failures or history:
                    memory_hint = (
                        f" [memory: family={family}"
                        f" failures={len(failures)}"
                        f" history={len(history)}]"
                    )

        # Doom-loop check: switch strategy if stuck
        doom_check = self._check_doom_loop()
        blocked_families: list[str] = []
        if doom_check.get("guard_active"):
            blocked_families = doom_check.get("blocked_candidate_families", [])

        # Creative escalation trigger: exhausted families need new algorithm
        if action == "report_results" and self._should_trigger_creative_escalation(
            blocked_families
        ):
            creative_reason = (
                f"{recommendation['reasoning']}{memory_hint}; "
                f"creative escalation triggered — "
                f"blocked_families={blocked_families}, "
                f"sandbox_count={self._sandbox_count()}, "
                f"score_stagnant={self._score_stagnant()}"
            )
            if autonomy_level == "deterministic":
                # Deterministic mode: just report results, no creative escalation
                return {
                    "action": action,
                    "reasoning": creative_reason + " (deterministic: reporting only)",
                    "required_inputs": recommendation.get("required_inputs", []),
                }
            if autonomy_level == "guarded":
                return {
                    "action": "generate_new_executor",
                    "reasoning": creative_reason,
                    "required_inputs": recommendation.get("required_inputs", []),
                    "approval_required": True,
                    "escalation_required": True,
                    "creative_context": build_creative_escalation_context(
                        self.session, self.memory
                    ),
                }
            # full mode
            return {
                "action": "generate_new_executor",
                "reasoning": creative_reason,
                "required_inputs": recommendation.get("required_inputs", []),
                "escalation_required": True,
                "creative_context": build_creative_escalation_context(
                    self.session, self.memory
                ),
            }

        if doom_check.get("guard_active"):
            return {
                "action": action,
                "reasoning": (
                    f"{recommendation['reasoning']}{memory_hint}; "
                    f"doom-loop detected for families: {blocked_families}. "
                    "Switching strategy."
                ),
                "required_inputs": recommendation.get("required_inputs", []),
                "doom_loop_triggered": True,
                "blocked_families": blocked_families,
            }

        # Guarded mode: pause before sandbox execution
        if autonomy_level == "guarded" and action == "run_blueprint_sandbox":
            return {
                "action": action,
                "reasoning": (
                    f"{recommendation['reasoning']}{memory_hint}; "
                    "guarded mode requires approval before sandbox execution"
                ),
                "required_inputs": recommendation.get("required_inputs", []),
                "approval_required": True,
            }

        # Full mode: escalate non-deterministic actions to LLM
        if autonomy_level == "full" and not is_deterministic_action(action):
            return {
                "action": action,
                "reasoning": (
                    f"{recommendation['reasoning']}{memory_hint}; "
                    f"action '{action}' requires LLM escalation in full mode"
                ),
                "required_inputs": recommendation.get("required_inputs", []),
                "escalation_required": True,
            }

        return {
            "action": action,
            "reasoning": f"{recommendation['reasoning']}{memory_hint}",
            "required_inputs": recommendation.get("required_inputs", []),
        }

    def _execute(
        self,
        action: str,
        creative_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Map action name to primitive function call."""
        kwargs: dict[str, Any] = {}

        if action == "build_knowledge_cards":
            kwargs["manual_report_paths"] = self._manual_report_paths or []
        elif action == "generate_new_executor":
            if creative_context is not None:
                kwargs["creative_context"] = creative_context
        elif action in ("run_blueprint_sandbox", "run_next_blueprint_sandbox"):
            # Use strategy layer to pick which blueprint to run
            from pde_agent.core.hypothesis_lab_queries import (
                recommend_family_for_sandbox,
            )

            rec = recommend_family_for_sandbox(self.session, self.memory)
            if rec.get("recommended_index") is not None:
                kwargs["blueprint_index"] = rec["recommended_index"]
            elif action == "run_next_blueprint_sandbox":
                kwargs["blueprint_index"] = self._next_pending_blueprint_index()

            if self._context_path is not None:
                kwargs["context_path"] = self._context_path
            if self._target_path is not None:
                kwargs["target_path"] = self._target_path
            if self._reference_path is not None:
                kwargs["reference_path"] = self._reference_path
            if self._reference_sample_limit is not None:
                kwargs["reference_sample_limit"] = self._reference_sample_limit

            # run_next_blueprint_sandbox maps to run_blueprint_sandbox
            action = "run_blueprint_sandbox"

        try:
            return _dispatch_action(action, self.session, **kwargs)
        except ActionDispatchError as exc:
            return {
                "action": action,
                "status": "error",
                "error": str(exc),
            }

    def _check_doom_loop(self) -> dict[str, Any]:
        """Check for repeated non-improving families via session events."""
        trace_path = self.session.workspace / "events.jsonl"
        if not trace_path.exists():
            return {"guard_active": False}

        try:
            return check_hypothesis_doom_loop(
                trace_path=trace_path,
                output_path=None,
                recent_window=20,
                repeated_family_threshold=3,
                failed_family_threshold=2,
                min_score_delta=0.1,
                force=True,
            )
        except Exception:
            return {"guard_active": False}

    def _next_pending_blueprint_index(self) -> int:
        """Return the index of the next blueprint that has not been sandboxed."""
        from pde_agent.core.hypothesis_lab_queries import list_pending_blueprints

        pending = list_pending_blueprints(self.session)
        for item in pending:
            if not item.get("sandboxed"):
                return item.get("index", 0)
        return 0

    def _peek_next_blueprint_family(self) -> str | None:
        """Return the candidate_family of the next pending blueprint, if any."""
        from pde_agent.core.hypothesis_lab_queries import list_pending_blueprints

        pending = list_pending_blueprints(self.session)
        for item in pending:
            if not item.get("sandboxed"):
                return str(item.get("candidate_family", "unknown"))
        return None

    def _should_trigger_creative_escalation(
        self,
        blocked_families: list[str],
    ) -> bool:
        """Return True if creative escalation should be triggered."""
        # Must have run at least one sandbox
        if self._sandbox_count() == 0:
            return False
        # Trigger if there are blocked families from doom-loop
        if blocked_families:
            return True
        # Trigger if score hasn't improved in last N iterations
        if self._score_stagnant():
            return True
        return False

    def _sandbox_count(self) -> int:
        """Return number of sandbox artifacts in session."""
        summary = self.session.get_state_summary()
        artifact_counts = summary.get("artifact_counts", {})
        return int(artifact_counts.get("sandbox", 0))

    def _score_stagnant(self, window: int = 3, min_delta: float = 0.01) -> bool:
        """Return True if score hasn't improved in last `window` iterations."""
        from pde_agent.core.hypothesis_lab_queries import list_candidates

        candidates = list_candidates(self.session)
        if len(candidates) < window + 1:
            return False
        # Sort by some proxy of iteration order — use the last `window + 1`
        recent = candidates[-(window + 1):]
        scores = [c.get("score") for c in recent]
        scores = [s for s in scores if isinstance(s, (int, float))]
        if len(scores) < window + 1:
            return False
        # Check if any recent score is meaningfully better than earlier ones
        best_early = max(scores[:-window]) if scores[:-window] else 0.0
        best_recent = max(scores[-window:]) if scores[-window:] else 0.0
        return (best_recent - best_early) < min_delta

    def _build_final_summary(self) -> dict[str, Any]:
        """Build a final summary of the session state."""
        summary = self.session.get_state_summary()
        best_evidence = None
        try:
            from pde_agent.core.hypothesis_lab_queries import get_best_evidence

            best_evidence = get_best_evidence(self.session)
        except Exception:
            pass

        return {
            "session_id": self.session.session_id,
            "workspace": str(self.session.workspace),
            "artifact_counts": summary.get("artifact_counts", {}),
            "total_artifacts": summary.get("total_artifacts", 0),
            "event_count": summary.get("event_count", 0),
            "best_evidence": best_evidence,
        }

    def _build_summary(self) -> dict[str, Any]:
        """Build the run summary dict."""
        summary = self.session.get_state_summary()
        artifact_counts = summary.get("artifact_counts", {})

        metrics: dict[str, Any] = {
            "iterations": self._iteration,
            "total_artifacts": summary.get("total_artifacts", 0),
            "artifact_counts": artifact_counts,
            "event_count": summary.get("event_count", 0),
        }

        artifacts: dict[str, Any] = {
            "workspace": str(self.session.workspace),
            "session_id": self.session.session_id,
        }

        # Include latest artifact paths if available
        for atype in ("knowledge_cards", "hypotheses", "blueprints", "sandbox", "outcome"):
            latest = self.session.get_latest_artifact(atype)
            if latest is not None:
                artifacts[f"latest_{atype}"] = latest.get("artifact_id")

        return {
            "run_id": self.session.session_id,
            "command": "agent-hypothesis-run-autonomous-loop",
            "status": "ok",
            "state_hint": STATE_HINT,
            "artifacts": artifacts,
            "metrics": metrics,
            "iterations": self._iteration,
            "results": self._results,
            "errors": [],
            "next_hint": "review results or resume loop",
        }
