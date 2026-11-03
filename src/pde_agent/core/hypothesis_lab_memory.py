"""Cross-session long-term memory for Hypothesis Lab experiments.

Provides a global memory layer that aggregates experiment outcomes,
failure patterns, and candidate performance across all sessions.
Uses JSONL as the backend (append-only, no SQLite dependency).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_MEMORY_PATH = Path.home() / ".pde_agent" / "hypothesis_lab_memory.jsonl"


class HypothesisLabMemoryError(RuntimeError):
    pass


class HypothesisLabMemory:
    """Global cross-session memory for hypothesis-lab experiments.

    Each record is a JSON line with a ``record_type`` field:
    - "experiment" — full experiment outcome
    - "failure" — failure pattern observation
    - "candidate" — candidate record (promoted or not)
    - "hypothesis" — hypothesis proven or disproven
    """

    def __init__(self, memory_path: str | Path | None = None) -> None:
        self.memory_path = Path(memory_path) if memory_path else DEFAULT_MEMORY_PATH
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Recording --

    def record_experiment(self, session_id: str, outcome: dict[str, Any]) -> None:
        """Append an experiment outcome to global memory."""
        record = {
            "record_type": "experiment",
            "session_id": session_id,
            "timestamp": _now_iso(),
            "outcome": dict(outcome),
        }
        self._append(record)

    def record_failure_pattern(
        self,
        session_id: str,
        family: str,
        pattern: dict[str, Any],
    ) -> None:
        """Append a failure pattern observation to global memory."""
        record = {
            "record_type": "failure",
            "session_id": session_id,
            "family": family,
            "timestamp": _now_iso(),
            "pattern": dict(pattern),
        }
        self._append(record)

    def record_candidate(self, session_id: str, candidate: dict[str, Any]) -> None:
        """Append a candidate record to global memory."""
        record = {
            "record_type": "candidate",
            "session_id": session_id,
            "timestamp": _now_iso(),
            "candidate": dict(candidate),
        }
        self._append(record)

    def record_hypothesis(self, session_id: str, hypothesis: dict[str, Any]) -> None:
        """Append a hypothesis result to global memory."""
        record = {
            "record_type": "hypothesis",
            "session_id": session_id,
            "timestamp": _now_iso(),
            "hypothesis": dict(hypothesis),
        }
        self._append(record)

    # -- Queries --

    def get_similar_failures(self, family: str) -> list[dict[str, Any]]:
        """Aggregate failure patterns for a family across all sessions."""
        failures: list[dict[str, Any]] = []
        for record in self._read_all():
            if record.get("record_type") != "failure":
                continue
            if record.get("family") != family:
                continue
            failures.append(dict(record))
        return failures

    def get_proven_hypotheses(self) -> list[dict[str, Any]]:
        """Return hypotheses that led to promoted candidates."""
        proven: list[dict[str, Any]] = []
        for record in self._read_all():
            if record.get("record_type") != "hypothesis":
                continue
            hyp = record.get("hypothesis", {})
            if hyp.get("status") == "proven" or hyp.get("promoted") is True:
                proven.append(dict(record))
        return proven

    def get_family_performance_history(self, family: str) -> list[dict[str, Any]]:
        """Return score trajectory for a candidate family over time."""
        history: list[dict[str, Any]] = []
        for record in self._read_all():
            if record.get("record_type") != "experiment":
                continue
            outcome = record.get("outcome", {})
            cand = outcome.get("candidate", outcome.get("best_candidate", {}))
            if cand.get("candidate_family") != family:
                continue
            score = cand.get("task1_total_score_proxy") or cand.get("score")
            history.append({
                "session_id": record.get("session_id"),
                "timestamp": record.get("timestamp"),
                "score": score,
                "candidate_id": cand.get("candidate_id"),
                "segment_score": cand.get("segment_score"),
            })
        history.sort(key=lambda x: x.get("timestamp") or "")
        return history

    def get_global_pareto_front(self) -> list[dict[str, Any]]:
        """Return non-dominated candidates across all sessions.

        A candidate is dominated if another has both higher score
        and lower (or equal) runtime.
        """
        points = self._collect_candidate_points()
        if not points:
            return []

        pareto: list[dict[str, Any]] = []
        for p in points:
            dominated = False
            for q in points:
                if q is p:
                    continue
                if _dominates(q, p):
                    dominated = True
                    break
            if not dominated:
                pareto.append(p)

        pareto.sort(key=lambda x: x.get("score") or 0.0, reverse=True)
        return pareto

    def get_failure_rules(self) -> list[dict[str, Any]]:
        """Return pre-computed or on-demand extracted rules."""
        from pde_agent.core.hypothesis_lab_failure_strategy import (
            extract_rules_from_memory,
        )
        return extract_rules_from_memory(self)

    def record_rule(self, rule: dict[str, Any]) -> None:
        """Store a validated rule in memory."""
        record = {
            "record_type": "rule",
            "timestamp": _now_iso(),
            "rule": dict(rule),
        }
        self._append(record)

    def get_lessons_learned(self, n: int = 10) -> dict[str, Any]:
        """Summarize cross-session insights."""
        all_records = list(self._read_all())

        experiments = [r for r in all_records if r.get("record_type") == "experiment"]
        failures = [r for r in all_records if r.get("record_type") == "failure"]
        candidates = [r for r in all_records if r.get("record_type") == "candidate"]
        hypotheses = [r for r in all_records if r.get("record_type") == "hypothesis"]

        family_scores: dict[str, list[float]] = {}
        for exp in experiments:
            outcome = exp.get("outcome", {})
            cand = outcome.get("candidate", outcome.get("best_candidate", {}))
            family = str(cand.get("candidate_family", "unknown"))
            score = cand.get("task1_total_score_proxy") or cand.get("score")
            if isinstance(score, (int, float)):
                family_scores.setdefault(family, []).append(float(score))

        family_stats = {
            family: {
                "count": len(scores),
                "best": max(scores) if scores else None,
                "mean": sum(scores) / len(scores) if scores else None,
            }
            for family, scores in family_scores.items()
        }

        failure_reasons: dict[str, int] = {}
        for failure in failures:
            pattern = failure.get("pattern", {})
            for reason, count in pattern.get("failure_reasons", {}).items():
                failure_reasons[reason] = failure_reasons.get(reason, 0) + count

        top_lessons = sorted(
            failure_reasons.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]

        return {
            "total_sessions": len({r.get("session_id") for r in all_records}),
            "total_experiments": len(experiments),
            "total_failures_recorded": len(failures),
            "total_candidates_recorded": len(candidates),
            "total_hypotheses_recorded": len(hypotheses),
            "family_performance_summary": family_stats,
            "top_failure_patterns": [
                {"reason": reason, "count": count}
                for reason, count in top_lessons
            ],
            "recommendation": _lessons_recommendation(family_stats, failure_reasons),
        }

    # -- Internal --

    def _append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True) + "\n"
        with self.memory_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.memory_path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            with self.memory_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        return records

    def _collect_candidate_points(self) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for record in self._read_all():
            cand: dict[str, Any] | None = None
            if record.get("record_type") == "candidate":
                cand = record.get("candidate", {})
            elif record.get("record_type") == "experiment":
                outcome = record.get("outcome", {})
                cand = outcome.get("candidate", outcome.get("best_candidate", {}))

            if not isinstance(cand, dict):
                continue

            cid = str(cand.get("candidate_id", ""))
            if cid and cid in seen_ids:
                continue
            if cid:
                seen_ids.add(cid)

            score = cand.get("task1_total_score_proxy") or cand.get("score")
            if score is None:
                continue

            runtime = cand.get("inference_seconds") or cand.get("runtime")
            points.append({
                "candidate_id": cid or "unknown",
                "candidate_family": cand.get("candidate_family", "unknown"),
                "score": float(score),
                "runtime": float(runtime) if isinstance(runtime, (int, float)) else None,
                "session_id": record.get("session_id"),
                "timestamp": record.get("timestamp"),
            })

        return points


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _dominates(q: dict[str, Any], p: dict[str, Any]) -> bool:
    """Return True if q dominates p (higher score, lower runtime)."""
    q_score = q.get("score")
    p_score = p.get("score")
    q_runtime = q.get("runtime")
    p_runtime = p.get("runtime")

    if q_score is None or p_score is None:
        return False

    better_score = q_score > p_score
    better_runtime = (
        q_runtime is not None
        and p_runtime is not None
        and q_runtime <= p_runtime
    )
    equal_runtime = q_runtime is None or p_runtime is None

    return better_score and (better_runtime or equal_runtime)


def _lessons_recommendation(
    family_stats: dict[str, Any],
    failure_reasons: dict[str, int],
) -> str:
    if not family_stats:
        return "no_data"
    best_family = max(
        family_stats.items(),
        key=lambda x: x[1].get("best") or 0.0,
    )
    if best_family[1].get("best", 0) > 100:
        return f"focus_on_{best_family[0]}"
    if failure_reasons:
        top = max(failure_reasons, key=lambda r: failure_reasons[r])
        return f"address_{top}"
    return "continue_exploration"


def get_memory_path_from_env() -> Path | None:
    """Resolve global memory path from environment or session config."""
    env_path = os.environ.get("PDE_AGENT_HYPOTHESIS_MEMORY_PATH")
    if env_path:
        return Path(env_path)
    return None
