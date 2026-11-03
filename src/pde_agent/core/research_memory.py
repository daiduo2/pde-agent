"""Structured research memory for agent-driven experimentation.

Persists experiment history, hypotheses, failure patterns, and best-known
configurations as JSON.  The agent reads the memory to inform decisions and
writes updates after each experiment.

Usage::

    from pde_agent.core.research_memory import ResearchMemory

    mem = ResearchMemory.load("runs/research_memory.json")
    mem.record_experiment(
        experiment_id="exp_001",
        config={"model_family": "fno_lite", "modes": 16},
        metrics={"early_score": 71.32, "rel_mse": 0.017},
        decision="continue",
    )
    mem.save()

    # Agent queries memory to decide next action
    best = mem.best_experiment
    recent_failures = mem.get_recent_failures(n=5)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentRecord:
    """A single experiment attempt."""

    experiment_id: str
    timestamp: float
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    decision: str = ""  # "continue", "pivot", "abandon", "promote"
    artifacts: Dict[str, str] = field(default_factory=dict)
    notes: str = ""


@dataclass
class HypothesisRecord:
    """A hypothesis and its validation status."""

    hypothesis_id: str
    timestamp: float
    statement: str
    mechanism: str
    expected_observables: List[str]
    status: str = "open"  # "open", "validated", "rejected"
    supporting_experiments: List[str] = field(default_factory=list)
    rejecting_experiments: List[str] = field(default_factory=list)


@dataclass
class FailurePattern:
    """A recognized failure pattern."""

    pattern_id: str
    timestamp: float
    description: str
    trigger_config: Dict[str, Any]
    symptoms: List[str]
    resolution: str = ""
    related_experiments: List[str] = field(default_factory=list)


@dataclass
class StrategyRecord:
    """A promoted strategy."""

    strategy_id: str
    timestamp: float
    title: str
    hypothesis_id: str
    best_config: Dict[str, Any]
    best_score: float
    evidence_paths: List[str] = field(default_factory=list)


class ResearchMemory:
    """Structured persistent memory for research experiments."""

    def __init__(
        self,
        task: str = "task1",
        experiments: Optional[List[ExperimentRecord]] = None,
        hypotheses: Optional[List[HypothesisRecord]] = None,
        failure_patterns: Optional[List[FailurePattern]] = None,
        strategies: Optional[List[StrategyRecord]] = None,
        best_experiment_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.task = task
        self.experiments: List[ExperimentRecord] = experiments or []
        self.hypotheses: List[HypothesisRecord] = hypotheses or []
        self.failure_patterns: List[FailurePattern] = failure_patterns or []
        self.strategies: List[StrategyRecord] = strategies or []
        self.best_experiment_id = best_experiment_id
        self.metadata: Dict[str, Any] = metadata or {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "ResearchMemory":
        """Load from JSON file, or return empty memory if missing or empty."""
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return cls()
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls._from_dict(raw)

    def save(self, path: str | Path) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self._to_dict(), f, indent=2, sort_keys=True, default=str)
            f.write("\n")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_experiment(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        decision: str = "",
        artifacts: Optional[Dict[str, str]] = None,
        notes: str = "",
    ) -> None:
        """Append a new experiment record."""
        record = ExperimentRecord(
            experiment_id=experiment_id,
            timestamp=time.time(),
            config=config,
            metrics=metrics,
            decision=decision,
            artifacts=artifacts or {},
            notes=notes,
        )
        self.experiments.append(record)
        self._maybe_update_best(record)

    def record_hypothesis(
        self,
        hypothesis_id: str,
        statement: str,
        mechanism: str,
        expected_observables: List[str],
    ) -> None:
        """Record a new hypothesis."""
        self.hypotheses.append(
            HypothesisRecord(
                hypothesis_id=hypothesis_id,
                timestamp=time.time(),
                statement=statement,
                mechanism=mechanism,
                expected_observables=expected_observables,
            )
        )

    def record_failure_pattern(
        self,
        pattern_id: str,
        description: str,
        trigger_config: Dict[str, Any],
        symptoms: List[str],
        resolution: str = "",
    ) -> None:
        """Record a recognized failure pattern."""
        self.failure_patterns.append(
            FailurePattern(
                pattern_id=pattern_id,
                timestamp=time.time(),
                description=description,
                trigger_config=trigger_config,
                symptoms=symptoms,
                resolution=resolution,
            )
        )

    def record_strategy(
        self,
        strategy_id: str,
        title: str,
        hypothesis_id: str,
        best_config: Dict[str, Any],
        best_score: float,
        evidence_paths: Optional[List[str]] = None,
    ) -> None:
        """Record a promoted strategy."""
        self.strategies.append(
            StrategyRecord(
                strategy_id=strategy_id,
                timestamp=time.time(),
                title=title,
                hypothesis_id=hypothesis_id,
                best_config=best_config,
                best_score=best_score,
                evidence_paths=evidence_paths or [],
            )
        )

    # ------------------------------------------------------------------
    # Queries (for agent decision-making)
    # ------------------------------------------------------------------

    @property
    def best_experiment(self) -> Optional[ExperimentRecord]:
        """Return the best experiment record."""
        if not self.experiments:
            return None
        if self.best_experiment_id:
            for e in self.experiments:
                if e.experiment_id == self.best_experiment_id:
                    return e
        # Fallback: highest score
        scored = [e for e in self.experiments if "score" in e.metrics]
        if scored:
            return max(scored, key=lambda e: e.metrics.get("score", 0))
        return self.experiments[-1]

    def get_recent_experiments(self, n: int = 5) -> List[ExperimentRecord]:
        """Return the N most recent experiments."""
        return self.experiments[-n:]

    def get_recent_failures(self, n: int = 5) -> List[ExperimentRecord]:
        """Return the N most recent experiments with decision != continue/promote."""
        failures = [
            e for e in self.experiments
            if e.decision not in ("continue", "promote", "")
        ]
        return failures[-n:]

    def get_experiments_by_family(self, model_family: str) -> List[ExperimentRecord]:
        """Return all experiments for a given model family."""
        return [
            e for e in self.experiments
            if e.config.get("model_family") == model_family
        ]

    def get_score_trend(self, window: int = 5) -> List[float]:
        """Return score trend over the last N experiments."""
        recent = self.experiments[-window:]
        return [e.metrics.get("score", 0.0) for e in recent]

    def get_open_hypotheses(self) -> List[HypothesisRecord]:
        """Return hypotheses that have not been validated or rejected."""
        return [h for h in self.hypotheses if h.status == "open"]

    def get_best_strategy(self) -> Optional[StrategyRecord]:
        """Return the strategy with the highest score."""
        if not self.strategies:
            return None
        return max(self.strategies, key=lambda s: s.best_score)

    def summary(self) -> Dict[str, Any]:
        """Return a compact summary for agent consumption."""
        best = self.best_experiment
        best_strategy = self.get_best_strategy()
        return {
            "task": self.task,
            "total_experiments": len(self.experiments),
            "total_hypotheses": len(self.hypotheses),
            "open_hypotheses": len(self.get_open_hypotheses()),
            "total_strategies": len(self.strategies),
            "failure_patterns": len(self.failure_patterns),
            "best_experiment_id": self.best_experiment_id,
            "best_score": best.metrics.get("score") if best else None,
            "best_config": best.config if best else None,
            "best_strategy_id": best_strategy.strategy_id if best_strategy else None,
            "recent_score_trend": self.get_score_trend(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_update_best(self, record: ExperimentRecord) -> None:
        """Update best_experiment_id if this record is better."""
        score = record.metrics.get("score")
        if score is None:
            return
        current_best = self.best_experiment
        current_score = current_best.metrics.get("score", 0.0) if current_best else 0.0
        if score > current_score:
            self.best_experiment_id = record.experiment_id

    def _to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "experiments": [asdict(e) for e in self.experiments],
            "hypotheses": [asdict(h) for h in self.hypotheses],
            "failure_patterns": [asdict(p) for p in self.failure_patterns],
            "strategies": [asdict(s) for s in self.strategies],
            "best_experiment_id": self.best_experiment_id,
            "metadata": self.metadata,
        }

    @classmethod
    def _from_dict(cls, raw: Dict[str, Any]) -> "ResearchMemory":
        return cls(
            task=raw.get("task", "task1"),
            experiments=[ExperimentRecord(**e) for e in raw.get("experiments", [])],
            hypotheses=[HypothesisRecord(**h) for h in raw.get("hypotheses", [])],
            failure_patterns=[FailurePattern(**p) for p in raw.get("failure_patterns", [])],
            strategies=[StrategyRecord(**s) for s in raw.get("strategies", [])],
            best_experiment_id=raw.get("best_experiment_id", ""),
            metadata=raw.get("metadata", {}),
        )
