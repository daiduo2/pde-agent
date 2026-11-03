"""Agent-facing tool registrations for the ToolRegistry.

Each tool is a thin wrapper over existing core functionality.  Tools are
registered with the global ``registry`` so the agent can discover and invoke
them by name.

Usage::

    from pde_agent.core.tool_registry import registry
    schemas = registry.get_schemas()
    result = registry.call("train-short-window-model", {...})
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pde_agent.core.research_memory import ResearchMemory
from pde_agent.core.tool_registry import registry


# ---------------------------------------------------------------------------
# Discovery tools
# ---------------------------------------------------------------------------

@registry.register(
    name="list-models",
    description="List all registered neural-operator model families.",
    parameters={},
)
def _list_models() -> dict[str, Any]:
    from pde_agent.core.neural_operator import list_models
    return {"models": list_models()}


@registry.register(
    name="list-losses",
    description="List all registered loss functions.",
    parameters={},
)
def _list_losses() -> dict[str, Any]:
    from pde_agent.core.neural_operator import list_losses
    return {"losses": list_losses()}


# ---------------------------------------------------------------------------
# Data-preprocessing tools
# ---------------------------------------------------------------------------

@registry.register(
    name="normalize-windows",
    description=(
        "Normalize context/target windows. "
        "Returns normalized arrays and statistics dict for later denormalization."
    ),
    parameters={
        "context": {"type": "array", "description": "Context array [N, T, X]"},
        "target": {"type": "array", "description": "Target array [N, T, X]"},
        "normalization": {
            "type": "str",
            "description": "Normalization type: none or sample_standard",
        },
    },
)
def _normalize_windows(
    context: Any, target: Any, normalization: str = "sample_standard"
) -> dict[str, Any]:
    import numpy as np
    from pde_agent.core.task1_short_window_models import normalize_windows
    ctx, tgt, stats = normalize_windows(
        context=np.asarray(context, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
        normalization=normalization,
    )
    return {
        "context": ctx,
        "target": tgt,
        "stats": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in stats.items()},
    }


@registry.register(
    name="denormalize-prediction",
    description="Denormalize a prediction using statistics from normalize-windows.",
    parameters={
        "prediction": {"type": "array", "description": "Normalized prediction [N, T, X]"},
        "stats": {"type": "object", "description": "Statistics dict with mean/std"},
        "normalization": {
            "type": "str",
            "description": "Normalization type used during normalization",
        },
    },
)
def _denormalize_prediction(
    prediction: Any, stats: dict[str, Any], normalization: str = "sample_standard"
) -> dict[str, Any]:
    import numpy as np
    from pde_agent.core.task1_short_window_models import denormalize_prediction
    stats_arrays = {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}
    denorm = denormalize_prediction(
        prediction=np.asarray(prediction, dtype=np.float32),
        stats=stats_arrays,
        normalization=normalization,
    )
    return {"prediction": denorm.tolist() if hasattr(denorm, "tolist") else denorm}


# ---------------------------------------------------------------------------
# Training tools
# ---------------------------------------------------------------------------

@registry.register(
    name="train-short-window-model",
    description=(
        "Train a short-window neural-operator model. "
        "Produces checkpoint, predictions, and a JSON report."
    ),
    parameters={
        "source_path": {"type": "str", "description": "Path to source NPZ or HDF5 data"},
        "output_dir": {"type": "str", "description": "Directory for artifacts"},
        "dataset_key": {"type": "str", "description": "Dataset key in source file"},
        "model_family": {"type": "str", "description": "Model family name"},
        "train_sample_limit": {"type": "int"},
        "val_sample_limit": {"type": "int"},
        "context_steps": {"type": "int"},
        "forecast_steps": {"type": "int"},
        "train_steps": {"type": "int"},
        "batch_size": {"type": "int"},
        "learning_rate": {"type": "float"},
        "loss_type": {"type": "str"},
        "normalization": {"type": "str"},
        "fno_modes": {"type": "int"},
        "fno_width": {"type": "int"},
        "fno_depth": {"type": "int"},
        "time_reduction": {"type": "int"},
        "space_reduction": {"type": "int"},
        "window_stride": {"type": "int"},
        "seed": {"type": "int"},
        "force": {"type": "bool"},
    },
)
def _train_short_window_model(
    source_path: str,
    output_dir: str,
    dataset_key: str = "tensor",
    model_family: str = "conv1d_residual",
    train_sample_limit: int = 64,
    val_sample_limit: int = 32,
    context_steps: int = 10,
    forecast_steps: int = 47,
    train_steps: int = 8,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    loss_type: str = "mse",
    normalization: str = "none",
    fno_modes: int = 8,
    fno_width: int = 32,
    fno_depth: int = 1,
    time_reduction: int = 1,
    space_reduction: int = 4,
    window_stride: int = 1,
    seed: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    from pde_agent.core.task1_short_window_training_loops import (
        run_task1_short_window_training_loop,
    )
    report = run_task1_short_window_training_loop(
        source_path=source_path,
        output_dir=output_dir,
        dataset_key=dataset_key,
        model_family=model_family,
        train_sample_limit=train_sample_limit,
        val_sample_limit=val_sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        train_steps=train_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss_type=loss_type,
        normalization=normalization,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        window_stride=window_stride,
        fno_modes=fno_modes,
        fno_width=fno_width,
        fno_depth=fno_depth,
        seed=seed,
        force=force,
    )
    return {"report": report}


@registry.register(
    name="select-model-family",
    description=(
        "Run a short-window training loop across multiple model families "
        "and select the best one."
    ),
    parameters={
        "source_path": {"type": "str"},
        "output_dir": {"type": "str"},
        "model_families": {"type": "array", "description": "List of family names"},
        "dataset_key": {"type": "str"},
        "train_sample_limit": {"type": "int"},
        "val_sample_limit": {"type": "int"},
        "context_steps": {"type": "int"},
        "forecast_steps": {"type": "int"},
        "train_steps": {"type": "int"},
        "batch_size": {"type": "int"},
        "learning_rate": {"type": "float"},
        "time_reduction": {"type": "int"},
        "space_reduction": {"type": "int"},
        "seed": {"type": "int"},
        "force": {"type": "bool"},
    },
)
def _select_model_family(
    source_path: str,
    output_dir: str,
    model_families: list[str] | None = None,
    dataset_key: str = "tensor",
    train_sample_limit: int = 64,
    val_sample_limit: int = 32,
    context_steps: int = 10,
    forecast_steps: int = 47,
    train_steps: int = 8,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    time_reduction: int = 1,
    space_reduction: int = 4,
    seed: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    from pde_agent.core.task1_short_window_training_loops import (
        run_task1_short_window_family_selection_loop,
    )
    report = run_task1_short_window_family_selection_loop(
        source_path=source_path,
        output_dir=output_dir,
        model_families=model_families,
        dataset_key=dataset_key,
        train_sample_limit=train_sample_limit,
        val_sample_limit=val_sample_limit,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        train_steps=train_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        time_reduction=time_reduction,
        space_reduction=space_reduction,
        seed=seed,
        force=force,
    )
    return {"report": report}


# ---------------------------------------------------------------------------
# Evaluation / scoring tools
# ---------------------------------------------------------------------------

@registry.register(
    name="compute-segment-scores",
    description=(
        "Compute early/middle/late segment Rel-MSE scores for a prediction "
        "against a target. Returns segment scores and overall score."
    ),
    parameters={
        "prediction_path": {"type": "str", "description": "Path to prediction NPZ"},
        "target_path": {"type": "str", "description": "Path to target NPZ"},
        "prediction_key": {"type": "str"},
        "target_key": {"type": "str"},
    },
)
def _compute_segment_scores(
    prediction_path: str,
    target_path: str,
    prediction_key: str = "prediction",
    target_key: str = "target",
) -> dict[str, Any]:
    pred_file = np.load(prediction_path)
    tgt_file = np.load(target_path)
    prediction = pred_file[prediction_key]
    target = tgt_file[target_key]
    squared_error = (prediction - target) ** 2
    target_squared = target**2
    from pde_agent.core.competition_scoring import _rel_mse_segment, _late_segment
    EARLY_FUTURE_STOP = 47
    MIDDLE_FUTURE_STOP = 95
    FUTURE_STEPS = 190 - 10
    early = _rel_mse_segment(
        squared_error=squared_error,
        target_squared=target_squared,
        future_start=0,
        future_stop=EARLY_FUTURE_STOP,
    )
    middle = _rel_mse_segment(
        squared_error=squared_error,
        target_squared=target_squared,
        future_start=EARLY_FUTURE_STOP,
        future_stop=MIDDLE_FUTURE_STOP,
    )
    late = _late_segment(
        squared_error=squared_error,
        future_start=MIDDLE_FUTURE_STOP,
        future_stop=FUTURE_STEPS,
    )
    segment_score = float(
        0.25 * early["score"] + 0.25 * middle["score"] + 0.5 * late["score"]
    )
    return {
        "segment_score": segment_score,
        "early": early,
        "middle": middle,
        "late": late,
    }


# ---------------------------------------------------------------------------
# Hypothesis-lab artifact tools
# ---------------------------------------------------------------------------

@registry.register(
    name="create-strategy-proposal",
    description="Create a strategy_proposal.json artifact for the hypothesis-lab pipeline.",
    parameters={
        "output_dir": {"type": "str"},
        "hypothesis": {"type": "str"},
        "mechanism": {"type": "str"},
        "plan": {"type": "array"},
        "observables": {"type": "array"},
        "risks": {"type": "array"},
    },
)
def _create_strategy_proposal(
    output_dir: str,
    hypothesis: str,
    mechanism: str,
    plan: list[str],
    observables: list[str],
    risks: list[str],
) -> dict[str, Any]:
    from pde_agent.core.neural_operator.hypothesis_lab_integration import (
        create_strategy_proposal,
    )
    path = create_strategy_proposal(
        output_dir=Path(output_dir),
        hypothesis=hypothesis,
        mechanism=mechanism,
        plan=plan,
        observables=observables,
        risks=risks,
    )
    return {"path": str(path)}


@registry.register(
    name="build-experiment-rubric",
    description="Create an experiment_rubric.json artifact.",
    parameters={
        "output_dir": {"type": "str"},
        "metrics": {"type": "object", "description": "Dict of metric thresholds"},
    },
)
def _build_experiment_rubric(
    output_dir: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    from pde_agent.core.neural_operator.hypothesis_lab_integration import (
        create_experiment_rubric,
    )
    path = create_experiment_rubric(
        output_dir=Path(output_dir),
        metrics=metrics,
    )
    return {"path": str(path)}


@registry.register(
    name="create-evidence-summary",
    description="Create an evidence_summary.json from training evidence.",
    parameters={
        "output_dir": {"type": "str"},
        "evidence_path": {"type": "str"},
    },
)
def _create_evidence_summary(
    output_dir: str,
    evidence_path: str,
) -> dict[str, Any]:
    from pde_agent.core.neural_operator.hypothesis_lab_integration import (
        create_evidence_summary,
    )
    path = create_evidence_summary(
        output_dir=Path(output_dir),
        evidence_path=Path(evidence_path),
    )
    return {"path": str(path)}


@registry.register(
    name="evaluate-promotion",
    description="Create a promotion_report.json evaluating evidence against a rubric.",
    parameters={
        "output_dir": {"type": "str"},
        "evidence_path": {"type": "str"},
        "rubric_path": {"type": "str"},
    },
)
def _evaluate_promotion(
    output_dir: str,
    evidence_path: str,
    rubric_path: str,
) -> dict[str, Any]:
    from pde_agent.core.neural_operator.hypothesis_lab_integration import (
        create_promotion_report,
    )
    path = create_promotion_report(
        output_dir=Path(output_dir),
        evidence_path=Path(evidence_path),
        rubric_path=Path(rubric_path),
    )
    return {"path": str(path)}


@registry.register(
    name="append-strategy-ledger",
    description="Append a strategy attempt to the strategy_ledger.jsonl.",
    parameters={
        "output_dir": {"type": "str"},
        "proposal_path": {"type": "str"},
        "rubric_path": {"type": "str"},
        "evidence_path": {"type": "str"},
        "promotion_path": {"type": "str"},
        "outcome": {"type": "str"},
    },
)
def _append_strategy_ledger(
    output_dir: str,
    proposal_path: str,
    rubric_path: str,
    evidence_path: str,
    promotion_path: str,
    outcome: str,
) -> dict[str, Any]:
    from pde_agent.core.neural_operator.hypothesis_lab_integration import (
        append_strategy_ledger,
    )
    path = append_strategy_ledger(
        output_dir=Path(output_dir),
        proposal_path=Path(proposal_path),
        rubric_path=Path(rubric_path),
        evidence_path=Path(evidence_path),
        promotion_path=Path(promotion_path),
        outcome=outcome,
    )
    return {"path": str(path)}


# ---------------------------------------------------------------------------
# ResearchMemory tools
# ---------------------------------------------------------------------------

@registry.register(
    name="record-experiment",
    description="Record an experiment result to ResearchMemory JSON.",
    parameters={
        "memory_path": {"type": "str"},
        "experiment_id": {"type": "str"},
        "config": {"type": "object"},
        "metrics": {"type": "object"},
        "decision": {"type": "str"},
        "notes": {"type": "str"},
    },
)
def _record_experiment(
    memory_path: str,
    experiment_id: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    decision: str = "",
    notes: str = "",
) -> dict[str, Any]:
    mem = ResearchMemory.load(memory_path)
    mem.record_experiment(
        experiment_id=experiment_id,
        config=config,
        metrics=metrics,
        decision=decision,
        notes=notes,
    )
    mem.save(memory_path)
    return {"status": "ok", "best_experiment_id": mem.best_experiment_id}


@registry.register(
    name="query-research-memory",
    description="Query ResearchMemory for experiments, failures, or summary.",
    parameters={
        "memory_path": {"type": "str"},
        "query": {
            "type": "str",
            "description": "One of: summary, best, recent, failures, trend",
        },
        "n": {"type": "int", "description": "Number of records for recent/failures/trend"},
        "model_family": {"type": "str", "description": "Filter by model family"},
    },
)
def _query_research_memory(
    memory_path: str,
    query: str = "summary",
    n: int = 5,
    model_family: str = "",
) -> dict[str, Any]:
    mem = ResearchMemory.load(memory_path)
    if query == "summary":
        return mem.summary()
    if query == "best":
        best = mem.best_experiment
        return {"best": best.config if best else None, "score": best.metrics.get("score") if best else None}
    if query == "recent":
        return {"experiments": [e.config for e in mem.get_recent_experiments(n)]}
    if query == "failures":
        return {"failures": [e.config for e in mem.get_recent_failures(n)]}
    if query == "trend":
        return {"trend": mem.get_score_trend(window=n)}
    if query == "by_family" and model_family:
        return {"experiments": [e.config for e in mem.get_experiments_by_family(model_family)]}
    return {"error": f"unknown query: {query}"}
