"""Neural operator component registry for PDE agent experiments.

Provides a pluggable component registry for models, losses, datasets, and
optimizers.  Components are registered via decorators and discovered at runtime.

Design principles:
- Registry-driven discovery (no hardcoded lists)
- Configuration-driven instantiation
- Full provenance tracking via ResearchMemory
"""

from __future__ import annotations

from pde_agent.core.neural_operator.config import ExperimentConfig
from pde_agent.core.neural_operator.hypothesis_lab_integration import (
    append_strategy_ledger,
    create_evidence_summary,
    create_experiment_rubric,
    create_promotion_report,
    create_strategy_proposal,
    run_full_hypothesis_lab_pipeline,
)
from pde_agent.core.neural_operator.registry import (
    list_datasets,
    list_losses,
    list_models,
    list_optimizers,
    register_dataset,
    register_loss,
    register_model,
    register_optimizer,
)

# Trigger builtin registrations (models, losses from existing core modules)
try:
    import pde_agent.core.task1_short_window_models  # noqa: F401
except Exception:
    pass  # torch may not be available

__all__ = [
    "ExperimentConfig",
    "register_model",
    "register_loss",
    "register_dataset",
    "register_optimizer",
    "list_models",
    "list_losses",
    "list_datasets",
    "list_optimizers",
    "create_strategy_proposal",
    "create_experiment_rubric",
    "create_evidence_summary",
    "create_promotion_report",
    "append_strategy_ledger",
    "run_full_hypothesis_lab_pipeline",
]
