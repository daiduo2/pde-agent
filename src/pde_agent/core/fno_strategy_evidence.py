from __future__ import annotations

from pathlib import Path
from typing import Any

from pde_agent.core.operator_strategy_evidence import (
    STATE_HINT,
    OperatorStrategyEvidenceError,
    convert_operator_evidence_summary,
)


SOURCE_STATE_HINT = "FNO_EVIDENCE_SWEEP_COMPLETED"


class FNOStrategyEvidenceError(OperatorStrategyEvidenceError):
    """Raised when an FNO sweep summary cannot become strategy-growth evidence."""


def convert_fno_evidence_summary(
    *,
    sweep_summary_path: Path,
    output_path: Path,
    long_window_threshold: int,
    force: bool = False,
) -> dict[str, Any]:
    try:
        return convert_operator_evidence_summary(
            sweep_summary_path=sweep_summary_path,
            output_path=output_path,
            model_family="FNO",
            source_state_hint=SOURCE_STATE_HINT,
            long_window_threshold=long_window_threshold,
            force=force,
        )
    except OperatorStrategyEvidenceError as exc:
        raise FNOStrategyEvidenceError(str(exc)) from exc
