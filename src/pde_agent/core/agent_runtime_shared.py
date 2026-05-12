from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONTRACT_STATE_HINT = "AGENT_RUNTIME_CONTRACT_BUILT"
CLAUDE_RUNTIME_STATE_HINT = "CLAUDE_AGENT_SDK_RUNTIME_RAN"
CLAUDE_CLI_RUNTIME_STATE_HINT = "CLAUDE_CLI_RUNTIME_RAN"
CLAUDE_FNO_TRAINING_SMOKE_TASK_STATE_HINT = "CLAUDE_FNO_TRAINING_SMOKE_TASK_BUILT"
CLAUDE_PROMOTION_SMOKE_TASK_STATE_HINT = "CLAUDE_PROMOTION_SMOKE_TASK_BUILT"
CLAUDE_HYPOTHESIS_LAB_CLOSURE_TASK_STATE_HINT = (
    "CLAUDE_HYPOTHESIS_LAB_CLOSURE_TASK_BUILT"
)
CLAUDE_LITERATURE_CANDIDATE_CLOSURE_TASK_STATE_HINT = (
    "CLAUDE_LITERATURE_CANDIDATE_CLOSURE_TASK_BUILT"
)
AGENT_RUNTIME_ISSUE_RECORDED_STATE_HINT = "AGENT_RUNTIME_ISSUE_RECORDED"
COMPETITION_LLM_LOG_STATE_HINT = "COMPETITION_LLM_LOG_EXPORTED"


class AgentRuntimeError(RuntimeError):
    pass


def _write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise AgentRuntimeError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

