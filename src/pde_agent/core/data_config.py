from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pde_agent.core.array_contract import SourceMode


@dataclass(frozen=True)
class DataConfig:
    mode: SourceMode
    problem_id: str
    source_path: Path | None
    dataset_key: str | None
    context_steps: int
    forecast_steps: int
    time_reduction: int
    space_reduction: int
    sample_limit: int | None


def load_data_config(resolved_config_path: Path) -> DataConfig:
    raw_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    raw_data = raw_config.get("data") if isinstance(raw_config, dict) else None
    data = raw_data if isinstance(raw_data, dict) else {}

    source_path = data.get("source_path")
    return DataConfig(
        mode=str(data.get("mode", "synthetic")),  # type: ignore[arg-type]
        problem_id=str(data.get("problem_id", "PDE_Burgers")),
        source_path=None if source_path is None else Path(str(source_path)),
        dataset_key=_optional_str(data.get("dataset_key")),
        context_steps=int(data.get("context_steps", 10)),
        forecast_steps=int(data.get("forecast_steps", 190)),
        time_reduction=int(data.get("time_reduction", 5)),
        space_reduction=int(data.get("space_reduction", 4)),
        sample_limit=_optional_int(data.get("sample_limit")),
    )


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)
