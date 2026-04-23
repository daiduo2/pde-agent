import json
from pathlib import Path


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_config_files(base_path: Path, override_path: Path, task_path: Path) -> dict[str, object]:
    resolved: dict[str, object] = {}
    resolved.update(_load_json(base_path))
    resolved.update(_load_json(task_path))
    if override_path.exists():
        resolved.update(_load_json(override_path))
    return resolved
