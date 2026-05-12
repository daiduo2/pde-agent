from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(
    path: Path,
    payload: object,
    *,
    indent: int = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii),
        encoding="utf-8",
    )
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_dict(
    path: Path,
    *,
    error_cls: type[Exception] = ValueError,
    error_message: str = "JSON artifact must be an object",
) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise error_cls(error_message)
    return payload
