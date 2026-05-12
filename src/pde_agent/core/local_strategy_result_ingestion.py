"""Normalize heterogeneous experiment artifacts into local strategy results."""


import json
from pathlib import Path
from typing import Any


RESULT_STATE_HINT = "LOCAL_EXPERIMENT_RESULT"


class LocalStrategyResultIngestionError(RuntimeError):
    pass


def ingest_local_strategy_result(
    *,
    source_path: str | Path,
    output_path: str | Path,
    baseline_score: float | None,
    result_status: str,
    failure_reason: str | None,
) -> dict[str, Any]:
    source_file = Path(source_path)
    source = _read_json_object(source_file)
    normalized = _normalize_source(
        source=source,
        baseline_score_override=baseline_score,
        result_status=result_status,
        failure_reason=failure_reason,
    )
    result = {
        "schema_version": 1,
        "state_hint": RESULT_STATE_HINT,
        "source_path": str(source_file),
        "source_state_hint": source.get("state_hint"),
        **normalized,
        "source": _source_excerpt(source),
        "result_path": str(output_path),
    }
    _write_json(Path(output_path), result)
    return result


def _normalize_source(
    *,
    source: dict[str, Any],
    baseline_score_override: float | None,
    result_status: str,
    failure_reason: str | None,
) -> dict[str, Any]:
    score = _extract_score(source)
    baseline_score = (
        baseline_score_override
        if baseline_score_override is not None
        else _extract_baseline_score(source)
    )
    score_delta = _extract_score_delta(source)
    if score_delta is None and score is not None and baseline_score is not None:
        score_delta = float(score - baseline_score)
    inference_time = _extract_inference_time(source)
    return {
        "result_status": result_status,
        "score": score,
        "baseline_score": baseline_score,
        "score_delta": score_delta,
        "inference_time_seconds": inference_time,
        "failure_reason": failure_reason or _string_or_none(source.get("error")),
        "result_summary": _result_summary(source=source, result_status=result_status),
    }


def _extract_score(source: dict[str, Any]) -> float | None:
    for key in ("score", "segment_score", "task_prediction_score"):
        value = _number_or_none(source.get(key))
        if value is not None:
            return value
    score = source.get("score")
    if isinstance(score, dict):
        for key in ("competition_like_score_short", "segment_score", "score"):
            value = _number_or_none(score.get(key))
            if value is not None:
                return value
    metrics = source.get("metrics")
    if isinstance(metrics, dict):
        for key in ("score", "segment_score", "competition_like_score_short"):
            value = _number_or_none(metrics.get(key))
            if value is not None:
                return value
    return None


def _extract_baseline_score(source: dict[str, Any]) -> float | None:
    value = _number_or_none(source.get("baseline_score"))
    if value is not None:
        return value
    comparison = source.get("baseline_comparison")
    if isinstance(comparison, dict):
        return _number_or_none(comparison.get("baseline_score"))
    return None


def _extract_score_delta(source: dict[str, Any]) -> float | None:
    value = _number_or_none(source.get("score_delta"))
    if value is not None:
        return value
    comparison = source.get("baseline_comparison")
    if isinstance(comparison, dict):
        return _number_or_none(comparison.get("score_delta"))
    metrics = source.get("metrics")
    if isinstance(metrics, dict):
        return _number_or_none(metrics.get("score_delta"))
    return None


def _extract_inference_time(source: dict[str, Any]) -> float | None:
    for key in ("inference_time_seconds", "inference_time", "elapsed_seconds"):
        value = _number_or_none(source.get(key))
        if value is not None:
            return value
    metrics = source.get("metrics")
    if isinstance(metrics, dict):
        for key in ("inference_time_seconds", "inference_time", "elapsed_seconds"):
            value = _number_or_none(metrics.get(key))
            if value is not None:
                return value
    return None


def _source_excerpt(source: dict[str, Any]) -> dict[str, Any]:
    excerpt: dict[str, Any] = {"state_hint": source.get("state_hint")}
    for key in ("task_id", "model_family", "candidate", "metrics", "segments"):
        if key in source:
            excerpt[key] = source[key]
    return excerpt


def _result_summary(*, source: dict[str, Any], result_status: str) -> str:
    state_hint = source.get("state_hint", "unknown")
    if result_status == "completed":
        return f"ingested completed result from {state_hint}"
    return f"ingested {result_status} result from {state_hint}"


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LocalStrategyResultIngestionError(f"missing source artifact: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LocalStrategyResultIngestionError(f"invalid source JSON: {path}") from exc
    if not isinstance(data, dict):
        raise LocalStrategyResultIngestionError("source JSON must be an object")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
