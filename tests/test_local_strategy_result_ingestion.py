import json
from pathlib import Path


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_ingest_local_strategy_result_from_competition_score(tmp_path: Path) -> None:
    from pde_agent.core.local_strategy_result_ingestion import (
        ingest_local_strategy_result,
    )

    source_path = _write(
        tmp_path / "score.json",
        {
            "state_hint": "COMPETITION_PREDICTION_SCORED",
            "segment_score": 82.5,
            "baseline_score": 80.0,
            "inference_time": 59.7,
        },
    )

    result = ingest_local_strategy_result(
        source_path=source_path,
        output_path=tmp_path / "result.json",
        baseline_score=None,
        result_status="completed",
        failure_reason=None,
    )

    assert result["state_hint"] == "LOCAL_EXPERIMENT_RESULT"
    assert result["source_state_hint"] == "COMPETITION_PREDICTION_SCORED"
    assert result["score"] == 82.5
    assert result["baseline_score"] == 80.0
    assert result["score_delta"] == 2.5
    assert result["inference_time_seconds"] == 59.7


def test_ingest_local_strategy_result_from_candidate_evaluation(tmp_path: Path) -> None:
    from pde_agent.core.local_strategy_result_ingestion import (
        ingest_local_strategy_result,
    )

    source_path = _write(
        tmp_path / "candidate.json",
        {
            "state_hint": "FNO_CANDIDATE_EVALUATED",
            "candidate": {"experiment_id": "candidate_001"},
            "score": {"competition_like_score_short": 43.0},
            "baseline_comparison": {
                "baseline_score": 40.0,
                "score_delta": 3.0,
            },
        },
    )

    result = ingest_local_strategy_result(
        source_path=source_path,
        output_path=tmp_path / "result.json",
        baseline_score=None,
        result_status="completed",
        failure_reason=None,
    )

    assert result["score"] == 43.0
    assert result["baseline_score"] == 40.0
    assert result["score_delta"] == 3.0
    assert result["source"]["candidate"]["experiment_id"] == "candidate_001"


def test_ingest_local_strategy_result_cli_records_timeout(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    source_path = _write(
        tmp_path / "timeout.json",
        {"state_hint": "COMMAND_TIMED_OUT", "elapsed_seconds": 125.0},
    )
    output_path = tmp_path / "result.json"
    result = CliRunner().invoke(
        app,
        [
            "ingest-local-strategy-result",
            "--source-path",
            str(source_path),
            "--output-path",
            str(output_path),
            "--result-status",
            "failed",
            "--failure-reason",
            "inference timeout",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "LOCAL_EXPERIMENT_RESULT"
    assert payload["metrics"]["result_status"] == "failed"
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["failure_reason"] == "inference timeout"
    assert saved["inference_time_seconds"] == 125.0
