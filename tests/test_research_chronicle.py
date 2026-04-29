import json
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_research_chronicle_summarizes_artifacts_and_updates_logs(
    tmp_path: Path,
) -> None:
    from pde_agent.core.research_chronicle import build_research_chronicle

    hermes_audit = _write_json(
        tmp_path / "hermes_audit.json",
        {
            "status": "passed",
            "arm": "real_hdf5_deeponet_strategy_sweep",
            "session_id": "20260428_172142_d7472e",
            "tool_call_summary": {"read_file": 3, "terminal": 6},
            "terminal_artifact": {
                "state_hint": "STRATEGY_PROPOSAL_RECORDED",
                "strategy_id": "burgers_spectral_viscosity_real_hdf5_v0",
            },
        },
    )
    benchmark = _write_json(
        tmp_path / "benchmark.json",
        {
            "state_hint": "BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED",
            "decision": {
                "decision": "retrieve_external_strategy",
                "reason_codes": ["negative_mean_score_delta"],
            },
            "metrics": {
                "best_strategy_id": "branch_trunk_spectral",
                "mean_score_delta": -11.8,
                "failure_rate": 1.0,
            },
        },
    )
    pipeline = _write_json(
        tmp_path / "pipeline.json",
        {
            "state_hint": "COMPETITION_TASK_PIPELINE_COMPLETED",
            "task_id": 1,
            "model_family": "fno",
            "audit": {"decision": "passed", "issue_count": 0},
            "materialized": {
                "submission": {
                    "metrics": {
                        "prediction_shape": [1000, 200, 256],
                        "inference_time": 51.2,
                    }
                }
            },
        },
    )
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    (submission_dir / "task1_logs.log").write_text(
        "Hermes PDE Agent Scientific Log\nExperiment Tracking:\n",
        encoding="utf-8",
    )

    report = build_research_chronicle(
        artifact_paths=[hermes_audit, benchmark, pipeline],
        output_path=tmp_path / "research_chronicle.md",
        submission_dir=submission_dir,
        task_ids=[1],
        force=True,
    )

    chronicle = Path(report["artifacts"]["chronicle_path"]).read_text(
        encoding="utf-8"
    )
    assert report["state_hint"] == "RESEARCH_CHRONICLE_BUILT"
    assert "Hermes Research Chronicle" in chronicle
    assert "real_hdf5_deeponet_strategy_sweep" in chronicle
    assert "retrieve_external_strategy" in chronicle
    assert "negative_mean_score_delta" in chronicle
    assert "burgers_spectral_viscosity_real_hdf5_v0" in chronicle
    assert "model_family=fno" in chronicle
    assert "Arxiv Retrieval Policy" in chronicle

    updated_log = (submission_dir / "task1_logs.log").read_text(encoding="utf-8")
    assert "Hermes Research Chronicle" in updated_log
    assert "Evidence Sources" in updated_log


def test_build_research_chronicle_cli_returns_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    artifact = _write_json(
        tmp_path / "pipeline.json",
        {
            "state_hint": "COMPETITION_TASK_PIPELINE_COMPLETED",
            "task_id": 2,
            "model_family": "fno",
            "audit": {"decision": "passed", "issue_count": 0},
        },
    )
    output_path = tmp_path / "chronicle.md"

    result = CliRunner().invoke(
        app,
        [
            "build-research-chronicle",
            "--artifact-path",
            str(artifact),
            "--output-path",
            str(output_path),
            "--force",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "build-research-chronicle"
    assert payload["state_hint"] == "RESEARCH_CHRONICLE_BUILT"
    assert payload["metrics"]["artifact_count"] == 1
