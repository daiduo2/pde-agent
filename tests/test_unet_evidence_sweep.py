import json
from pathlib import Path

from pde_agent.core.unet_candidate_evaluation import evaluate_unet_candidate
from pde_agent.core.unet_candidate_execution import execute_unet_candidate
from pde_agent.core.unet_evidence_sweep import build_unet_evidence_sweep


def _write_windows(path: Path) -> None:
    import numpy as np

    x = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    context = np.zeros((3, 4, 12), dtype=np.float32)
    target = np.zeros((3, 5, 12), dtype=np.float32)
    for sample in range(3):
        base = np.sin(2.0 * np.pi * (x + 0.02 * sample))
        trend = 0.03 * np.cos(2.0 * np.pi * x) + 0.01 * sample
        for time_index in range(context.shape[1]):
            context[sample, time_index, :] = base + time_index * trend
        for time_index in range(target.shape[1]):
            target[sample, time_index, :] = base + (time_index + 4) * trend
    np.savez(path, context=context, target=target)


def _evaluation_report(
    tmp_path: Path,
    *,
    name: str,
    training_strategy: str,
    start_index: int,
) -> Path:
    windows_path = tmp_path / f"{name}_windows.npz"
    baseline_score_path = tmp_path / f"{name}_baseline_score.json"
    _write_windows(windows_path)
    baseline_score_path.write_text(
        json.dumps({"competition_like_score_short": 1.0}),
        encoding="utf-8",
    )
    candidate = execute_unet_candidate(
        windows_path=windows_path,
        output_dir=tmp_path / name / "candidate",
        start_index=start_index,
        train_steps=4,
        training_strategy=training_strategy,
        base_channels=8,
        depth=2,
        learning_rate=1e-2,
        seed=7,
        force=False,
    )
    evaluation = evaluate_unet_candidate(
        candidate_report_path=candidate["artifacts"]["candidate_execution_report_path"],
        target_path=windows_path,
        output_dir=tmp_path / name / "evaluation",
        baseline_score_path=baseline_score_path,
        force=False,
    )
    return Path(evaluation["artifacts"]["evaluation_report_path"])


def test_build_unet_evidence_sweep_writes_summary_and_csv(tmp_path: Path) -> None:
    supervised = _evaluation_report(
        tmp_path,
        name="supervised",
        training_strategy="unet_supervised",
        start_index=1,
    )
    spectral = _evaluation_report(
        tmp_path,
        name="spectral",
        training_strategy="unet_spectral",
        start_index=1,
    )
    output_dir = tmp_path / "evidence"

    summary = build_unet_evidence_sweep(
        evaluation_report_paths=[supervised, spectral],
        output_dir=output_dir,
        force=False,
    )

    assert summary["state_hint"] == "UNET_EVIDENCE_SWEEP_COMPLETED"
    assert summary["model_family"] == "UNet"
    assert summary["record_count"] == 2
    assert summary["comparison_count"] == 1
    assert summary["comparison_winners"][0]["winner_experiment_id"].startswith("unet_")
    assert summary["strategy_statistics"]["unet_supervised"]["record_count"] == 1
    assert summary["strategy_statistics"]["unet_spectral"]["record_count"] == 1
    assert summary["next_command"] == "convert-operator-evidence-summary"
    assert Path(summary["artifacts"]["summary_json_path"]).exists()
    assert Path(summary["artifacts"]["summary_csv_path"]).exists()
    written = json.loads(
        Path(summary["artifacts"]["summary_json_path"]).read_text(encoding="utf-8")
    )
    assert written == summary


def test_build_unet_evidence_sweep_rejects_duplicate_experiment_ids(
    tmp_path: Path,
) -> None:
    import pytest

    report_path = _evaluation_report(
        tmp_path,
        name="supervised",
        training_strategy="unet_supervised",
        start_index=1,
    )

    with pytest.raises(Exception, match="duplicate experiment_id"):
        build_unet_evidence_sweep(
            evaluation_report_paths=[report_path, report_path],
            output_dir=tmp_path / "evidence",
            force=False,
        )


def test_build_unet_evidence_sweep_cli_writes_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    report_path = _evaluation_report(
        tmp_path,
        name="supervised",
        training_strategy="unet_supervised",
        start_index=1,
    )
    output_dir = tmp_path / "evidence"

    result = CliRunner().invoke(
        app,
        [
            "build-unet-evidence-sweep",
            "--evaluation-report-path",
            str(report_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "build-unet-evidence-sweep"
    assert payload["state_hint"] == "UNET_EVIDENCE_SWEEP_COMPLETED"
    assert payload["metrics"]["record_count"] == 1
    assert payload["metrics"]["comparison_count"] == 1
    assert payload["artifacts"]["summary_json_path"]
    assert payload["next_hint"] == "convert-operator-evidence-summary"
