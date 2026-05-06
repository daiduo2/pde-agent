from pathlib import Path


def test_task2_source_policy_allows_task2_training_without_checkpoint(tmp_path: Path) -> None:
    from pde_agent.core.task2_source_policy import audit_task2_source_policy

    train_path = tmp_path / "task2_multi_nu_train.hdf5"
    test_path = tmp_path / "task2_test.hdf5"

    report = audit_task2_source_policy(
        train_data_paths=[train_path],
        test_data_paths=[test_path],
        checkpoint_paths=[],
        candidate_source_paths=[],
        output_path=None,
    )

    assert report["state_hint"] == "TASK2_SOURCE_POLICY_AUDITED"
    assert report["decision"] == "passed"
    assert report["issue_codes"] == []
    assert report["allowed_reuse"]["agent_code_and_search_policy"] is True
    assert report["forbidden_reuse"]["task1_data_or_checkpoints"] is True


def test_task2_source_policy_rejects_task1_data_and_checkpoints(tmp_path: Path) -> None:
    from pde_agent.core.task2_source_policy import audit_task2_source_policy

    report = audit_task2_source_policy(
        train_data_paths=[
            tmp_path / "1D_Burgers_Sols_Nu0.001.hdf5",
            tmp_path / "task1_val.hdf5",
        ],
        test_data_paths=[tmp_path / "task2_test.hdf5"],
        checkpoint_paths=[tmp_path / "1D_Burgers_Sols_Nu0.001_FNO.pt"],
        candidate_source_paths=[
            tmp_path / "runs" / "task1_current_best_20260503" / "submission",
        ],
        output_path=None,
    )

    assert report["decision"] == "failed"
    assert "task1_data_path_forbidden" in report["issue_codes"]
    assert "task1_checkpoint_forbidden" in report["issue_codes"]
    assert "task1_candidate_artifact_forbidden" in report["issue_codes"]
    assert report["next_command"] == "replace_task1_sources_before_task2_training"


def test_task2_source_policy_allows_task2_from_scratch_checkpoint_artifact(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task2_source_policy import audit_task2_source_policy

    report = audit_task2_source_policy(
        train_data_paths=[tmp_path / "task2_train.hdf5"],
        test_data_paths=[tmp_path / "task2_test.hdf5"],
        checkpoint_paths=[tmp_path / "runs" / "task2_fno_from_scratch" / "checkpoint.pt"],
        candidate_source_paths=[],
        output_path=None,
    )

    assert report["decision"] == "passed"
    assert report["issue_codes"] == []


def test_task2_source_policy_writes_report(tmp_path: Path) -> None:
    from pde_agent.core.task2_source_policy import audit_task2_source_policy

    output_path = tmp_path / "audit.json"
    report = audit_task2_source_policy(
        train_data_paths=[tmp_path / "task2_train.hdf5"],
        test_data_paths=[],
        checkpoint_paths=[],
        candidate_source_paths=[],
        output_path=output_path,
    )

    assert report["audit_path"] == str(output_path)
    assert output_path.exists()


def test_task2_source_policy_cli_returns_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    output_path = tmp_path / "audit.json"
    result = CliRunner().invoke(
        app,
        [
            "audit-task2-source-policy",
            "--train-data-path",
            str(tmp_path / "task2_train.hdf5"),
            "--checkpoint-path",
            str(tmp_path / "1D_Burgers_Sols_Nu0.001_Unet-PF-20.pt"),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = result.stdout
    assert "TASK2_SOURCE_POLICY_AUDITED" in payload
    assert "task1_checkpoint_forbidden" in output_path.read_text(encoding="utf-8")
