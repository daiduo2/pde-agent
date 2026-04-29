from pathlib import Path

import numpy as np


def _write_hdf5(path: Path, data: np.ndarray, key: str = "tensor") -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=data)
    return path


def test_generate_hermes_postprocess_candidate_builds_valid_context_anchor_candidate(
    tmp_path: Path,
) -> None:
    from pde_agent.core.competition_scoring import score_competition_prediction
    from pde_agent.core.competition_submission import build_competition_submission
    from pde_agent.core.competition_submission_audit import audit_competition_submission
    from pde_agent.core.hermes_postprocess_candidate import (
        generate_hermes_postprocess_candidate,
    )

    context = np.ones((1, 10, 256), dtype=np.float32)
    poor_future = np.zeros((1, 190, 256), dtype=np.float32)
    target = np.concatenate([context, np.ones_like(poor_future)], axis=1)
    test_path = _write_hdf5(tmp_path / "task1_test.hdf5", context)
    target_path = _write_hdf5(tmp_path / "task1_val.hdf5", target)
    poor_prediction_path = tmp_path / "poor_future.npz"
    np.savez(poor_prediction_path, prediction=poor_future)
    base = build_competition_submission(
        task_id=1,
        test_path=test_path,
        output_dir=tmp_path / "base",
        submission_id="team_hermes",
        method="external_prediction",
        source_prediction_path=poor_prediction_path,
        force=True,
    )
    base_score = score_competition_prediction(
        task_id=1,
        prediction_path=base["artifacts"]["prediction_path"],
        target_path=target_path,
        output_path=None,
    )

    candidate = generate_hermes_postprocess_candidate(
        base_submission_dir=base["artifacts"]["submission_dir"],
        output_dir=tmp_path / "candidate",
        task_id=1,
        test_path=test_path,
        strategy="context_anchor_blend",
        blend_weight=1.0,
        submission_id="team_hermes",
        force=False,
    )

    audit = audit_competition_submission(
        task_id=1,
        submission_dir=candidate["submission_dir"],
        test_path=test_path,
        output_path=None,
    )
    candidate_score = score_competition_prediction(
        task_id=1,
        prediction_path=candidate["artifacts"]["prediction_path"],
        target_path=target_path,
        output_path=None,
    )

    assert candidate["state_hint"] == "HERMES_POSTPROCESS_CANDIDATE_READY"
    assert candidate["strategy"] == "context_anchor_blend"
    assert audit["decision"] == "passed"
    assert candidate_score["segment_score"] > base_score["segment_score"]


def test_generate_hermes_postprocess_candidate_cli_returns_payload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from pde_agent.cli import app
    import pde_agent.commands.generate_hermes_postprocess_candidate as command_module
    from typer.testing import CliRunner

    def fake_generator(**_: object) -> dict[str, object]:
        report_path = tmp_path / "candidate.json"
        report_path.write_text("{}", encoding="utf-8")
        return {
            "state_hint": "HERMES_POSTPROCESS_CANDIDATE_READY",
            "candidate_report_path": str(report_path),
            "candidate_id": "postprocess_context_anchor_blend",
            "strategy": "context_anchor_blend",
            "submission_dir": str(tmp_path / "submission"),
            "artifacts": {
                "candidate_report_path": str(report_path),
                "submission_dir": str(tmp_path / "submission"),
                "submission_zip_path": str(tmp_path / "submission.zip"),
            },
        }

    monkeypatch.setattr(command_module, "generate_hermes_postprocess_candidate", fake_generator)

    result = CliRunner().invoke(
        app,
        [
            "generate-hermes-postprocess-candidate",
            "--base-submission-dir",
            str(tmp_path / "base"),
            "--output-dir",
            str(tmp_path / "candidate"),
            "--task-id",
            "1",
            "--test-path",
            "task1_test.hdf5",
            "--strategy",
            "context_anchor_blend",
        ],
    )

    assert result.exit_code == 0
    payload = __import__("json").loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "HERMES_POSTPROCESS_CANDIDATE_READY"
    assert payload["metrics"]["strategy"] == "context_anchor_blend"
