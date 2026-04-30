from __future__ import annotations

import json
from pathlib import Path


def _blocked_fno_candidates(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": "candidate_fno_blocked",
                        "parent_algorithm_card_id": "card_fno",
                        "candidate_family": "unsupported",
                        "target_problem": "PDE_Burgers",
                        "target_task": "task1",
                        "hypothesis": "FNO improves Burgers rollout.",
                        "mechanism": "Fourier kernels learn global modes.",
                        "executor_spec": {
                            "cli_command": "blocked_requires_executor_design",
                            "unsupported_hook": "fourier_neural_operator_architecture",
                            "required_options": [],
                        },
                        "risk_controls": [
                            "blocked until a bounded executor is designed"
                        ],
                        "source_grounding": {
                            "source_ids": ["source_fno"],
                            "algorithm_card_id": "card_fno",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_hermes_literature_expansion_adapts_fno_candidate_to_checkpoint(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_hermes_literature_strategy_expansion,
    )

    candidates_path = tmp_path / "algorithm_candidates.json"
    _blocked_fno_candidates(candidates_path)
    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")

    report = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=candidates_path,
        workspace_dir=tmp_path / "workspace",
        checkpoint_paths=[checkpoint],
        force=False,
    )

    assert report["state_hint"] == "HERMES_LITERATURE_STRATEGY_EXPANSION_RAN"
    assert report["action_count"] == 1
    action = report["actions"][0]
    assert action["action"] == "adapt_to_checkpoint_sandbox"
    assert action["executable"] is True
    assert action["next_command"] == "run-checkpoint-family-sandbox"
    assert Path(action["adapted_candidate_path"]).exists()
    assert Path(action["sandbox_plan_path"]).exists()
    assert report["next_command"] == "run-checkpoint-family-sandbox"


def test_hermes_literature_expansion_records_proposal_for_blocked_candidate(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_hermes_literature_strategy_expansion,
    )

    candidates_path = tmp_path / "algorithm_candidates.json"
    _blocked_fno_candidates(candidates_path)

    report = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=candidates_path,
        workspace_dir=tmp_path / "workspace",
        checkpoint_paths=[],
        force=False,
    )

    action = report["actions"][0]
    assert action["action"] == "record_executor_design_proposal"
    assert action["executable"] is False
    assert action["next_command"] == "build-strategy-rubric"
    proposal = json.loads(Path(action["strategy_proposal_path"]).read_text())
    assert proposal["state_hint"] == "STRATEGY_PROPOSAL_RECORDED"
    assert proposal["source_type"] == "external_retrieval"
    assert proposal["implementation_plan"]["executor_design_request_path"] == (
        action["executor_design_request_path"]
    )
    assert report["proposal_count"] == 1


def test_hermes_literature_expansion_cli_returns_payload(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    candidates_path = tmp_path / "algorithm_candidates.json"
    _blocked_fno_candidates(candidates_path)

    result = CliRunner().invoke(
        app,
        [
            "run-hermes-literature-strategy-expansion",
            "--algorithm-candidates-path",
            str(candidates_path),
            "--workspace-dir",
            str(tmp_path / "workspace"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "HERMES_LITERATURE_STRATEGY_EXPANSION_RAN"
    assert payload["metrics"]["proposal_count"] == 1
    assert payload["next_hint"] == "build-strategy-rubric"
