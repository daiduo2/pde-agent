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


def _mixed_candidates(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "ALGORITHM_CANDIDATES_COMPILED",
                "algorithm_candidates": [
                    {
                        "algorithm_candidate_id": "candidate_residual",
                        "candidate_family": "lightweight_residual_corrector",
                        "target_problem": "PDE_Burgers",
                        "target_task": "task1",
                        "hypothesis": "Residual correction may improve rollout.",
                        "mechanism": "Correct horizon bias.",
                        "executor_spec": {
                            "cli_command": "run-blueprint-sandbox",
                            "required_options": [],
                        },
                        "risk_controls": ["overfit"],
                        "source_grounding": {
                            "source_ids": ["source_residual"],
                            "algorithm_card_id": "card_residual",
                        },
                    },
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
                    },
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
    assert report["sandbox_queue"] == [action["sandbox_plan_path"]]
    assert report["strategy_growth_queue"] == []
    assert report["next_command"] == "run-checkpoint-family-sandbox"


def test_hermes_literature_expansion_ranks_checkpoint_adapters_first(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_hermes_literature_strategy_expansion,
    )

    candidates_path = tmp_path / "algorithm_candidates.json"
    _mixed_candidates(candidates_path)
    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")

    report = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=candidates_path,
        workspace_dir=tmp_path / "workspace",
        checkpoint_paths=[checkpoint],
        force=False,
    )

    rankings = report["candidate_rankings"]
    assert rankings[0]["algorithm_candidate_id"] == "candidate_fno_blocked"
    assert rankings[0]["route"] == "checkpoint_adapter"
    assert rankings[0]["selection_score"] > rankings[1]["selection_score"]
    assert report["actions"][0]["algorithm_candidate_id"] == "candidate_fno_blocked"


def test_hermes_literature_expansion_skips_failed_memory(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        run_hermes_literature_strategy_expansion,
    )

    candidates_path = tmp_path / "algorithm_candidates.json"
    _mixed_candidates(candidates_path)
    checkpoint = tmp_path / "burgers_FNO.tar"
    checkpoint.write_bytes(b"checkpoint")
    memory_path = tmp_path / "failure_memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED",
                "failed_algorithm_candidate_ids": ["candidate_fno_blocked"],
                "failed_candidate_keys": [],
            }
        ),
        encoding="utf-8",
    )

    report = run_hermes_literature_strategy_expansion(
        algorithm_candidates_path=candidates_path,
        workspace_dir=tmp_path / "workspace",
        checkpoint_paths=[checkpoint],
        failure_memory_path=memory_path,
        force=False,
    )

    assert report["skipped_count"] == 1
    assert report["skipped_candidates"][0]["algorithm_candidate_id"] == (
        "candidate_fno_blocked"
    )
    assert report["actions"][0]["algorithm_candidate_id"] == "candidate_residual"


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
    assert report["sandbox_queue"] == []
    assert report["strategy_growth_queue"] == [action["strategy_proposal_path"]]


def test_record_hermes_literature_candidate_outcome_adds_failure_memory(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hermes_literature_expansion import (
        record_hermes_literature_candidate_outcome,
    )

    report = record_hermes_literature_candidate_outcome(
        algorithm_candidate_id="candidate_fno_blocked",
        candidate_key="card_fno|fourier_neural_operator_architecture",
        outcome="failed",
        reason_codes=["sandbox_score_regressed"],
        output_path=tmp_path / "failure_memory.json",
        force=False,
    )

    assert report["state_hint"] == "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED"
    assert report["failed_algorithm_candidate_ids"] == ["candidate_fno_blocked"]
    assert report["failed_candidate_keys"] == [
        "card_fno|fourier_neural_operator_architecture"
    ]


def test_hypothesis_lab_builds_card_from_literature_expansion_failure(
    tmp_path: Path,
) -> None:
    from pde_agent.core.hypothesis_lab import build_knowledge_cards

    expansion_path = tmp_path / "expansion.json"
    expansion_path.write_text(
        json.dumps(
            {
                "state_hint": "HERMES_LITERATURE_STRATEGY_EXPANSION_RAN",
                "actions": [
                    {
                        "algorithm_candidate_id": "candidate_fno_blocked",
                        "action": "record_executor_design_proposal",
                        "executable": False,
                        "next_command": "build-strategy-rubric",
                    }
                ],
                "skipped_candidates": [
                    {
                        "algorithm_candidate_id": "candidate_old",
                        "skip_reason": "failed_memory",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cards = build_knowledge_cards(
        manual_report_paths=[],
        literature_paths=[expansion_path],
        output_path=tmp_path / "cards.json",
        force=False,
    )

    card = cards["cards"][0]
    assert card["source_type"] == "literature_expansion_report"
    assert card["implementation_hooks"] == ["avoid_failed_literature_candidate"]
    assert "candidate_old" in card["known_risks"][0]


def test_record_hermes_literature_candidate_outcome_cli_returns_payload(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from pde_agent.cli import app

    result = CliRunner().invoke(
        app,
        [
            "record-hermes-literature-candidate-outcome",
            "--algorithm-candidate-id",
            "candidate_fno_blocked",
            "--candidate-key",
            "card_fno|fourier_neural_operator_architecture",
            "--outcome",
            "failed",
            "--reason-code",
            "sandbox_score_regressed",
            "--output-path",
            str(tmp_path / "failure_memory.json"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["state_hint"] == "HERMES_LITERATURE_FAILURE_MEMORY_RECORDED"
    assert payload["metrics"]["failed_candidate_count"] == 1


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
