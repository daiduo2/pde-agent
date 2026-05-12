"""Tests for FNO training loop monitoring dashboard."""

import json
from pathlib import Path
from typing import Any

import pytest


class TestTrainingLoopSnapshot:
    def test_snapshot_empty_workspace(self, tmp_path: Path) -> None:
        from pde_agent.core.interactive_cockpit import (
            FNO_TRAINING_LOOP_PANEL_STATE_HINT,
            build_fno_training_loop_snapshot,
        )

        result = build_fno_training_loop_snapshot(workspace_dir=str(tmp_path))
        assert result["state_hint"] == FNO_TRAINING_LOOP_PANEL_STATE_HINT
        assert result["iteration_count"] == 0
        assert result["score_history"] == []
        assert result["blocked_families"] == []
        assert result["current_decision"] == "unknown"
        assert result["has_final_report"] is False
        assert result["best_candidate"] is None
        assert "panel_text" in result

    def test_snapshot_with_iterations(self, tmp_path: Path) -> None:
        from pde_agent.core.interactive_cockpit import build_fno_training_loop_snapshot

        # Write memory files
        for i in range(3):
            mem = {
                "schema_version": 1,
                "iteration_index": i,
                "best_so_far": {
                    "candidate_id": f"c{i}",
                    "score": 0.5 + i * 0.1,
                    "model_family": "fno",
                },
                "family_outcomes": {
                    "unet": {"status": "blocked", "failure_reasons": ["diverged"]},
                },
            }
            (tmp_path / f"fno_training_memory_{i:04d}.json").write_text(
                json.dumps(mem, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        # Write convergence files
        for i in range(3):
            conv = {
                "schema_version": 1,
                "decision": "continue" if i < 2 else "converged",
            }
            (tmp_path / f"fno_training_convergence_{i:04d}.json").write_text(
                json.dumps(conv, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        # Write checkpoint
        (tmp_path / "checkpoint_0002.json").write_text("{}")

        # Write final report
        (tmp_path / "fno_training_final_report.json").write_text("{}")

        result = build_fno_training_loop_snapshot(workspace_dir=str(tmp_path))

        assert result["iteration_count"] == 3
        assert len(result["score_history"]) == 3
        assert result["score_history"][0]["score"] == 0.5
        assert result["score_history"][2]["score"] == 0.7
        assert result["current_decision"] == "converged"
        assert result["blocked_families"] == ["unet"]
        assert result["has_final_report"] is True
        assert result["best_candidate"]["score"] == 0.7
        assert "checkpoint_0002.json" in (result["last_checkpoint"] or "")

    def test_snapshot_ignores_non_dict_memory(self, tmp_path: Path) -> None:
        from pde_agent.core.interactive_cockpit import build_fno_training_loop_snapshot

        (tmp_path / "fno_training_memory_0000.json").write_text("null")
        result = build_fno_training_loop_snapshot(workspace_dir=str(tmp_path))
        assert result["iteration_count"] == 1
        assert result["score_history"] == []

    def test_panel_text_format(self, tmp_path: Path) -> None:
        from pde_agent.core.interactive_cockpit import build_fno_training_loop_snapshot

        mem = {
            "schema_version": 1,
            "iteration_index": 0,
            "best_so_far": {"candidate_id": "c1", "score": 0.8, "model_family": "fno"},
            "family_outcomes": {},
        }
        (tmp_path / "fno_training_memory_0000.json").write_text(json.dumps(mem))
        conv = {"schema_version": 1, "decision": "converged"}
        (tmp_path / "fno_training_convergence_0000.json").write_text(json.dumps(conv))

        result = build_fno_training_loop_snapshot(workspace_dir=str(tmp_path))
        text = result["panel_text"]

        assert "FNO Training Loop Monitor" in text
        assert "Iterations: 1" in text
        assert "converged" in text
        assert "score=0.8" in text
        assert "c1" in text

    def test_cli_monitor_command(self, tmp_path: Path) -> None:
        from pde_agent.commands.interactive_cockpit import (
            build_fno_training_loop_snapshot_command,
        )

        result = build_fno_training_loop_snapshot_command(workspace_dir=str(tmp_path))
        assert result["state_hint"] == "FNO_TRAINING_LOOP_PANEL_BUILT"
        assert result["iteration_count"] == 0
