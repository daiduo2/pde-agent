import json
from pathlib import Path


def _write_registry(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state_hint": "STRATEGY_SKILL_REGISTRY_UPDATED",
                "strategy_registry_path": str(path),
                "entry_count": 1,
                "entries": [
                    {
                        "state_hint": "STRATEGY_SKILL_REGISTERED",
                        "strategy_id": "spectral_residual_m16_ridge1e-03",
                        "skill_name": "spectral-residual-window-stratified-burgers",
                        "installed_skill_dir": str(path.parent / "skill"),
                        "installed_skill_md_path": str(path.parent / "skill" / "SKILL.md"),
                        "model_family": "SpectralResidual",
                        "source_state_hint": (
                            "SPECTRAL_RESIDUAL_WINDOW_STRATIFICATION_COMPLETED"
                        ),
                        "metrics": {
                            "evaluation_window_count": 8,
                            "mean_score_delta": 42.95,
                            "score_delta_std": 0.66,
                            "failure_rate": 0.0,
                        },
                        "routing": {
                            "applicable_pde": "1D Burgers",
                            "preferred_when": ["real-HDF5 Burgers rollout evidence is required"],
                        },
                    }
                ],
                "next_command": "strategy-growth-complete",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_lookup_strategy_skills_returns_ranked_burgers_match(tmp_path: Path) -> None:
    from pde_agent.core.strategy_skill_registry import lookup_strategy_skills

    registry_path = _write_registry(tmp_path / "registry.json")

    result = lookup_strategy_skills(
        registry_path=registry_path,
        applicable_pde="1D Burgers",
        model_family="SpectralResidual",
        max_results=3,
        output_path=tmp_path / "lookup.json",
    )

    assert result["state_hint"] == "STRATEGY_SKILL_LOOKUP_COMPLETED"
    assert result["match_count"] == 1
    assert result["matches"][0]["skill_name"] == (
        "spectral-residual-window-stratified-burgers"
    )
    assert result["matches"][0]["selection_score"] > 0
    assert Path(result["lookup_report_path"]).exists()


def test_lookup_strategy_skill_cli_returns_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    registry_path = _write_registry(tmp_path / "registry.json")
    output_path = tmp_path / "lookup.json"

    result = CliRunner().invoke(
        app,
        [
            "lookup-strategy-skill",
            "--registry-path",
            str(registry_path),
            "--applicable-pde",
            "1D Burgers",
            "--model-family",
            "SpectralResidual",
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "lookup-strategy-skill"
    assert payload["state_hint"] == "STRATEGY_SKILL_LOOKUP_COMPLETED"
    assert payload["metrics"]["match_count"] == 1
    assert payload["next_hint"] == "use-registered-strategy-skill"
