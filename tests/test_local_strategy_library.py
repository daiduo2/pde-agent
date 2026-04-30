import json
from pathlib import Path


def test_build_local_strategy_library_contains_canonical_operator_entries(
    tmp_path: Path,
) -> None:
    from pde_agent.core.local_strategy_library import build_local_strategy_library

    output_path = tmp_path / "local_strategy_library.json"

    result = build_local_strategy_library(output_path=output_path, force=False)

    assert result["state_hint"] == "LOCAL_STRATEGY_LIBRARY_BUILT"
    assert result["entry_count"] >= 10
    assert output_path.exists()
    assert {entry["source_type"] for entry in result["entries"]} == {
        "local_canonical_skill"
    }

    model_families = {
        family
        for entry in result["entries"]
        for family in entry["model_families"]
    }
    assert {"FNO", "UNet", "DeepONet", "PI-DeepONet", "DataInference"} <= (
        model_families
    )

    for entry in result["entries"]:
        assert entry["strategy_id"]
        assert entry["title"]
        assert entry["hypothesis"]
        assert entry["implementation_hooks"]
        assert entry["expected_observables"]
        assert entry["acceptance_criteria"]
        assert entry["evaluation_protocol"]["primary_metric"] == (
            "competition_segment_score"
        )


def test_lookup_local_strategies_ranks_fno_burgers_long_horizon(
    tmp_path: Path,
) -> None:
    from pde_agent.core.local_strategy_library import (
        build_local_strategy_library,
        lookup_local_strategies,
    )

    library_path = tmp_path / "library.json"
    report_path = tmp_path / "lookup.json"
    build_local_strategy_library(output_path=library_path, force=False)

    result = lookup_local_strategies(
        library_path=library_path,
        applicable_pde="1D Burgers",
        model_family="FNO",
        failure_modes=["long_horizon_drift", "spectral_dissipation_mismatch"],
        max_results=3,
        output_path=report_path,
    )

    assert result["state_hint"] == "LOCAL_STRATEGY_LOOKUP_COMPLETED"
    assert result["match_count"] == 3
    assert report_path.exists()
    assert result["matches"][0]["strategy_id"] == "fno_spectral_viscosity_filter_v0"
    assert result["matches"][0]["selection_score"] > result["matches"][-1][
        "selection_score"
    ]
    assert result["matches"][0]["implementation_hooks"]
    assert result["matches"][0]["acceptance_criteria"]
    assert result["next_command"] == "schedule-local-strategy-evaluation"


def test_lookup_local_strategies_prioritizes_unet_runtime_tuning(
    tmp_path: Path,
) -> None:
    from pde_agent.core.local_strategy_library import (
        build_local_strategy_library,
        lookup_local_strategies,
    )

    library_path = tmp_path / "library.json"
    build_local_strategy_library(output_path=library_path, force=False)

    result = lookup_local_strategies(
        library_path=library_path,
        applicable_pde="1D Burgers",
        model_family="UNet",
        failure_modes=["inference_timeout", "runtime_degradation"],
        max_results=2,
        output_path=None,
    )

    assert result["matches"][0]["strategy_id"] == "unet_runtime_chunk_thread_tuning_v0"
    assert "run-hermes-operator-tuning-loop" in result["matches"][0][
        "implementation_hooks"
    ]


def test_lookup_local_strategies_exposes_hypothesis_lab_linear_trend_spectral(
    tmp_path: Path,
) -> None:
    from pde_agent.core.local_strategy_library import (
        build_local_strategy_library,
        lookup_local_strategies,
    )

    library_path = tmp_path / "library.json"
    build_local_strategy_library(output_path=library_path, force=False)

    result = lookup_local_strategies(
        library_path=library_path,
        applicable_pde="1D Burgers",
        model_family="DataInference",
        failure_modes=["persistence_plateau", "long_horizon_drift"],
        max_results=1,
        output_path=None,
    )

    assert result["matches"][0]["strategy_id"] == (
        "data_inference_linear_trend_spectral_v0"
    )
    assert "run-hypothesis-lab" in result["matches"][0]["implementation_hooks"]
    assert "linear_trend_spectral" in result["matches"][0]["implementation_hooks"]


def test_local_strategy_library_cli_build_and_lookup(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    runner = CliRunner()
    library_path = tmp_path / "library.json"
    lookup_path = tmp_path / "lookup.json"

    build = runner.invoke(
        app,
        [
            "build-local-strategy-library",
            "--output-path",
            str(library_path),
        ],
    )

    assert build.exit_code == 0
    build_payload = json.loads(build.stdout)
    assert build_payload["status"] == "ok"
    assert build_payload["state_hint"] == "LOCAL_STRATEGY_LIBRARY_BUILT"
    assert build_payload["metrics"]["entry_count"] >= 10

    lookup = runner.invoke(
        app,
        [
            "lookup-local-strategy",
            "--library-path",
            str(library_path),
            "--applicable-pde",
            "1D Burgers",
            "--model-family",
            "UNet",
            "--failure-mode",
            "autoregressive_blur",
            "--output-path",
            str(lookup_path),
        ],
    )

    assert lookup.exit_code == 0
    lookup_payload = json.loads(lookup.stdout)
    assert lookup_payload["status"] == "ok"
    assert lookup_payload["state_hint"] == "LOCAL_STRATEGY_LOOKUP_COMPLETED"
    assert lookup_payload["metrics"]["match_count"] >= 1
    assert lookup_payload["metrics"]["top_strategy_id"].startswith("unet_")
