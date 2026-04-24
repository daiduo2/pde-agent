import json
from pathlib import Path

import numpy as np
import pytest

from pde_agent.commands.analyze_spectrum import analyze_spectrum
from pde_agent.commands.check_physics import check_physics
from pde_agent.commands.collect_artifacts import collect_artifacts
from pde_agent.commands.eval_rollout import eval_rollout
from pde_agent.commands.interpret_error import interpret_error
from pde_agent.commands.record_timing import record_timing
from pde_agent.commands.select_hypothesis import select_hypothesis


def _prepare_rollout(run_root: Path) -> None:
    eval_rollout(run_id="diagnostic_smoke_001", run_root=str(run_root))


def _write_npz_configured_run(tmp_path: Path) -> tuple[Path, Path]:
    source_path = tmp_path / "diagnostic_fixture.npz"
    x = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    target_frame = np.sin(x)
    prediction_frame = target_frame + 0.2 * np.sin(3.0 * x)
    target = np.tile(target_frame, (2, 4, 1)).astype(np.float64)
    prediction = np.tile(prediction_frame, (2, 4, 1)).astype(np.float64)
    np.savez(
        source_path,
        context=np.zeros((2, 3, 8), dtype=np.float64),
        prediction=prediction,
        target=target,
    )
    run_root = tmp_path / "diagnostic_real_001"
    config_path = run_root / "config" / "resolved.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "mode": "npz_fixture",
                    "source_path": str(source_path),
                    "context_steps": 3,
                    "forecast_steps": 4,
                }
            }
        ),
        encoding="utf-8",
    )
    return run_root, source_path


def test_analyze_spectrum_writes_deterministic_spectrum_metrics(tmp_path: Path) -> None:
    run_root = tmp_path / "diagnostic_smoke_001"
    _prepare_rollout(run_root)

    result = analyze_spectrum(run_id="diagnostic_smoke_001", run_root=str(run_root))

    spectrum_path = run_root / "metrics" / "spectrum.json"
    payload = json.loads(spectrum_path.read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert result["state_hint"] == "SPECTRUM_ANALYZED"
    assert result["artifacts"]["spectrum_metrics_path"] == str(spectrum_path)
    assert payload["mode"] == "synthetic_pde_diagnostic"
    assert payload["low_frequency_error"] == 0.024
    assert payload["mid_frequency_error"] == 0.052
    assert payload["high_frequency_error"] == 0.141
    assert payload["spectral_drift_score"] == 0.37
    assert payload["shock_region_proxy"] == 0.68
    assert payload["dominant_spectral_issue"] == "excessive_smoothing"


def test_analyze_spectrum_writes_real_npz_spectrum_metrics(tmp_path: Path) -> None:
    run_root, _source_path = _write_npz_configured_run(tmp_path)

    result = analyze_spectrum(run_id="diagnostic_real_001", run_root=str(run_root))

    spectrum_path = run_root / "metrics" / "spectrum.json"
    payload = json.loads(spectrum_path.read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert result["metrics"]["source_mode"] == "npz_fixture"
    assert result["metrics"]["high_frequency_error"] == pytest.approx(
        payload["high_frequency_error"]
    )
    assert payload["mode"] == "real_array_diagnostic"
    assert payload["source_mode"] == "npz_fixture"
    assert payload["run_id"] == "diagnostic_real_001"
    assert payload["spectral_drift_score"] > 0.0
    assert payload["dominant_spectral_issue"] in {
        "low_frequency_error",
        "mid_frequency_error",
        "high_frequency_error",
    }


def test_check_physics_writes_deterministic_physics_indicators(tmp_path: Path) -> None:
    run_root = tmp_path / "diagnostic_smoke_001"
    _prepare_rollout(run_root)

    result = check_physics(run_id="diagnostic_smoke_001", run_root=str(run_root))

    physics_path = run_root / "metrics" / "physics_check.json"
    payload = json.loads(physics_path.read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert result["state_hint"] == "PHYSICS_CHECKED"
    assert result["artifacts"]["physics_check_path"] == str(physics_path)
    assert payload["mode"] == "synthetic_pde_diagnostic"
    assert payload["mass_drift"] == 0.006
    assert payload["energy_decay_ratio"] == 0.82
    assert payload["total_variation_ratio"] == 0.71
    assert payload["nonphysical_oscillation"] is False
    assert payload["blowup_like_behavior"] is False
    assert payload["excessive_smoothing"] is True
    assert payload["physics_issue"] == "overdissipation"


def test_check_physics_writes_real_npz_physics_metrics(tmp_path: Path) -> None:
    run_root, _source_path = _write_npz_configured_run(tmp_path)

    result = check_physics(run_id="diagnostic_real_001", run_root=str(run_root))

    physics_path = run_root / "metrics" / "physics_check.json"
    payload = json.loads(physics_path.read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert result["metrics"]["source_mode"] == "npz_fixture"
    assert result["metrics"]["mass_drift"] == pytest.approx(payload["mass_drift"])
    assert payload["mode"] == "real_array_diagnostic"
    assert payload["source_mode"] == "npz_fixture"
    assert payload["run_id"] == "diagnostic_real_001"
    assert payload["energy_decay_ratio"] > 0.0
    assert payload["physics_issue"] in {
        "none",
        "blowup_like_behavior",
        "nonphysical_oscillation",
        "excessive_smoothing",
        "mass_drift",
    }


def test_interpret_error_writes_failure_mode_from_rollout_spectrum_and_physics(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "diagnostic_smoke_001"
    _prepare_rollout(run_root)
    analyze_spectrum(run_id="diagnostic_smoke_001", run_root=str(run_root))
    check_physics(run_id="diagnostic_smoke_001", run_root=str(run_root))

    result = interpret_error(run_id="diagnostic_smoke_001", run_root=str(run_root))

    interpretation_path = run_root / "diagnostics" / "error_interpretation.json"
    payload = json.loads(interpretation_path.read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert result["state_hint"] == "ERROR_INTERPRETED"
    assert result["artifacts"]["error_interpretation_path"] == str(interpretation_path)
    assert payload["dominant_failure_mode"] == "excessive_smoothing"
    assert payload["confidence"] == 0.82
    assert payload["recommended_focus"] == (
        "restore_high_frequency_structure_without_destabilizing_rollout"
    )
    assert "high_frequency_error=0.141" in payload["evidence"]
    assert "physics_issue=overdissipation" in payload["evidence"]
    assert payload["rejected_explanations"] == [
        "blowup_like_behavior",
        "nonphysical_oscillation",
    ]


def test_select_hypothesis_writes_bounded_next_experiment(tmp_path: Path) -> None:
    run_root = tmp_path / "diagnostic_smoke_001"
    _prepare_rollout(run_root)
    analyze_spectrum(run_id="diagnostic_smoke_001", run_root=str(run_root))
    check_physics(run_id="diagnostic_smoke_001", run_root=str(run_root))
    interpret_error(run_id="diagnostic_smoke_001", run_root=str(run_root))

    result = select_hypothesis(run_id="diagnostic_smoke_001", run_root=str(run_root))

    hypothesis_path = run_root / "diagnostics" / "selected_hypothesis.json"
    payload = json.loads(hypothesis_path.read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert result["state_hint"] == "HYPOTHESIS_SELECTED"
    assert result["artifacts"]["selected_hypothesis_path"] == str(hypothesis_path)
    assert payload["selected_hypothesis"] == "add_spectral_consistency_loss"
    assert payload["expected_effect"] == (
        "reduce high-frequency damping while preserving rollout stability"
    )
    assert payload["risk"] == "may amplify oscillations near shocks if overweighted"
    assert payload["next_experiment_scope"] == "read_only_proposal_no_training"


def test_collect_artifacts_writes_hypothesis_selected_manifest_when_diagnostics_exist(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "diagnostic_smoke_001"
    (run_root / "config").mkdir(parents=True)
    (run_root / "config" / "resolved.json").write_text("{}", encoding="utf-8")
    _prepare_rollout(run_root)
    record_timing(run_id="diagnostic_smoke_001", run_root=str(run_root))
    analyze_spectrum(run_id="diagnostic_smoke_001", run_root=str(run_root))
    check_physics(run_id="diagnostic_smoke_001", run_root=str(run_root))
    interpret_error(run_id="diagnostic_smoke_001", run_root=str(run_root))
    select_hypothesis(run_id="diagnostic_smoke_001", run_root=str(run_root))

    result = collect_artifacts(
        run_id="diagnostic_smoke_001",
        run_root=str(run_root),
        config_hash="diagnostic-smoke-001",
    )

    manifest_path = run_root / "artifacts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert result["state_hint"] == "HYPOTHESIS_SELECTED"
    assert manifest["phase"] == "hypothesis_selected"
    assert manifest["summary"]["dominant_spectral_issue"] == "excessive_smoothing"
    assert manifest["summary"]["physics_issue"] == "overdissipation"
    assert manifest["summary"]["dominant_failure_mode"] == "excessive_smoothing"
    assert manifest["summary"]["selected_hypothesis"] == "add_spectral_consistency_loss"
    assert manifest["completeness"]["has_rollout"] is True
    assert manifest["completeness"]["has_spectrum"] is True
    assert manifest["completeness"]["has_physics_check"] is True
    assert manifest["completeness"]["has_data_inspection"] is False
    assert manifest["completeness"]["has_error_interpretation"] is True
    assert manifest["completeness"]["has_selected_hypothesis"] is True


def test_collect_artifacts_includes_data_inspection_when_present(tmp_path: Path) -> None:
    run_root = tmp_path / "diagnostic_smoke_001"
    _prepare_rollout(run_root)
    record_timing(run_id="diagnostic_smoke_001", run_root=str(run_root))
    inspection_path = run_root / "metrics" / "data_inspection.json"
    inspection_path.write_text(
        json.dumps(
            {
                "source_mode": "npz_fixture",
                "is_compatible": True,
                "sample_count": 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    collect_artifacts(
        run_id="diagnostic_smoke_001",
        run_root=str(run_root),
        config_hash="diagnostic-smoke-001",
    )

    manifest_path = run_root / "artifacts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["artifacts"]["data_inspection_path"] == str(inspection_path)
    assert manifest["summary"]["source_mode"] == "npz_fixture"
    assert manifest["summary"]["data_is_compatible"] is True
    assert manifest["completeness"]["has_data_inspection"] is True
