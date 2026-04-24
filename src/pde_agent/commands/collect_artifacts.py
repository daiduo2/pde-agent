import json
from pathlib import Path

from pde_agent.core.json_output import success_payload
from pde_agent.core.manifest_builder import build_manifest


def collect_artifacts(
    run_id: str, run_root: str, config_hash: str = "unknown"
) -> dict[str, object]:
    root = Path(run_root)
    rollout_path = root / "metrics" / "rollout.json"
    timing_path = root / "metrics" / "timing.json"
    rollout_payload = json.loads(rollout_path.read_text(encoding="utf-8"))
    timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
    diagnostic = _load_diagnostic_payloads(root)

    artifacts = _base_artifacts(rollout_path, timing_path, rollout_payload, timing_payload)
    artifacts.update(diagnostic["artifacts"])

    summary = {
        "one_step_nrmse": rollout_payload["one_step"]["nrmse"],
        "long_horizon_nrmse": rollout_payload["long_horizon"]["nrmse"],
        "train_time_seconds": timing_payload["train_time_seconds"],
        "inference_time_seconds": timing_payload["inference_time_seconds"],
    }
    summary.update(diagnostic["summary"])

    completeness = {
        "has_rollout": True,
        "has_spectrum": diagnostic["has_spectrum"],
        "has_physics_check": diagnostic["has_physics_check"],
        "has_error_interpretation": diagnostic["has_error_interpretation"],
        "has_selected_hypothesis": diagnostic["has_selected_hypothesis"],
        "has_timing": True,
    }
    if "has_data_inspection" in diagnostic:
        completeness["has_data_inspection"] = diagnostic["has_data_inspection"]
    phase = (
        "hypothesis_selected"
        if diagnostic["has_selected_hypothesis"]
        else "results_collected"
    )
    state_hint = (
        "HYPOTHESIS_SELECTED"
        if diagnostic["has_selected_hypothesis"]
        else "RESULTS_COLLECTED"
    )

    manifest = build_manifest(
        run_id=run_id,
        config={
            "resolved_config_path": str(root / "config" / "resolved.json"),
            "config_hash": config_hash,
        },
        artifacts=artifacts,
        summary=summary,
        completeness=completeness,
        phase=phase,
    )
    manifest_path = root / "artifacts" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return success_payload(
        run_id=run_id,
        command="collect-artifacts",
        state_hint=state_hint,
        artifacts={"artifact_manifest_path": str(manifest_path), **artifacts},
        metrics={"artifact_count": len(artifacts)},
    )


def _base_artifacts(
    rollout_path: Path,
    timing_path: Path,
    rollout_payload: dict[str, object],
    timing_payload: dict[str, object],
) -> dict[str, object]:
    prediction = rollout_payload["prediction"]
    curve = rollout_payload["curve"]
    if not isinstance(prediction, dict) or not isinstance(curve, dict):
        raise KeyError("rollout prediction and curve must be objects")
    return {
        "rollout_metrics_path": str(rollout_path),
        "timing_metrics_path": str(timing_path),
        "prediction_hdf5_path": prediction["path"],
        "error_curve_path": curve["path"],
        "time_csv_path": timing_payload["time_csv_path"],
    }


def _load_diagnostic_payloads(root: Path) -> dict[str, object]:
    spectrum_path = root / "metrics" / "spectrum.json"
    physics_path = root / "metrics" / "physics_check.json"
    data_inspection_path = root / "metrics" / "data_inspection.json"
    interpretation_path = root / "diagnostics" / "error_interpretation.json"
    hypothesis_path = root / "diagnostics" / "selected_hypothesis.json"

    artifacts: dict[str, object] = {}
    summary: dict[str, object] = {}

    if spectrum_path.exists():
        spectrum = json.loads(spectrum_path.read_text(encoding="utf-8"))
        artifacts["spectrum_metrics_path"] = str(spectrum_path)
        summary["dominant_spectral_issue"] = spectrum["dominant_spectral_issue"]
    if physics_path.exists():
        physics = json.loads(physics_path.read_text(encoding="utf-8"))
        artifacts["physics_check_path"] = str(physics_path)
        summary["physics_issue"] = physics["physics_issue"]
    if data_inspection_path.exists():
        data_inspection = json.loads(data_inspection_path.read_text(encoding="utf-8"))
        artifacts["data_inspection_path"] = str(data_inspection_path)
        summary["source_mode"] = data_inspection["source_mode"]
        summary["data_is_compatible"] = data_inspection["is_compatible"]
    if interpretation_path.exists():
        interpretation = json.loads(interpretation_path.read_text(encoding="utf-8"))
        artifacts["error_interpretation_path"] = str(interpretation_path)
        summary["dominant_failure_mode"] = interpretation["dominant_failure_mode"]
    if hypothesis_path.exists():
        hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
        artifacts["selected_hypothesis_path"] = str(hypothesis_path)
        summary["selected_hypothesis"] = hypothesis["selected_hypothesis"]

    return {
        "artifacts": artifacts,
        "summary": summary,
        "has_spectrum": spectrum_path.exists(),
        "has_physics_check": physics_path.exists(),
        **(
            {"has_data_inspection": True}
            if data_inspection_path.exists()
            else {}
        ),
        "has_error_interpretation": interpretation_path.exists(),
        "has_selected_hypothesis": hypothesis_path.exists(),
    }
