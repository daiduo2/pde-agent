from pathlib import Path

from pde_agent.core.diagnostic_synthesis import read_json, spectrum_payload, write_json
from pde_agent.core.data_config import load_data_config
from pde_agent.core.json_output import success_payload
from pde_agent.core.pdebench_adapter import load_array_bundle
from pde_agent.core.real_metrics import spectrum_metrics


def analyze_spectrum(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
    resolved_config_path = root / "config" / "resolved.json"
    if resolved_config_path.exists():
        data_config = load_data_config(resolved_config_path)
        if data_config.mode != "synthetic":
            payload = spectrum_metrics(load_array_bundle(data_config))
            spectrum_path = root / "metrics" / "spectrum.json"
            write_json(spectrum_path, payload)
            return success_payload(
                run_id=run_id,
                command="analyze-spectrum",
                state_hint="SPECTRUM_ANALYZED",
                artifacts={"spectrum_metrics_path": str(spectrum_path)},
                metrics={
                    "source_mode": payload["source_mode"],
                    "low_frequency_error": payload["low_frequency_error"],
                    "high_frequency_error": payload["high_frequency_error"],
                    "spectral_drift_score": payload["spectral_drift_score"],
                    "dominant_spectral_issue": payload["dominant_spectral_issue"],
                },
            )

    rollout = read_json(root / "metrics" / "rollout.json")
    payload = spectrum_payload(run_id=run_id, rollout=rollout)
    spectrum_path = root / "metrics" / "spectrum.json"
    write_json(spectrum_path, payload)
    return success_payload(
        run_id=run_id,
        command="analyze-spectrum",
        state_hint="SPECTRUM_ANALYZED",
        artifacts={"spectrum_metrics_path": str(spectrum_path)},
        metrics={
            "low_frequency_error": payload["low_frequency_error"],
            "high_frequency_error": payload["high_frequency_error"],
            "spectral_drift_score": payload["spectral_drift_score"],
            "dominant_spectral_issue": payload["dominant_spectral_issue"],
        },
    )
