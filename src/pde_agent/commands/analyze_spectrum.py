from pathlib import Path

from pde_agent.core.diagnostic_synthesis import read_json, spectrum_payload, write_json
from pde_agent.core.json_output import success_payload


def analyze_spectrum(run_id: str, run_root: str) -> dict[str, object]:
    root = Path(run_root)
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
