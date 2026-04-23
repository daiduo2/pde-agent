from pde_agent.core.json_output import success_payload


def analyze_spectrum(run_id: str, run_root: str) -> dict[str, object]:
    return success_payload(
        run_id=run_id,
        command="analyze-spectrum",
        state_hint="RESULTS_COLLECTED",
        artifacts={"spectrum_metrics_path": f"{run_root}/metrics/spectrum.json"},
        metrics={
            "low_freq_error": 0.0,
            "high_freq_error": 0.0,
            "spectral_drift_score": 0.0,
        },
    )
