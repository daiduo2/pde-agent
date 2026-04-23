def build_manifest(
    run_id: str,
    config: dict[str, object],
    artifacts: dict[str, object],
    summary: dict[str, object],
    completeness: dict[str, object],
    phase: str = "results_collected",
) -> dict[str, object]:
    return {
        "status": "ok",
        "run_id": run_id,
        "phase": phase,
        "config": config,
        "artifacts": artifacts,
        "summary": summary,
        "completeness": completeness,
        "errors": [],
    }
