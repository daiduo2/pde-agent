from pde_agent.core.json_output import success_payload


def build_submission(run_id: str, run_root: str) -> dict[str, object]:
    return success_payload(
        run_id=run_id,
        command="build-submission",
        state_hint="SUBMISSION_READY",
        artifacts={
            "submit_hdf5_path": f"{run_root}/submission/result.hdf5",
            "submission_json_path": f"{run_root}/submission/submission.json",
            "time_csv_path": f"{run_root}/submission/time.csv",
        },
        metrics={},
    )
