
from pathlib import Path
import zipfile

from pde_agent.core.competition_submission import (
    STATE_HINT,
    build_competition_submission,
)
from pde_agent.core.json_output import success_payload


def build_competition_submission_command(
    *,
    task_id: int,
    test_path: str,
    output_dir: str,
    submission_id: str,
    dataset_key: str,
    prediction_dataset_key: str,
    method: str,
    source_prediction_path: str | None,
    source_prediction_key: str,
    train_time: float,
    inference_time: float | None,
    timing_profile: str | None = None,
    timing_factor: float | None = None,
    force: bool = False,
    local_loop_chronicle_path: str | None = None,
) -> dict[str, object]:
    report = build_competition_submission(
        task_id=task_id,
        test_path=Path(test_path),
        output_dir=Path(output_dir),
        submission_id=submission_id,
        dataset_key=dataset_key,
        prediction_dataset_key=prediction_dataset_key,
        method=method,
        source_prediction_path=source_prediction_path,
        source_prediction_key=source_prediction_key,
        train_time=train_time,
        inference_time=inference_time,
        timing_profile=timing_profile,
        timing_factor=timing_factor,
        force=force,
    )
    if local_loop_chronicle_path is not None:
        _append_local_loop_chronicle(
            log_path=Path(report["artifacts"]["log_path"]),
            chronicle_path=Path(local_loop_chronicle_path),
            submission_dir=Path(report["artifacts"]["submission_dir"]),
            zip_path=Path(report["artifacts"]["submission_zip_path"]),
        )
        report["metrics"]["local_loop_chronicle_appended"] = True
    else:
        report["metrics"]["local_loop_chronicle_appended"] = False
    return success_payload(
        run_id="local",
        command="build-competition-submission",
        state_hint=STATE_HINT,
        artifacts=report["artifacts"],
        metrics=report["metrics"],
        next_hint=str(report["next_command"]),
    )


def _append_local_loop_chronicle(
    *,
    log_path: Path,
    chronicle_path: Path,
    submission_dir: Path,
    zip_path: Path,
) -> None:
    if not chronicle_path.is_file():
        raise FileNotFoundError(f"local-loop chronicle not found: {chronicle_path}")
    chronicle = chronicle_path.read_text(encoding="utf-8").strip()
    original = log_path.read_text(encoding="utf-8")
    marker = "\nHermes Research Chronicle\n"
    if marker in original:
        original = original.split(marker, maxsplit=1)[0].rstrip() + "\n"
    log_path.write_text(original.rstrip() + "\n\n" + chronicle + "\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(submission_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(submission_dir.parent))
