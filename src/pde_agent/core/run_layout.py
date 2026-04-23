from pathlib import Path


RUN_SUBDIRS = (
    "config",
    "logs",
    "checkpoints",
    "metrics",
    "plots",
    "artifacts",
    "submission",
)


def create_run_tree(runs_root: Path, run_id: str) -> Path:
    run_root = runs_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    for subdir in RUN_SUBDIRS:
        (run_root / subdir).mkdir(exist_ok=True)

    (run_root / "config" / "request.json").write_text("{}\n", encoding="utf-8")
    return run_root
