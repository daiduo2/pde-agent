from pathlib import Path

from pde_agent.core.run_layout import create_run_tree


def test_create_run_tree_builds_expected_subdirs(tmp_path: Path) -> None:
    run_root = create_run_tree(tmp_path, "run_001")
    for subdir in [
        "config",
        "logs",
        "checkpoints",
        "metrics",
        "plots",
        "artifacts",
        "submission",
    ]:
        assert (run_root / subdir).exists()


def test_create_run_tree_creates_request_json(tmp_path: Path) -> None:
    run_root = create_run_tree(tmp_path, "run_001")
    assert (run_root / "config" / "request.json").exists()
