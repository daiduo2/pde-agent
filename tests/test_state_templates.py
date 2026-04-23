from pathlib import Path


def test_core_state_templates_exist() -> None:
    root = Path("/Users/daiduo2/pde-agent/src/pde_agent/templates/state")
    assert (root / "task_parsed.md").exists()
    assert (root / "error_attributed.md").exists()
    assert (root / "hypothesis_selected.md").exists()
    assert (root / "experiment_interpreted.md").exists()
