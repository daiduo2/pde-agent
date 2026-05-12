import json
import importlib.util
from pathlib import Path

import numpy as np

from pde_agent.commands.analyze_spectrum import analyze_spectrum
from pde_agent.commands.check_physics import check_physics
from pde_agent.commands.collect_artifacts import collect_artifacts
from pde_agent.commands.eval_rollout import eval_rollout
from pde_agent.commands.inspect_data import inspect_data
from pde_agent.commands.interpret_error import interpret_error
from pde_agent.commands.record_timing import record_timing
from pde_agent.commands.select_hypothesis import select_hypothesis


def test_prepare_array_fixture_smoke_renders_portable_runtime_files() -> None:
    repo_root = Path("/tmp/example-checkout")
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "smoke"
        / "prepare_array_fixture_smoke.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_array_fixture_smoke", script_path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    base_config = module.build_base_config(repo_root)
    run_state = module.render_run_state(repo_root)

    assert base_config["data"]["source_path"] == (
        "/tmp/example-checkout/integration_tests/array_fixture_smoke/fixture.npz"
    )
    assert (
        "command_prefix: /tmp/example-checkout/.venv/bin/pde-agent"
        in run_state
    )
    assert "{{REPO_ROOT}}" not in run_state
    assert ("/Users/" + "daiduo2") not in run_state


def test_array_fixture_reaches_hypothesis_selected_manifest(tmp_path: Path) -> None:
    source_path = tmp_path / "fixture.npz"
    target = np.ones((2, 5, 16), dtype=np.float32)
    prediction = target * 0.85
    np.savez(
        source_path,
        context=np.zeros((2, 2, 16), dtype=np.float32),
        prediction=prediction,
        target=target,
    )

    run_id = "array_fixture_flow_001"
    run_root = tmp_path / "runs" / run_id
    config_path = run_root / "config" / "resolved.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "mode": "npz_fixture",
                    "problem_id": "PDE_Burgers",
                    "source_path": str(source_path),
                    "context_steps": 2,
                    "forecast_steps": 5,
                    "time_reduction": 5,
                    "space_reduction": 4,
                }
            }
        ),
        encoding="utf-8",
    )

    inspect_data(run_id=run_id, run_root=str(run_root))
    eval_rollout(run_id=run_id, run_root=str(run_root))
    analyze_spectrum(run_id=run_id, run_root=str(run_root))
    check_physics(run_id=run_id, run_root=str(run_root))
    interpret_error(run_id=run_id, run_root=str(run_root))
    select_hypothesis(run_id=run_id, run_root=str(run_root))
    record_timing(run_id=run_id, run_root=str(run_root))
    collect_artifacts(
        run_id=run_id,
        run_root=str(run_root),
        config_hash="array-fixture-flow-001",
    )

    manifest = json.loads(
        (run_root / "artifacts" / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["phase"] == "hypothesis_selected"
    assert manifest["summary"]["source_mode"] == "npz_fixture"
    assert manifest["summary"]["data_is_compatible"] is True
    assert manifest["completeness"]["has_data_inspection"] is True
    assert manifest["completeness"]["has_selected_hypothesis"] is True
