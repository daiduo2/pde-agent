import json
from pathlib import Path

import numpy as np


def _write_hdf5(path: Path, tensor: np.ndarray, nu: np.ndarray | None = None) -> Path:
    h5py = __import__("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=tensor.astype("float32"))
        if nu is not None:
            handle.create_dataset("nu", data=nu.astype("float32"))
    return path


def test_diagnose_task2_nu_shift_reports_context_features_and_nu_stats(
    tmp_path: Path,
) -> None:
    from pde_agent.core.task2_nu_shift_diagnostic import diagnose_task2_nu_shift

    x = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    base = np.sin(x)[None, None, :]
    time = np.linspace(1.0, 0.8, 10)[None, :, None]
    tensor = np.repeat(base * time, 4, axis=0)
    nu = np.array([0.001, 0.002, 0.003, 0.004], dtype=np.float32)
    target_path = _write_hdf5(tmp_path / "task2_val.h5", tensor, nu=nu)

    report = diagnose_task2_nu_shift(
        target_path=target_path,
        output_path=tmp_path / "diagnostic.json",
        reference_path=None,
        dataset_key="tensor",
        nu_key="nu",
        context_steps=10,
        sample_limit=None,
    )

    assert report["state_hint"] == "TASK2_NU_SHIFT_DIAGNOSED"
    assert report["target"]["sample_count"] == 4
    assert report["target"]["nu_available"] is True
    assert report["target"]["nu_stats"]["min"] == 0.0010000000474974513
    assert report["target"]["context_signature"]["energy_decay_ratio_mean"] > 0.0
    assert report["recommendation"]["strategy_id"] == (
        "nu_inference_context_classifier_v0"
    )
    assert Path(report["diagnostic_path"]).exists()


def test_diagnose_task2_nu_shift_compares_reference_domain(tmp_path: Path) -> None:
    from pde_agent.core.task2_nu_shift_diagnostic import diagnose_task2_nu_shift

    reference = np.ones((3, 10, 256), dtype=np.float32)
    target = np.ones((3, 10, 256), dtype=np.float32)
    target[:, -1, :] *= 0.2
    reference_path = _write_hdf5(tmp_path / "reference.h5", reference)
    target_path = _write_hdf5(tmp_path / "target.h5", target)

    report = diagnose_task2_nu_shift(
        target_path=target_path,
        output_path=tmp_path / "diagnostic.json",
        reference_path=reference_path,
        dataset_key="tensor",
        nu_key="nu",
        context_steps=10,
        sample_limit=3,
    )

    assert report["domain_shift"]["reference_path"] == str(reference_path)
    assert report["domain_shift"]["shift_score"] > 0.0
    assert "energy_decay_ratio_mean" in report["domain_shift"]["feature_deltas"]
    assert report["recommendation"]["next_command"] == "lookup-local-strategy"


def test_diagnose_task2_nu_shift_cli_returns_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    tensor = np.ones((2, 10, 256), dtype=np.float32)
    target_path = _write_hdf5(tmp_path / "task2_test.h5", tensor)
    output_path = tmp_path / "diagnostic.json"
    result = CliRunner().invoke(
        app,
        [
            "diagnose-task2-nu-shift",
            "--target-path",
            str(target_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["state_hint"] == "TASK2_NU_SHIFT_DIAGNOSED"
    assert payload["metrics"]["sample_count"] == 2
    assert payload["next_hint"] == "lookup-local-strategy"
