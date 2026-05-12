"""Task 2 context-based Nu/domain-shift diagnostics."""


import json
from pathlib import Path
from typing import Any

import numpy as np


STATE_HINT = "TASK2_NU_SHIFT_DIAGNOSED"


class Task2NuShiftDiagnosticError(RuntimeError):
    pass


def diagnose_task2_nu_shift(
    *,
    target_path: str | Path,
    output_path: str | Path,
    reference_path: str | Path | None,
    dataset_key: str,
    nu_key: str,
    context_steps: int,
    sample_limit: int | None,
) -> dict[str, Any]:
    if context_steps < 2:
        raise Task2NuShiftDiagnosticError("context_steps must be at least 2")
    target_file = Path(target_path)
    target = _load_dataset(
        path=target_file,
        dataset_key=dataset_key,
        context_steps=context_steps,
        sample_limit=sample_limit,
    )
    target_nu = _load_optional_nu(path=target_file, nu_key=nu_key, sample_limit=sample_limit)
    target_summary = _summarize_domain(
        path=target_file,
        context=target,
        nu=target_nu,
    )
    reference_summary = None
    domain_shift = None
    if reference_path is not None:
        reference_file = Path(reference_path)
        reference = _load_dataset(
            path=reference_file,
            dataset_key=dataset_key,
            context_steps=context_steps,
            sample_limit=sample_limit,
        )
        reference_nu = _load_optional_nu(
            path=reference_file,
            nu_key=nu_key,
            sample_limit=sample_limit,
        )
        reference_summary = _summarize_domain(
            path=reference_file,
            context=reference,
            nu=reference_nu,
        )
        domain_shift = _domain_shift(
            target=target_summary,
            reference=reference_summary,
        )
    report = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "diagnostic_path": str(output_path),
        "target": target_summary,
        "reference": reference_summary,
        "domain_shift": domain_shift
        or {
            "reference_path": None,
            "shift_score": None,
            "feature_deltas": {},
        },
        "recommendation": {
            "strategy_id": "nu_inference_context_classifier_v0",
            "reason_codes": _reason_codes(
                target=target_summary,
                domain_shift=domain_shift,
            ),
            "next_command": "lookup-local-strategy",
            "failure_mode": "task2_domain_shift",
        },
        "next_command": "lookup-local-strategy",
    }
    _write_json(Path(output_path), report)
    return report


def _load_dataset(
    *,
    path: Path,
    dataset_key: str,
    context_steps: int,
    sample_limit: int | None,
) -> np.ndarray:
    if not path.is_file():
        raise Task2NuShiftDiagnosticError(f"missing HDF5 path: {path}")
    h5py = __import__("h5py")
    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise Task2NuShiftDiagnosticError(
                f"dataset {dataset_key!r} not found in {path}"
            )
        data = np.asarray(
            handle[dataset_key][
                : sample_limit if sample_limit is not None else None,
                :context_steps,
                :,
            ],
            dtype=np.float64,
        )
    if data.ndim != 3:
        raise Task2NuShiftDiagnosticError(f"expected rank-3 tensor, got {data.shape}")
    if data.shape[1] < context_steps:
        raise Task2NuShiftDiagnosticError(
            f"expected at least {context_steps} context steps, got {data.shape[1]}"
        )
    if data.shape[2] == 1024:
        data = data[:, :, ::4]
    if data.shape[2] != 256:
        raise Task2NuShiftDiagnosticError(
            f"expected spatial width 256 or 1024, got {data.shape[2]}"
        )
    if not np.isfinite(data).all():
        raise Task2NuShiftDiagnosticError("context tensor contains NaN or Inf")
    return data


def _load_optional_nu(
    *,
    path: Path,
    nu_key: str,
    sample_limit: int | None,
) -> np.ndarray | None:
    h5py = __import__("h5py")
    with h5py.File(path, "r") as handle:
        if nu_key not in handle:
            return None
        values = np.asarray(
            handle[nu_key][: sample_limit if sample_limit is not None else None],
            dtype=np.float64,
        )
    if values.ndim != 1:
        values = values.reshape(-1)
    return values


def _summarize_domain(
    *,
    path: Path,
    context: np.ndarray,
    nu: np.ndarray | None,
) -> dict[str, Any]:
    signature = _context_signature(context)
    return {
        "path": str(path),
        "sample_count": int(context.shape[0]),
        "context_shape": list(context.shape),
        "nu_available": nu is not None,
        "nu_stats": _nu_stats(nu),
        "context_signature": signature,
    }


def _context_signature(context: np.ndarray) -> dict[str, float]:
    start_energy = np.mean(context[:, 0, :] ** 2, axis=1)
    end_energy = np.mean(context[:, -1, :] ** 2, axis=1)
    decay = (start_energy - end_energy) / (np.abs(start_energy) + 1e-12)
    temporal_delta = np.mean(np.diff(context, axis=1) ** 2, axis=(1, 2))
    gradient = np.diff(context, axis=2)
    gradient_energy = np.mean(gradient**2, axis=(1, 2))
    total_variation = np.mean(np.abs(gradient), axis=(1, 2))
    return {
        "energy_start_mean": float(np.mean(start_energy)),
        "energy_end_mean": float(np.mean(end_energy)),
        "energy_decay_ratio_mean": float(np.mean(decay)),
        "energy_decay_ratio_std": float(np.std(decay)),
        "temporal_delta_energy_mean": float(np.mean(temporal_delta)),
        "gradient_energy_mean": float(np.mean(gradient_energy)),
        "total_variation_mean": float(np.mean(total_variation)),
    }


def _nu_stats(nu: np.ndarray | None) -> dict[str, float] | None:
    if nu is None or nu.size == 0:
        return None
    return {
        "min": float(np.min(nu)),
        "max": float(np.max(nu)),
        "mean": float(np.mean(nu)),
        "std": float(np.std(nu)),
    }


def _domain_shift(
    *,
    target: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    target_sig = target["context_signature"]
    ref_sig = reference["context_signature"]
    deltas = {
        key: float(target_sig[key] - ref_sig[key])
        for key in target_sig
        if key in ref_sig
    }
    return {
        "reference_path": reference["path"],
        "shift_score": float(sum(abs(value) for value in deltas.values()) / len(deltas)),
        "feature_deltas": deltas,
    }


def _reason_codes(
    *,
    target: dict[str, Any],
    domain_shift: dict[str, Any] | None,
) -> list[str]:
    reasons = ["task2_hidden_nu_requires_context_inference"]
    if target["nu_available"]:
        reasons.append("validation_nu_available_for_diagnostic")
    else:
        reasons.append("target_nu_hidden")
    if domain_shift is not None and float(domain_shift["shift_score"]) > 0.01:
        reasons.append("context_feature_shift_detected")
    return reasons


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
