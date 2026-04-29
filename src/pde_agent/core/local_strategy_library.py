"""Built-in operator strategy library for Hermes local search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LIBRARY_STATE_HINT = "LOCAL_STRATEGY_LIBRARY_BUILT"
LOOKUP_STATE_HINT = "LOCAL_STRATEGY_LOOKUP_COMPLETED"


class LocalStrategyLibraryError(RuntimeError):
    pass


def build_local_strategy_library(
    *,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    output_file = Path(output_path)
    if output_file.exists() and not force:
        raise LocalStrategyLibraryError(
            f"local strategy library already exists: {output_file}"
        )
    entries = _builtin_entries()
    library = {
        "schema_version": 1,
        "state_hint": LIBRARY_STATE_HINT,
        "library_path": str(output_file),
        "source": "hermes_local_operator_strategy_skills",
        "entry_count": len(entries),
        "entries": entries,
        "next_command": "lookup-local-strategy",
    }
    _write_json(output_file, library)
    return library


def lookup_local_strategies(
    *,
    library_path: str | Path,
    applicable_pde: str,
    model_family: str,
    failure_modes: list[str],
    max_results: int,
    output_path: str | Path | None,
) -> dict[str, Any]:
    if max_results < 1:
        raise LocalStrategyLibraryError("max_results must be positive")
    library_file = Path(library_path)
    library = _read_library(library_file)
    matches = []
    for entry in library["entries"]:
        score = _selection_score(
            entry=entry,
            applicable_pde=applicable_pde,
            model_family=model_family,
            failure_modes=failure_modes,
        )
        if score <= 0:
            continue
        row = dict(entry)
        row["selection_score"] = score
        matches.append(row)
    matches.sort(
        key=lambda item: (
            -float(item["selection_score"]),
            int(item.get("priority", 999)),
            str(item.get("strategy_id", "")),
        )
    )
    selected = matches[:max_results]
    result = {
        "schema_version": 1,
        "state_hint": LOOKUP_STATE_HINT,
        "library_path": str(library_file),
        "lookup": {
            "applicable_pde": applicable_pde,
            "model_family": model_family,
            "failure_modes": failure_modes,
            "max_results": max_results,
        },
        "match_count": len(selected),
        "matches": selected,
        "next_command": "schedule-local-strategy-evaluation",
    }
    if output_path is not None:
        output_file = Path(output_path)
        result["lookup_report_path"] = str(output_file)
        _write_json(output_file, result)
    return result


def _selection_score(
    *,
    entry: dict[str, Any],
    applicable_pde: str,
    model_family: str,
    failure_modes: list[str],
) -> int:
    score = 0
    model_families = entry.get("model_families", [])
    if model_family in model_families:
        score += 5
    if model_families and model_families[0] == model_family:
        score += 2
    if applicable_pde in entry.get("applicable_pdes", []):
        score += 4
    entry_modes = set(entry.get("targeted_failure_modes", []))
    requested_modes = set(failure_modes)
    mode_overlap = entry_modes & requested_modes
    score += 3 * len(mode_overlap)
    if requested_modes and not mode_overlap:
        score -= 3
    if "all" in entry.get("model_families", []):
        score += 1
    if (
        "1D Burgers" in entry.get("applicable_pdes", [])
        and applicable_pde == "Burgers"
    ):
        score += 2
    return score


def _read_library(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LocalStrategyLibraryError(f"local strategy library is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LocalStrategyLibraryError(
            f"invalid local strategy library JSON: {path}"
        ) from exc
    if data.get("state_hint") != LIBRARY_STATE_HINT:
        raise LocalStrategyLibraryError("local strategy library has invalid state_hint")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise LocalStrategyLibraryError("local strategy library entries must be a list")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _entry(
    *,
    strategy_id: str,
    title: str,
    model_families: list[str],
    applicable_pdes: list[str],
    targeted_failure_modes: list[str],
    hypothesis: str,
    mechanism: str,
    implementation_hooks: list[str],
    expected_observables: list[str],
    known_risks: list[str],
    acceptance_criteria: list[str],
    priority: int,
) -> dict[str, Any]:
    return {
        "strategy_id": strategy_id,
        "title": title,
        "source_type": "local_canonical_skill",
        "maturity": "seeded",
        "priority": priority,
        "model_families": model_families,
        "applicable_pdes": applicable_pdes,
        "targeted_failure_modes": targeted_failure_modes,
        "hypothesis": hypothesis,
        "mechanism": mechanism,
        "implementation_hooks": implementation_hooks,
        "expected_observables": expected_observables,
        "known_risks": known_risks,
        "evaluation_protocol": {
            "primary_metric": "competition_segment_score",
            "secondary_metrics": [
                "long_horizon_rmse",
                "physics_residual",
                "inference_time_seconds",
            ],
            "minimum_evidence": [
                "real_hdf5_smoke",
                "baseline_comparison",
                "long_window_rollout",
            ],
        },
        "acceptance_criteria": acceptance_criteria,
    }


def _builtin_entries() -> list[dict[str, Any]]:
    return [
        _entry(
            strategy_id="fno_spectral_viscosity_filter_v0",
            title="FNO spectral viscosity filter for stable Burgers rollouts",
            model_families=["FNO"],
            applicable_pdes=["1D Burgers", "Burgers"],
            targeted_failure_modes=[
                "long_horizon_drift",
                "spectral_dissipation_mismatch",
                "shock_oscillation",
            ],
            hypothesis=(
                "High-frequency FNO rollout error can be reduced by applying a "
                "light spectral viscosity correction during autoregressive inference."
            ),
            mechanism=(
                "Estimate unstable high-frequency energy growth and damp modes above "
                "a configured cutoff without changing the first 10 context steps."
            ),
            implementation_hooks=[
                "predict-fno-competition",
                "run-burgers-spectral-viscosity-benchmark",
                "score-competition-submission",
            ],
            expected_observables=[
                "lower late-window RMSE",
                "reduced high-frequency residual energy",
                "no violation of initial-condition preservation",
            ],
            known_risks=[
                "over-damping shocks can hurt short-window accuracy",
                "filter parameters may overfit Task 1 viscosity",
            ],
            acceptance_criteria=[
                "late segment score improves over unfiltered FNO",
                "Task inference remains under 120 seconds",
                "submission audit passes initial-condition tolerance",
            ],
            priority=10,
        ),
        _entry(
            strategy_id="fno_rollout_consistency_finetune_v0",
            title="FNO rollout consistency fine-tuning",
            model_families=["FNO"],
            applicable_pdes=["1D Burgers", "Burgers"],
            targeted_failure_modes=["exposure_bias", "long_horizon_drift"],
            hypothesis=(
                "Training on short autoregressive unrolls reduces the train-test "
                "mismatch created by one-step teacher forcing."
            ),
            mechanism=(
                "Fine-tune the checkpoint with multi-step loss, preserving official "
                "resolution reductions and checkpoint compatibility."
            ),
            implementation_hooks=[
                "execute-fno-finetune-candidate",
                "evaluate-fno-candidate",
                "run-fno-evidence-sweep",
            ],
            expected_observables=[
                "lower mid-window relative MSE",
                "smaller growth of rollout error slope",
            ],
            known_risks=["training time can harm Task 1 time score"],
            acceptance_criteria=[
                "short and mid segment scores both improve",
                "train_time stays within selected competition budget",
            ],
            priority=20,
        ),
        _entry(
            strategy_id="fno_mode_schedule_antialias_v0",
            title="FNO mode schedule and anti-aliasing sweep",
            model_families=["FNO"],
            applicable_pdes=["1D Burgers", "Burgers"],
            targeted_failure_modes=["spectral_aliasing", "resolution_mismatch"],
            hypothesis=(
                "Mode-count and anti-aliasing choices explain part of the gap between "
                "short-window accuracy and long-window stability."
            ),
            mechanism="Sweep retained Fourier modes and optional dealiasing filters.",
            implementation_hooks=[
                "inspect-fno-checkpoint",
                "run-fno-iteration-dryrun",
                "compare-fno-candidates",
            ],
            expected_observables=["mode-energy report", "candidate score ordering"],
            known_risks=["checkpoint architecture can make mode changes non-loadable"],
            acceptance_criteria=[
                "candidate remains checkpoint-compatible",
                "diagnostic evidence selects a bounded mode/filter range",
            ],
            priority=30,
        ),
        _entry(
            strategy_id="unet_autoregressive_residual_correction_v0",
            title="UNet autoregressive residual correction",
            model_families=["UNet"],
            applicable_pdes=["1D Burgers", "Burgers"],
            targeted_failure_modes=["autoregressive_blur", "long_horizon_drift"],
            hypothesis=(
                "UNet rollouts lose sharp Burgers structures; a residual correction "
                "head can restore local temporal increments."
            ),
            mechanism=(
                "Predict correction deltas on top of the official UNet output and "
                "validate on real long-window samples."
            ),
            implementation_hooks=[
                "predict-unet-competition",
                "execute-unet-candidate",
                "evaluate-unet-candidate",
            ],
            expected_observables=[
                "lower blur proxy",
                "improved late-window RMSE",
                "bounded inference overhead",
            ],
            known_risks=["full UNet inference may exceed the 2 minute limit"],
            acceptance_criteria=[
                "candidate beats persistence and raw UNet on local windows",
                "competition inference estimate is under 120 seconds",
            ],
            priority=10,
        ),
        _entry(
            strategy_id="unet_multiscale_smoothing_stabilizer_v0",
            title="UNet multiscale smoothing stabilizer",
            model_families=["UNet"],
            applicable_pdes=["1D Burgers", "Burgers"],
            targeted_failure_modes=["shock_oscillation", "autoregressive_blur"],
            hypothesis=(
                "A conservative multiscale smoother can reduce unstable UNet "
                "oscillations while preserving large-scale Burgers transport."
            ),
            mechanism="Blend local and coarse-grid corrections during rollout.",
            implementation_hooks=[
                "diagnose-unet-rollout",
                "run-unet-strategy-sweep",
                "build-unet-evidence-sweep",
            ],
            expected_observables=["reduced residual spikes", "stable late-window score"],
            known_risks=["smoothing can remove physically meaningful shocks"],
            acceptance_criteria=[
                "physics residual does not worsen",
                "late-window score improves without short-window collapse",
            ],
            priority=20,
        ),
        _entry(
            strategy_id="deeponet_branch_trunk_normalization_v0",
            title="DeepONet branch/trunk normalization for Burgers windows",
            model_families=["DeepONet"],
            applicable_pdes=["1D Burgers", "Burgers"],
            targeted_failure_modes=["underfit_operator_map", "coordinate_scale_mismatch"],
            hypothesis=(
                "DeepONet underperforms persistence when branch inputs and trunk "
                "coordinates are not normalized for the reduced Burgers grid."
            ),
            mechanism="Normalize context fields and trunk coordinates before training.",
            implementation_hooks=[
                "execute-deeponet-candidate",
                "evaluate-deeponet-candidate",
                "run-deeponet-strategy-sweep",
            ],
            expected_observables=["reduced validation loss", "beats persistence smoke"],
            known_risks=["small sweeps can mis-rank DeepONet capacity"],
            acceptance_criteria=[
                "beats persistence on real-HDF5 smoke",
                "operator evidence sweep records positive delta",
            ],
            priority=10,
        ),
        _entry(
            strategy_id="deeponet_physics_residual_spectral_loss_v0",
            title="DeepONet spectral Burgers residual loss",
            model_families=["DeepONet", "PI-DeepONet"],
            applicable_pdes=["1D Burgers", "Burgers"],
            targeted_failure_modes=["physics_residual_growth", "underfit_operator_map"],
            hypothesis=(
                "Adding a Burgers residual term in spectral derivative space can "
                "improve physical consistency for operator rollouts."
            ),
            mechanism="Combine supervised rollout loss with Burgers residual loss.",
            implementation_hooks=[
                "evaluate-burgers-residual",
                "run-deeponet-strategy-sweep",
                "convert-operator-evidence-summary",
            ],
            expected_observables=["lower residual", "improved long-window stability"],
            known_risks=["residual loss can slow or destabilize small training runs"],
            acceptance_criteria=[
                "physics residual improves over supervised-only candidate",
                "segment score does not regress beyond tolerance",
            ],
            priority=20,
        ),
        _entry(
            strategy_id="pi_deeponet_residual_weight_schedule_v0",
            title="PI-DeepONet residual weight schedule",
            model_families=["PI-DeepONet"],
            applicable_pdes=["1D Burgers", "Burgers"],
            targeted_failure_modes=["loss_term_imbalance", "physics_residual_growth"],
            hypothesis=(
                "A staged physics-weight schedule avoids early optimization collapse "
                "while preserving final residual constraints."
            ),
            mechanism="Warm up supervised loss before increasing residual weight.",
            implementation_hooks=[
                "evaluate-burgers-residual",
                "select-pde-hypothesis",
                "generate-experiment-config",
            ],
            expected_observables=["stable loss curves", "lower residual after warmup"],
            known_risks=["longer training can be expensive"],
            acceptance_criteria=[
                "residual improves on held-out windows",
                "supervised score remains competitive with non-PI baseline",
            ],
            priority=10,
        ),
        _entry(
            strategy_id="burgers_shock_aware_smoothness_weighting_v0",
            title="Shock-aware Burgers smoothness weighting",
            model_families=["FNO", "UNet", "DeepONet", "PI-DeepONet"],
            applicable_pdes=["1D Burgers", "Burgers"],
            targeted_failure_modes=["shock_oscillation", "over_smoothing"],
            hypothesis=(
                "Smoothness penalties should be weaker near shocks and stronger in "
                "smooth regions to avoid removing real discontinuity-like structure."
            ),
            mechanism="Gate smoothness or residual terms by spatial gradient magnitude.",
            implementation_hooks=[
                "check-burgers-physics-local",
                "diagnose-rollout-error",
                "evaluate-burgers-residual",
            ],
            expected_observables=["lower oscillation proxy", "preserved shock amplitude"],
            known_risks=["gradient gates can amplify noise"],
            acceptance_criteria=[
                "shock-region error does not regress",
                "smooth-region residual improves",
            ],
            priority=40,
        ),
        _entry(
            strategy_id="nu_inference_context_classifier_v0",
            title="Task 2 viscosity inference from context",
            model_families=["DataInference", "FNO", "UNet"],
            applicable_pdes=["1D Burgers", "Burgers"],
            targeted_failure_modes=["unknown_nu_generalization", "task2_domain_shift"],
            hypothesis=(
                "The first 10 context frames contain enough dissipation signal to "
                "infer a latent viscosity bucket for Task 2 routing."
            ),
            mechanism="Classify or regress viscosity bucket from context statistics.",
            implementation_hooks=[
                "inspect-data",
                "prepare-competition-windows",
                "materialize-competition-candidate",
            ],
            expected_observables=["separable context statistics", "Task 2 score lift"],
            known_risks=["test-time Nu is hidden and may not match train buckets"],
            acceptance_criteria=[
                "bucket predictor improves validation routing",
                "inference-time overhead is negligible",
            ],
            priority=10,
        ),
        _entry(
            strategy_id="ensemble_submission_selector_v0",
            title="Evidence-gated submission ensemble selector",
            model_families=["DataInference", "FNO", "UNet", "DeepONet"],
            applicable_pdes=["1D Burgers", "Burgers"],
            targeted_failure_modes=["candidate_selection_uncertainty", "task2_domain_shift"],
            hypothesis=(
                "A local validation leaderboard can select or blend candidates by "
                "window regime instead of relying on one globally best model."
            ),
            mechanism="Rank candidates by segment, physics residual, and inference time.",
            implementation_hooks=[
                "score-competition-submission",
                "audit-competition-submission",
                "build-research-chronicle",
            ],
            expected_observables=["segment-wise candidate ordering", "auditable decision"],
            known_risks=["validation distribution may differ from hidden test data"],
            acceptance_criteria=[
                "selected artifact has complete logs and timing",
                "leaderboard decision is reproducible from local evidence",
            ],
            priority=30,
        ),
    ]
