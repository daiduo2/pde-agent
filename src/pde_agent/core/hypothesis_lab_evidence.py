from __future__ import annotations

import json
from typing import Any

from pde_agent.core.hypothesis_lab_utils import (
    _normalize_retrieval_refinement_params,
)


def _merge_card_list(cards: list[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for card in cards:
        raw = card.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, str) or item in seen:
                continue
            values.append(item)
            seen.add(item)
    return values


def _ranked_evidence_params(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_key: dict[str, dict[str, Any]] = {}
    for card in cards:
        evidence = card.get("evidence")
        if not isinstance(evidence, dict):
            continue
        params = evidence.get("params")
        if not isinstance(params, dict):
            continue
        normalized = _normalize_evidence_params(params)
        if normalized is None:
            continue
        key = json.dumps(params, sort_keys=True)
        if key in by_key:
            _merge_ranked_evidence_row(by_key[key], evidence)
            continue
        row = {
            "params": normalized,
            "source_card_id": card["card_id"],
            "task1_total_fast_proxy": evidence.get("task1_total_fast_proxy"),
            "task1_total_score_proxy": evidence.get("task1_total_score_proxy"),
            "segment_score": evidence.get("segment_score"),
            "source_kind": evidence.get("source_kind"),
            "outcome": evidence.get("outcome"),
            "pareto_status": evidence.get("pareto_status"),
            "effective_axes": evidence.get("effective_axes", []),
            "retry_directives": evidence.get("retry_directives", []),
            "avoid_directives": evidence.get("avoid_directives", []),
            "negative_candidate_evidence": evidence.get(
                "negative_candidate_evidence",
                [],
            ),
            "candidate_pool_attribution": evidence.get(
                "candidate_pool_attribution",
                {},
            ),
            "search_directives": evidence.get("search_directives", []),
            "early_segment_diagnosis": evidence.get("early_segment_diagnosis"),
        }
        by_key[key] = row
        rows.append(row)
    rows.sort(
        key=lambda item: (
            -float(
                item.get("task1_total_fast_proxy")
                or item.get("task1_total_score_proxy")
            )
            if isinstance(
                item.get("task1_total_fast_proxy")
                or item.get("task1_total_score_proxy"),
                int | float,
            )
            else 0.0,
            str(item["source_card_id"]),
        )
    )
    return rows


def _merge_ranked_evidence_row(row: dict[str, Any], evidence: dict[str, Any]) -> None:
    for key in [
        "effective_axes",
        "retry_directives",
        "avoid_directives",
        "negative_candidate_evidence",
        "search_directives",
    ]:
        values = row.setdefault(key, [])
        if not isinstance(values, list):
            values = []
            row[key] = values
        for item in evidence.get(key, []):
            marker = json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item)
            existing = {
                json.dumps(value, sort_keys=True)
                if isinstance(value, dict)
                else str(value)
                for value in values
            }
            if marker not in existing:
                values.append(item)
    if row.get("early_segment_diagnosis") is None and isinstance(
        evidence.get("early_segment_diagnosis"),
        dict,
    ):
        row["early_segment_diagnosis"] = evidence["early_segment_diagnosis"]
    if not row.get("candidate_pool_attribution") and isinstance(
        evidence.get("candidate_pool_attribution"),
        dict,
    ):
        row["candidate_pool_attribution"] = evidence["candidate_pool_attribution"]


def _normalize_evidence_params(params: dict[str, Any]) -> dict[str, Any] | None:
    if {"k", "mode", "early_blend_steps", "persistence_weight_start"}.issubset(
        params
    ):
        return _normalize_retrieval_refinement_params(params)
    if {"slope_window", "gain", "decay", "spectral_strength"}.issubset(params):
        return {
            "slope_window": int(params["slope_window"]),
            "gain": float(params["gain"]),
            "decay": float(params["decay"]),
            "spectral_strength": float(params["spectral_strength"]),
        }
    return None


def _rank_hypotheses(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for hypothesis in hypotheses:
        item = dict(hypothesis)
        family = str(item.get("candidate_family", ""))
        priority = int(item.get("priority", 10))
        novelty_bonus = {
            "official_checkpoint_calibration": 45.0,
            "task1_early_transition_expert": 51.0,
            "task1_early_refinement": 47.0,
            "task1_retrieval_refinement_local_search": 46.0,
            "task1_retrieval_refinement": 44.0,
            "task1_retrieval_prior": 43.0,
            "task1_pi_deeponet_correction_head": 50.0,
            "task1_physics_residual_refinement": 49.0,
            "task1_sem_burgers_rollout": 48.0,
            "learned_residual_correction": 40.0,
            "linear_trend_late_spectral_ramp": 25.0,
            "linear_trend_spectral_escape": 20.0,
            "external_residual_blend": 4.0,
            "linear_trend_spectral": 5.0,
        }.get(family, 0.0)
        executable_bonus = 10.0 if family != "official_checkpoint_calibration" else 2.0
        source_bonus = min(10.0, 2.0 * len(item.get("source_card_ids", [])))
        ranking_score = novelty_bonus + executable_bonus + source_bonus - float(priority)
        item["ranking_score"] = ranking_score
        item["ranking_reasons"] = [
            f"novelty_bonus={novelty_bonus:g}",
            f"executable_bonus={executable_bonus:g}",
            f"source_bonus={source_bonus:g}",
            f"priority={priority}",
        ]
        ranked.append(item)
    ranked.sort(
        key=lambda item: (
            -float(item["ranking_score"]),
            int(item.get("priority", 10)),
            str(item.get("candidate_family", "")),
        )
    )
    return ranked
