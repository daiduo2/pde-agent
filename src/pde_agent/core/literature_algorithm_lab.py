"""Offline artifact backbone for PDE literature-to-algorithm candidates."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


SOURCE_RECORDS_STATE_HINT = "LITERATURE_SOURCE_RECORDS_BUILT"
SOURCE_SEARCH_STATE_HINT = "EXTERNAL_SOURCE_SEARCH_NORMALIZED"
SOURCE_DOCUMENT_STATE_HINT = "SOURCE_DOCUMENT_DOWNLOADED"
MINERU_TASK_SUBMITTED_STATE_HINT = "MINERU_PARSE_TASK_SUBMITTED"
MINERU_TASK_POLLED_STATE_HINT = "MINERU_PARSE_TASK_POLLED"
MINERU_RESULT_DOWNLOADED_STATE_HINT = "MINERU_PARSE_RESULT_DOWNLOADED"
PAPER_PARSE_STATE_HINT = "MINERU_PARSE_INGESTED"
ALGORITHM_CARDS_STATE_HINT = "ALGORITHM_CARDS_EXTRACTED"
ALGORITHM_CARDS_V2_VALIDATED_STATE_HINT = "ALGORITHM_CARDS_V2_VALIDATED"
ALGORITHM_CANDIDATES_STATE_HINT = "ALGORITHM_CANDIDATES_COMPILED"
ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT = "ALGORITHM_CANDIDATE_PROMOTION_EVALUATED"
LITERATURE_REGISTRY_EVENT_STATE_HINT = "LITERATURE_REGISTRY_EVENT_RECORDED"
EXECUTABLE_BLUEPRINTS_VALIDATED_STATE_HINT = "EXECUTABLE_BLUEPRINTS_VALIDATED"
SANDBOX_PLAN_STATE_HINT = "ALGORITHM_CANDIDATE_SANDBOX_PLANNED"
EXECUTOR_TAXONOMY_STATE_HINT = "EXECUTOR_TAXONOMY_BUILT"
FNO_CANDIDATE_ADAPTED_STATE_HINT = "FNO_ARCHITECTURE_CANDIDATE_ADAPTED"
EXECUTOR_DESIGN_REQUEST_STATE_HINT = "EXECUTOR_DESIGN_REQUEST_CREATED"
BLOCKED_CANDIDATE_HANDOFF_STATE_HINT = "BLOCKED_CANDIDATE_HANDOFF_BUILT"
LITERATURE_REGISTRY_UPDATED_STATE_HINT = "LITERATURE_REGISTRY_UPDATED"
LITERATURE_ALGORITHM_V2_PIPELINE_STATE_HINT = "LITERATURE_ALGORITHM_V2_PIPELINE_RAN"

VALID_SOURCE_TYPES = {
    "openalex_work",
    "arxiv_paper",
    "github_repo",
    "competition_resource",
    "manual_source",
}

MINERU_AGENT_URL_ENDPOINT = "https://mineru.net/api/v1/agent/parse/url"
MINERU_AGENT_POLL_ENDPOINT = "https://mineru.net/api/v1/agent/parse"
MINERU_PRECISE_URL_ENDPOINT = "https://mineru.net/api/v4/extract/task"

HOOK_TO_FAMILY = {
    "official_checkpoint_calibration": {
        "candidate_family": "official_checkpoint_family",
        "model_family": "checkpoint",
        "cli_command": "run-checkpoint-family-sandbox",
        "sandbox_level": 1,
    },
    "learned_residual_correction": {
        "candidate_family": "lightweight_residual_corrector",
        "model_family": "postprocess",
        "cli_command": "run-blueprint-sandbox",
        "sandbox_level": 0,
    },
    "spectral_rollout_stabilization": {
        "candidate_family": "spectral_rollout_stabilization",
        "model_family": "postprocess",
        "cli_command": "run-blueprint-sandbox",
        "sandbox_level": 0,
    },
    "physics_residual_regularization": {
        "candidate_family": "physics_residual_regularization",
        "model_family": "FNO",
        "cli_command": "run-blueprint-sandbox",
        "sandbox_level": 1,
    },
    "scheduled_rollout_training": {
        "candidate_family": "scheduled_rollout_training",
        "model_family": "FNO",
        "cli_command": "execute-fno-finetune-candidate",
        "sandbox_level": 1,
    },
    "operator_correction_postprocess": {
        "candidate_family": "operator_correction_postprocess",
        "model_family": "postprocess",
        "cli_command": "run-hypothesis-lab",
        "sandbox_level": 0,
    },
}

HOOK_ALIASES = {
    "physics_informed_loss": "physics_residual_regularization",
    "pinn_residual_loss": "physics_residual_regularization",
    "pi_deeponet_physics_loss": "physics_residual_regularization",
    "scheduled_sampling": "scheduled_rollout_training",
    "rollout_training": "scheduled_rollout_training",
    "autoregressive_rollout_training": "scheduled_rollout_training",
    "postprocess_operator_correction": "operator_correction_postprocess",
    "residual_blend_postprocess": "operator_correction_postprocess",
}

HOOK_TO_ALGORITHM_CARD_V2 = {
    "official_checkpoint_calibration": {
        "operator_target": "context_to_rollout",
        "mechanism_type": "checkpoint",
        "mathematical_object": "checkpoint_weights",
        "implementation_requirements": {
            "requires_training": False,
            "requires_checkpoint": True,
            "requires_target_data": False,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": False,
        },
    },
    "learned_residual_correction": {
        "operator_target": "residual_to_correction",
        "mechanism_type": "postprocess",
        "mathematical_object": "residual_operator",
        "implementation_requirements": {
            "requires_training": True,
            "requires_checkpoint": True,
            "requires_target_data": True,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": True,
        },
    },
    "spectral_rollout_stabilization": {
        "operator_target": "postprocess_to_solution",
        "mechanism_type": "postprocess",
        "mathematical_object": "spectral_filter",
        "implementation_requirements": {
            "requires_training": False,
            "requires_checkpoint": False,
            "requires_target_data": False,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": True,
        },
    },
    "physics_residual_regularization": {
        "operator_target": "ic_to_trajectory",
        "mechanism_type": "loss",
        "mathematical_object": "burgers_residual",
        "implementation_requirements": {
            "requires_training": True,
            "requires_checkpoint": False,
            "requires_target_data": True,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": False,
        },
    },
    "scheduled_rollout_training": {
        "operator_target": "context_to_rollout",
        "mechanism_type": "rollout",
        "mathematical_object": "time_integrator",
        "implementation_requirements": {
            "requires_training": True,
            "requires_checkpoint": False,
            "requires_target_data": True,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": False,
        },
    },
    "operator_correction_postprocess": {
        "operator_target": "postprocess_to_solution",
        "mechanism_type": "postprocess",
        "mathematical_object": "correction_operator",
        "implementation_requirements": {
            "requires_training": False,
            "requires_checkpoint": False,
            "requires_target_data": True,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": True,
        },
    },
    "fourier_neural_operator_architecture": {
        "operator_target": "ic_to_trajectory",
        "mechanism_type": "architecture",
        "mathematical_object": "fourier_modes",
        "implementation_requirements": {
            "requires_training": True,
            "requires_checkpoint": False,
            "requires_target_data": True,
            "requires_new_executor": True,
            "requires_external_code": False,
            "can_run_as_postprocess": False,
        },
    },
}

ALGORITHM_CARD_V2_OPERATOR_TARGETS = {
    "ic_to_trajectory",
    "coefficient_to_solution",
    "residual_to_correction",
    "context_to_rollout",
    "postprocess_to_solution",
}

ALGORITHM_CARD_V2_MECHANISM_TYPES = {
    "architecture",
    "loss",
    "rollout",
    "postprocess",
    "conditioning",
    "checkpoint",
    "hybrid",
}

ALGORITHM_CARD_V2_EVIDENCE_STATUSES = {
    "extracted",
    "candidate_compiled",
    "sandbox_planned",
    "tested",
    "promoted",
    "rejected",
    "blocked",
}

ALGORITHM_CARD_V2_IMPLEMENTATION_REQUIREMENTS = {
    "requires_training",
    "requires_checkpoint",
    "requires_target_data",
    "requires_new_executor",
    "requires_external_code",
    "can_run_as_postprocess",
}

FAILURE_TAXONOMY = {
    "source_download_failed",
    "parse_failed",
    "parse_quality_insufficient",
    "no_algorithmic_claim",
    "unsupported_executor",
    "requires_external_code",
    "candidate_schema_invalid",
    "blueprint_schema_invalid",
    "sandbox_rule_failed",
    "context_mismatch",
    "target_leakage_risk",
    "score_regressed",
    "runtime_budget_exceeded",
    "physics_regressed",
    "ablation_missing_or_weak",
    "insufficient_reproducibility",
    "runtime_exceeded",
    "duplicate_family",
    "insufficient_evidence",
    "reproducibility_failed",
}

EXECUTABLE_BLUEPRINT_BASELINES = {
    "persistence",
    "fno_checkpoint",
    "retrieval_prior",
}

EXECUTABLE_BLUEPRINT_TRANSFORMS = {
    "linear_trend",
    "spectral_smooth",
    "residual_fit",
    "retrieval_prior",
    "retrieval_refinement",
    "checkpoint_calibration",
}

EXECUTABLE_BLUEPRINT_TRAINING_MODES = {
    "none",
    "finetune",
    "supervised",
    "physics_loss",
}

EXECUTABLE_BLUEPRINT_EXECUTORS = {
    "controlled_numpy_operator",
    "controlled_numpy_residual_operator",
    "controlled_numpy_retrieval_prior",
    "controlled_numpy_retrieval_refinement",
    "official_checkpoint_candidate_pipeline",
}

EXECUTOR_DESIGN_HOOKS = {
    "fourier_neural_operator_architecture": {
        "route": "requires_executor_design",
        "requested_executor_id": "bounded_fno_architecture_executor",
        "fallback_route": "official_checkpoint_family",
        "rationale": (
            "A paper-level FNO architecture claim needs bounded local implementation "
            "or an official-checkpoint adapter before sandbox execution."
        ),
    }
}


class LiteratureAlgorithmLabError(ValueError):
    """Raised when literature algorithm lab artifacts are invalid."""


def normalize_openalex_search_results(
    *,
    search_results_path: str | Path,
    query: str,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    payload = _read_json(_require_file(Path(search_results_path), "OpenAlex results"))
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    retrieval_query = _nonempty_text(query, "query")
    candidates = []
    for item in _object_list(payload.get("results"), "results"):
        title = _nonempty_text(item.get("display_name"), "display_name")
        open_access = item.get("open_access") if isinstance(item.get("open_access"), dict) else {}
        primary = (
            item.get("primary_location")
            if isinstance(item.get("primary_location"), dict)
            else {}
        )
        url = _first_text(
            primary.get("landing_page_url"),
            item.get("doi"),
            item.get("id"),
        )
        document_url = _first_text(open_access.get("oa_url"), url)
        source_id = f"openalex_work_{_slug(title)}"
        candidates.append(
            _source_candidate(
                source_id=source_id,
                source_type="openalex_work",
                title=title,
                url=url,
                document_url=document_url,
                retrieval_query=retrieval_query,
                discovery_provider="openalex",
                license_note="OpenAlex metadata; document license must be checked at source",
                external_ids={
                    "openalex": item.get("id"),
                    "doi": item.get("doi"),
                    "publication_year": item.get("publication_year"),
                    "cited_by_count": item.get("cited_by_count"),
                },
            )
        )
    return _write_source_search_report(
        provider="openalex",
        retrieval_query=retrieval_query,
        candidates=candidates,
        output_file=output_file,
    )


def normalize_arxiv_search_results(
    *,
    search_results_path: str | Path,
    query: str,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    xml_file = _require_file(Path(search_results_path), "arXiv results")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    retrieval_query = _nonempty_text(query, "query")
    root = ET.fromstring(xml_file.read_text(encoding="utf-8"))
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    candidates = []
    for entry in root.findall("atom:entry", namespace):
        title = _nonempty_text(_xml_text(entry, "atom:title", namespace), "title")
        entry_id = _nonempty_text(_xml_text(entry, "atom:id", namespace), "id")
        pdf_url = None
        for link in entry.findall("atom:link", namespace):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
                break
        document_url = _nonempty_text(pdf_url or entry_id.replace("/abs/", "/pdf/"), "document_url")
        arxiv_id = entry_id.rstrip("/").split("/")[-1]
        candidates.append(
            _source_candidate(
                source_id=f"arxiv_paper_{_slug(title)}",
                source_type="arxiv_paper",
                title=" ".join(title.split()),
                url=entry_id,
                document_url=document_url,
                retrieval_query=retrieval_query,
                discovery_provider="arxiv",
                license_note="arXiv paper; check entry license before reuse",
                external_ids={
                    "arxiv_id": arxiv_id,
                    "published": _xml_text(entry, "atom:published", namespace),
                    "updated": _xml_text(entry, "atom:updated", namespace),
                },
            )
        )
    return _write_source_search_report(
        provider="arxiv",
        retrieval_query=retrieval_query,
        candidates=candidates,
        output_file=output_file,
    )


def normalize_github_search_results(
    *,
    search_results_path: str | Path,
    query: str,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    payload = _read_json(_require_file(Path(search_results_path), "GitHub results"))
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    retrieval_query = _nonempty_text(query, "query")
    candidates = []
    for item in _object_list(payload.get("items"), "items"):
        full_name = _nonempty_text(item.get("full_name"), "full_name")
        url = _nonempty_text(item.get("html_url"), "html_url")
        candidates.append(
            _source_candidate(
                source_id=f"github_repo_{_slug(full_name)}",
                source_type="github_repo",
                title=full_name,
                url=url,
                document_url=url,
                retrieval_query=retrieval_query,
                discovery_provider="github",
                license_note="GitHub repository metadata; inspect repository license before reuse",
                external_ids={
                    "full_name": full_name,
                    "default_branch": item.get("default_branch"),
                    "stargazers_count": item.get("stargazers_count"),
                },
            )
        )
    return _write_source_search_report(
        provider="github",
        retrieval_query=retrieval_query,
        candidates=candidates,
        output_file=output_file,
    )


def build_literature_source_records(
    *,
    source_fixture_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    source_file = _require_file(Path(source_fixture_path), "source fixture")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    fixture = _read_json(source_file)
    query = _nonempty_text(fixture.get("query"), "query")
    raw_sources = fixture.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise LiteratureAlgorithmLabError("sources must be a non-empty list")

    records = []
    seen_ids: set[str] = set()
    for item in raw_sources:
        if not isinstance(item, dict):
            raise LiteratureAlgorithmLabError("each source must be an object")
        source_type = _source_type(item.get("source_type"))
        title = _nonempty_text(item.get("title"), "title")
        raw_path = _require_file(
            Path(_nonempty_text(item.get("local_raw_path"), "local_raw_path")),
            "local raw source",
        )
        source_id = _dedupe_id(f"{source_type}_{_slug(title)}", seen_ids)
        seen_ids.add(source_id)
        records.append(
            {
                "source_id": source_id,
                "source_type": source_type,
                "title": title,
                "url": _nonempty_text(item.get("url"), "url"),
                "retrieved_at": _utc_now(),
                "license_note": _nonempty_text(
                    item.get("license_note"),
                    "license_note",
                ),
                "local_raw_path": str(raw_path),
                "content_hash": _sha256_file(raw_path),
                "related_source_ids": _text_list(
                    item.get("related_source_ids", []),
                    "related_source_ids",
                ),
                "retrieval_query": query,
            }
        )

    report = {
        "schema_version": 1,
        "state_hint": SOURCE_RECORDS_STATE_HINT,
        "source_records_path": str(output_file),
        "source_count": len(records),
        "source_records": records,
        "next_command": "ingest-mineru-parse",
    }
    _write_json(output_file, report)
    return report


def update_literature_registry(
    *,
    registry_path: str | Path,
    downloaded_source_paths: list[str | Path] | None = None,
    paper_parse_paths: list[str | Path] | None = None,
    algorithm_cards_paths: list[str | Path] | None = None,
    algorithm_candidates_paths: list[str | Path] | None = None,
    event_paths: list[str | Path] | None = None,
    force: bool,
) -> dict[str, Any]:
    registry_file = Path(registry_path)
    if registry_file.exists() and not force:
        registry = _read_json(registry_file)
    else:
        registry = _empty_literature_registry(registry_file)
    events = registry.setdefault("events", [])
    if not isinstance(events, list):
        raise LiteratureAlgorithmLabError("literature registry events must be a list")

    for artifact_path in downloaded_source_paths or []:
        artifact_file = _require_file(Path(artifact_path), "downloaded source")
        payload = _read_json(artifact_file)
        if payload.get("state_hint") != SOURCE_DOCUMENT_STATE_HINT:
            raise LiteratureAlgorithmLabError(
                "downloaded source artifact has invalid state_hint"
            )
        record = payload.get("source_record")
        if not isinstance(record, dict):
            raise LiteratureAlgorithmLabError("downloaded source missing source_record")
        source = _upsert_registry_source(registry, record)
        source["download_status"] = "downloaded"
        source["local_raw_path"] = record.get("local_raw_path")
        source["content_hash"] = record.get("content_hash")
        _registry_event(
            registry,
            artifact_file=artifact_file,
            event_type="source_downloaded",
            source_ids=[source["source_id"]],
        )

    for artifact_path in paper_parse_paths or []:
        artifact_file = _require_file(Path(artifact_path), "paper parse")
        payload = _read_json(artifact_file)
        if payload.get("state_hint") != PAPER_PARSE_STATE_HINT:
            raise LiteratureAlgorithmLabError(
                "paper parse artifact has invalid state_hint"
            )
        source_id = _nonempty_text(payload.get("source_id"), "source_id")
        source = _upsert_registry_source(
            registry,
            {"source_id": source_id, "source_type": "unknown", "title": source_id},
        )
        quality = payload.get("parse_quality")
        parse_quality = (
            quality.get("quality")
            if isinstance(quality, dict) and isinstance(quality.get("quality"), str)
            else "unknown"
        )
        source["parse_status"] = parse_quality
        source["markdown_path"] = payload.get("markdown_path")
        _registry_event(
            registry,
            artifact_file=artifact_file,
            event_type="paper_parse_ingested",
            source_ids=[source_id],
        )

    for artifact_path in algorithm_cards_paths or []:
        artifact_file = _require_file(Path(artifact_path), "algorithm cards")
        payload = _read_json(artifact_file)
        if payload.get("state_hint") != ALGORITHM_CARDS_STATE_HINT:
            raise LiteratureAlgorithmLabError(
                "algorithm cards artifact has invalid state_hint"
            )
        for card in _object_list(payload.get("algorithm_cards"), "algorithm_cards"):
            node = _upsert_algorithm_node(registry, card)
            for source_id in node["source_ids"]:
                source = _upsert_registry_source(
                    registry,
                    {
                        "source_id": source_id,
                        "source_type": "unknown",
                        "title": source_id,
                    },
                )
                _append_unique_value(source, "algorithm_card_ids", card["algorithm_card_id"])
            _registry_event(
                registry,
                artifact_file=artifact_file,
                event_type="algorithm_card_extracted",
                source_ids=node["source_ids"],
                algorithm_node_id=node["node_id"],
            )

    for artifact_path in algorithm_candidates_paths or []:
        artifact_file = _require_file(Path(artifact_path), "algorithm candidates")
        payload = _read_json(artifact_file)
        if payload.get("state_hint") not in {
            ALGORITHM_CANDIDATES_STATE_HINT,
            FNO_CANDIDATE_ADAPTED_STATE_HINT,
        }:
            raise LiteratureAlgorithmLabError(
                "algorithm candidates artifact has invalid state_hint"
            )
        for candidate in _object_list(
            payload.get("algorithm_candidates"),
            "algorithm_candidates",
        ):
            candidate_node = _upsert_candidate_node(registry, candidate)
            parent_card_id = candidate.get("parent_algorithm_card_id")
            if isinstance(parent_card_id, str) and parent_card_id.strip():
                algorithm = _upsert_algorithm_node(
                    registry,
                    {
                        "algorithm_card_id": parent_card_id,
                        "source_ids": _candidate_source_ids(candidate),
                        "claim": candidate.get("hypothesis", parent_card_id),
                        "core_mechanism": candidate.get("mechanism", ""),
                        "implementation_hooks": _candidate_hooks_for_registry(
                            candidate
                        ),
                    },
                )
                _append_unique_value(
                    algorithm,
                    "candidate_ids",
                    candidate_node["candidate_id"],
                )
            _registry_event(
                registry,
                artifact_file=artifact_file,
                event_type="algorithm_candidate_compiled",
                source_ids=_candidate_source_ids(candidate),
                algorithm_candidate_id=candidate_node["candidate_id"],
            )

    for artifact_path in event_paths or []:
        artifact_file = _require_file(Path(artifact_path), "literature event")
        payload = _read_json(artifact_file)
        _apply_literature_registry_event_artifact(
            registry,
            artifact_file=artifact_file,
            payload=payload,
        )

    _finalize_literature_registry(registry)
    _write_json(registry_file, registry)
    return {
        "schema_version": 1,
        "state_hint": LITERATURE_REGISTRY_UPDATED_STATE_HINT,
        "registry_path": str(registry_file),
        "source_count": registry["source_count"],
        "algorithm_node_count": registry["algorithm_node_count"],
        "candidate_node_count": registry["candidate_node_count"],
        "event_count": registry["event_count"],
        "next_command": "extract-algorithm-cards",
    }


def record_literature_registry_event(
    *,
    event_source_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    source_file = _require_file(Path(event_source_path), "registry event source")
    payload = _read_json(source_file)
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    if payload.get("state_hint") != ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "event source has unsupported state_hint for registry event"
        )
    ledger_event = (
        payload.get("ledger_event")
        if isinstance(payload.get("ledger_event"), dict)
        else {}
    )
    candidate_id = _nonempty_text(
        payload.get("algorithm_candidate_id")
        or ledger_event.get("algorithm_candidate_id"),
        "algorithm_candidate_id",
    )
    decision = _nonempty_text(payload.get("decision"), "decision")
    failure_codes = _normalize_failure_codes(payload.get("reason_codes", []))
    event_type = _registry_event_type_for_candidate_decision(decision)
    source_grounding = (
        payload.get("source_grounding")
        if isinstance(payload.get("source_grounding"), dict)
        else {}
    )
    report = {
        "schema_version": 1,
        "state_hint": LITERATURE_REGISTRY_EVENT_STATE_HINT,
        "literature_registry_event_path": str(output_file),
        "event_source_path": str(source_file),
        "event_type": event_type,
        "algorithm_candidate_id": candidate_id,
        "candidate_family": payload.get("candidate_family")
        or ledger_event.get("candidate_family"),
        "evidence_id": payload.get("best_evidence_id")
        or ledger_event.get("evidence_id"),
        "decision": decision,
        "failure_codes": failure_codes,
        "source_ids": [
            item for item in source_grounding.get("source_ids", []) if isinstance(item, str)
        ],
        "algorithm_card_id": source_grounding.get("algorithm_card_id"),
        "recommended_next": _recommended_next_for_candidate_decision(decision),
        "retry_allowed": _retry_allowed_for_candidate_decision(decision),
        "metric_results": payload.get("metric_results")
        if isinstance(payload.get("metric_results"), dict)
        else {},
        "next_command": "update-literature-registry",
    }
    _write_json(output_file, report)
    return report


def validate_executable_blueprints(
    *,
    blueprints_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    blueprint_file = _require_file(Path(blueprints_path), "blueprints")
    payload = _read_json(blueprint_file)
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    report = audit_executable_blueprint_payload(
        payload,
        blueprints_path=str(blueprint_file),
    )
    report["executable_blueprint_audit_path"] = str(output_file)
    _write_json(output_file, report)
    return report


def run_literature_algorithm_v2_pipeline(
    *,
    algorithm_cards_path: str | Path,
    workspace_dir: str | Path,
    registry_path: str | Path,
    current_best_score: float,
    algorithm_candidate_id: str | None = None,
    sandbox_evidence_paths: list[str | Path] | None = None,
    force: bool,
) -> dict[str, Any]:
    from pde_agent.core.hermes_literature_expansion import (
        summarize_literature_registry_for_hermes,
    )

    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    cards_file = _require_file(Path(algorithm_cards_path), "algorithm cards")
    card_audit_path = workspace / "algorithm_cards_v2_audit.json"
    candidates_path = workspace / "algorithm_candidates.json"
    blueprints_path = workspace / "executable_blueprints.json"
    blueprint_audit_path = workspace / "executable_blueprint_audit.json"
    registry_file = Path(registry_path)
    summary_path = workspace / "hermes_registry_summary.json"

    card_audit = validate_algorithm_cards_v2(
        algorithm_cards_path=cards_file,
        output_path=card_audit_path,
        force=force,
    )
    candidates = compile_algorithm_candidates(
        algorithm_cards_path=cards_file,
        output_path=candidates_path,
        force=force,
    )
    blueprints = _materialize_executable_blueprints_from_candidates(
        candidates=candidates,
        output_path=blueprints_path,
        force=force,
    )
    blueprint_audit = validate_executable_blueprints(
        blueprints_path=blueprints["executable_blueprints_path"],
        output_path=blueprint_audit_path,
        force=force,
    )

    artifacts: dict[str, Any] = {
        "algorithm_cards_v2_audit_path": card_audit["algorithm_card_v2_audit_path"],
        "algorithm_candidates_path": candidates["algorithm_candidates_path"],
        "executable_blueprints_path": blueprints["executable_blueprints_path"],
        "executable_blueprint_audit_path": blueprint_audit[
            "executable_blueprint_audit_path"
        ],
    }
    evidence_paths = sandbox_evidence_paths or []
    promotion = None
    registry_event = None
    if evidence_paths:
        selected_candidate_id = algorithm_candidate_id or _first_candidate_id(candidates)
        promotion_path = workspace / "candidate_promotion.json"
        promotion = evaluate_algorithm_candidate_promotion(
            algorithm_candidates_path=candidates_path,
            algorithm_candidate_id=selected_candidate_id,
            sandbox_evidence_paths=evidence_paths,
            output_path=promotion_path,
            current_best_score=float(current_best_score),
            force=force,
        )
        event_path = workspace / "literature_registry_event.json"
        registry_event = record_literature_registry_event(
            event_source_path=promotion_path,
            output_path=event_path,
            force=force,
        )
        artifacts["candidate_promotion_path"] = promotion[
            "algorithm_candidate_promotion_path"
        ]
        artifacts["literature_registry_event_path"] = registry_event[
            "literature_registry_event_path"
        ]

    event_paths = (
        [registry_event["literature_registry_event_path"]]
        if registry_event is not None
        else []
    )
    registry = update_literature_registry(
        registry_path=registry_file,
        algorithm_cards_paths=[cards_file],
        algorithm_candidates_paths=[candidates_path],
        event_paths=event_paths,
        force=force,
    )
    summary = summarize_literature_registry_for_hermes(
        registry_path=registry_file,
        output_path=summary_path,
        current_best_score=float(current_best_score),
        force=force,
    )
    artifacts["literature_registry_path"] = registry["registry_path"]
    artifacts["hermes_registry_summary_path"] = summary["summary_path"]
    pipeline_decision = (
        "promotion_recorded" if promotion is not None else "awaiting_sandbox_evidence"
    )
    next_command = (
        str(promotion["next_command"])
        if promotion is not None
        else str(blueprint_audit["next_command"])
    )
    report = {
        "schema_version": 1,
        "state_hint": LITERATURE_ALGORITHM_V2_PIPELINE_STATE_HINT,
        "workspace_dir": str(workspace),
        "algorithm_cards_path": str(cards_file),
        "current_best_score": float(current_best_score),
        "pipeline_decision": pipeline_decision,
        "artifacts": artifacts,
        "metrics": {
            "card_issue_count": card_audit["issue_count"],
            "candidate_count": candidates["algorithm_candidate_count"],
            "blueprint_count": blueprints["blueprint_count"],
            "blueprint_issue_count": blueprint_audit["issue_count"],
            "registry_event_count": registry["event_count"],
            "hermes_candidate_action_count": summary["metrics"][
                "candidate_action_count"
            ],
        },
        "next_command": next_command,
    }
    _write_json(workspace / "literature_algorithm_v2_pipeline.json", report)
    return report


def audit_executable_blueprint_payload(
    payload: dict[str, Any],
    *,
    blueprints_path: str | None = None,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    blueprints_value = payload.get("blueprints")
    if not isinstance(blueprints_value, list):
        _append_blueprint_issue(
            issues,
            code="invalid_blueprints",
            path="blueprints",
            failure_code="blueprint_schema_invalid",
            message="blueprints must be a list",
        )
        blueprints: list[Any] = []
    else:
        blueprints = blueprints_value
    normalized = []
    for index, blueprint in enumerate(blueprints):
        item = _normalize_executable_blueprint(blueprint, index=index, issues=issues)
        if item is not None:
            normalized.append(item)
            _validate_executable_blueprint(item, index=index, issues=issues)
    decision = "approved" if not issues else "rejected"
    report: dict[str, Any] = {
        "schema_version": 1,
        "state_hint": EXECUTABLE_BLUEPRINTS_VALIDATED_STATE_HINT,
        "decision": decision,
        "blueprint_count": len(blueprints),
        "normalized_blueprint_count": len(normalized),
        "issue_count": len(issues),
        "issues": issues,
        "normalized_blueprints": normalized,
        "next_command": "run-blueprint-sandbox"
        if decision == "approved"
        else "fix-executable-blueprint-dsl",
    }
    if blueprints_path is not None:
        report["blueprints_path"] = blueprints_path
    return report


def download_source_document(
    *,
    source_candidates_path: str | Path,
    source_id: str,
    output_path: str | Path,
    cache_dir: str | Path,
    downloader: Any | None = None,
    force: bool,
) -> dict[str, Any]:
    candidates_report = _read_json(
        _require_file(Path(source_candidates_path), "source candidates")
    )
    if candidates_report.get("state_hint") != SOURCE_SEARCH_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "source candidates report has invalid state_hint"
        )
    selected_id = _nonempty_text(source_id, "source_id")
    candidate = _find_by_id(
        candidates_report.get("source_candidates"),
        selected_id,
        "source_id",
    )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    download_url = _nonempty_text(candidate.get("document_url"), "document_url")
    extension = _extension_for_url(download_url)
    local_raw_path = Path(cache_dir) / f"{_slug(selected_id)}{extension}"
    if local_raw_path.exists() and not force:
        content = local_raw_path.read_bytes()
    else:
        fetch = downloader or _download_bytes
        content = fetch(download_url)
        if not isinstance(content, bytes) or not content:
            raise LiteratureAlgorithmLabError("downloaded content must be non-empty bytes")
        local_raw_path.parent.mkdir(parents=True, exist_ok=True)
        local_raw_path.write_bytes(content)

    source_record = {
        "source_id": candidate["source_id"],
        "source_type": candidate["source_type"],
        "title": candidate["title"],
        "url": candidate["url"],
        "retrieved_at": _utc_now(),
        "license_note": candidate["license_note"],
        "local_raw_path": str(local_raw_path),
        "content_hash": hashlib.sha256(content).hexdigest(),
        "related_source_ids": [],
        "retrieval_query": candidate["retrieval_query"],
        "download_url": download_url,
        "discovery_provider": candidate["discovery_provider"],
        "external_ids": candidate.get("external_ids", {}),
    }
    report = {
        "schema_version": 1,
        "state_hint": SOURCE_DOCUMENT_STATE_HINT,
        "downloaded_source_path": str(output_file),
        "source_record": source_record,
        "next_command": "submit-mineru-parse",
    }
    _write_json(output_file, report)
    return report


def submit_mineru_parse_task(
    *,
    downloaded_source_path: str | Path,
    output_path: str | Path,
    mode: str,
    token: str | None,
    model_version: str,
    client: Any | None = None,
    force: bool,
) -> dict[str, Any]:
    downloaded = _read_json(
        _require_file(Path(downloaded_source_path), "downloaded source")
    )
    if downloaded.get("state_hint") != SOURCE_DOCUMENT_STATE_HINT:
        raise LiteratureAlgorithmLabError("downloaded source has invalid state_hint")
    source_record = downloaded.get("source_record")
    if not isinstance(source_record, dict):
        raise LiteratureAlgorithmLabError("downloaded source missing source_record")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    mineru_mode = _mineru_mode(mode)
    http = client or MinerUHttpClient()
    download_url = _nonempty_text(source_record.get("download_url"), "download_url")
    if mineru_mode == "agent_url":
        endpoint = MINERU_AGENT_URL_ENDPOINT
        payload = {
            "url": download_url,
            "language": "en",
            "enable_table": True,
            "is_ocr": False,
            "enable_formula": True,
        }
        auth_token = None
    else:
        endpoint = MINERU_PRECISE_URL_ENDPOINT
        payload = {
            "url": download_url,
            "model_version": _nonempty_text(model_version, "model_version"),
            "enable_formula": True,
            "enable_table": True,
            "is_ocr": False,
            "data_id": source_record["source_id"],
        }
        auth_token = _nonempty_text(token, "MINERU_API_TOKEN")
    response = http.post_json(endpoint, payload, auth_token)
    data = _mineru_data(response)
    task_id = _nonempty_text(data.get("task_id"), "task_id")
    report = {
        "schema_version": 1,
        "state_hint": MINERU_TASK_SUBMITTED_STATE_HINT,
        "mineru_task_path": str(output_file),
        "source_id": source_record["source_id"],
        "source_title": source_record.get("title"),
        "source_content_hash": source_record.get("content_hash"),
        "mineru_mode": mineru_mode,
        "submit_endpoint": endpoint,
        "model_version": model_version,
        "task_id": task_id,
        "request_payload": payload,
        "token_source": "MINERU_API_TOKEN" if auth_token else None,
        "next_command": "poll-mineru-parse",
    }
    _write_json(output_file, report)
    return report


def poll_mineru_parse_task(
    *,
    mineru_task_path: str | Path,
    output_path: str | Path,
    token: str | None,
    client: Any | None = None,
    force: bool,
) -> dict[str, Any]:
    task = _read_json(_require_file(Path(mineru_task_path), "MinerU task"))
    if task.get("state_hint") != MINERU_TASK_SUBMITTED_STATE_HINT:
        raise LiteratureAlgorithmLabError("MinerU task has invalid state_hint")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    http = client or MinerUHttpClient()
    task_id = _nonempty_text(task.get("task_id"), "task_id")
    mode = _mineru_mode(task.get("mineru_mode"))
    endpoint = (
        f"{MINERU_AGENT_POLL_ENDPOINT}/{task_id}"
        if mode == "agent_url"
        else f"{MINERU_PRECISE_URL_ENDPOINT}/{task_id}"
    )
    response = http.get_json(endpoint, None if mode == "agent_url" else token)
    data = _mineru_data(response)
    state = _nonempty_text(data.get("state"), "state")
    report = {
        "schema_version": 1,
        "state_hint": MINERU_TASK_POLLED_STATE_HINT,
        "mineru_poll_path": str(output_file),
        "mineru_task_path": str(Path(mineru_task_path)),
        "source_id": task.get("source_id"),
        "mineru_mode": mode,
        "task_id": task_id,
        "poll_endpoint": endpoint,
        "state": state,
        "markdown_url": data.get("markdown_url"),
        "full_zip_url": data.get("full_zip_url"),
        "err_msg": data.get("err_msg"),
        "err_code": data.get("err_code"),
        "next_command": "download-mineru-result"
        if state == "done"
        else "poll-mineru-parse",
    }
    _write_json(output_file, report)
    return report


def download_mineru_parse_result(
    *,
    mineru_poll_path: str | Path,
    output_path: str | Path,
    output_dir: str | Path,
    client: Any | None = None,
    force: bool,
) -> dict[str, Any]:
    poll = _read_json(_require_file(Path(mineru_poll_path), "MinerU poll"))
    if poll.get("state_hint") != MINERU_TASK_POLLED_STATE_HINT:
        raise LiteratureAlgorithmLabError("MinerU poll has invalid state_hint")
    if poll.get("state") != "done":
        raise LiteratureAlgorithmLabError("MinerU task is not done")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    http = client or MinerUHttpClient()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    markdown_url = poll.get("markdown_url")
    full_zip_url = poll.get("full_zip_url")
    local_markdown_path = None
    local_zip_path = None
    if isinstance(markdown_url, str) and markdown_url.strip():
        local_markdown_path = root / "full.md"
        local_markdown_path.write_bytes(http.download_bytes(markdown_url))
    if isinstance(full_zip_url, str) and full_zip_url.strip():
        local_zip_path = root / "mineru_result.zip"
        local_zip_path.write_bytes(http.download_bytes(full_zip_url))
    if local_markdown_path is None and local_zip_path is None:
        raise LiteratureAlgorithmLabError(
            "MinerU result has neither markdown_url nor full_zip_url"
        )
    report = {
        "schema_version": 1,
        "state_hint": MINERU_RESULT_DOWNLOADED_STATE_HINT,
        "mineru_result_path": str(output_file),
        "mineru_poll_path": str(Path(mineru_poll_path)),
        "source_id": poll.get("source_id"),
        "task_id": poll.get("task_id"),
        "local_markdown_path": str(local_markdown_path)
        if local_markdown_path is not None
        else None,
        "local_zip_path": str(local_zip_path) if local_zip_path is not None else None,
        "markdown_url": markdown_url,
        "full_zip_url": full_zip_url,
        "next_command": "ingest-mineru-parse",
    }
    _write_json(output_file, report)
    return report


def ingest_mineru_parse(
    *,
    source_records_path: str | Path,
    source_id: str,
    markdown_path: str | Path,
    asset_dir: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    source_records = _read_json(
        _require_file(Path(source_records_path), "source records")
    )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    markdown_file = _require_file(Path(markdown_path), "MinerU markdown")
    selected_source_id = _nonempty_text(source_id, "source_id")
    source_ids = _source_ids_from_source_artifact(source_records)
    if selected_source_id not in source_ids:
        raise LiteratureAlgorithmLabError(
            f"source_id not found in source records: {selected_source_id}"
        )

    markdown = markdown_file.read_text(encoding="utf-8")
    section_index = _section_index(markdown)
    formula_count = markdown.count("$$") // 2
    figure_count = markdown.count("![")
    table_count = markdown.count("\n|")
    quality_flags = _parse_quality_flags(
        markdown=markdown,
        section_index=section_index,
        formula_count=formula_count,
    )
    quality = "usable" if not quality_flags else "low_quality"
    report = {
        "schema_version": 1,
        "state_hint": PAPER_PARSE_STATE_HINT,
        "parse_id": f"parse_{selected_source_id}",
        "source_id": selected_source_id,
        "parser": "mineru",
        "parser_version": "external",
        "markdown_path": str(markdown_file),
        "asset_dir": str(Path(asset_dir)),
        "parse_quality": {
            "quality": quality,
            "quality_flags": quality_flags,
        },
        "section_index": section_index,
        "formula_count": formula_count,
        "figure_count": figure_count,
        "table_count": table_count,
        "paper_parse_path": str(output_file),
        "next_command": "extract-algorithm-cards",
    }
    _write_json(output_file, report)
    return report


def extract_algorithm_cards(
    *,
    paper_parse_paths: list[str | Path],
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    if not paper_parse_paths:
        raise LiteratureAlgorithmLabError("paper_parse_paths must be non-empty")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    cards = []
    extraction_audit = {
        "parse_count": len(paper_parse_paths),
        "usable_parse_count": 0,
        "low_quality_parse_count": 0,
        "scanned_block_count": 0,
        "extracted_block_count": 0,
        "inferred_block_count": 0,
        "skipped_parses": [],
        "skipped_blocks": [],
    }
    for parse_path in paper_parse_paths:
        parse = _read_json(_require_file(Path(parse_path), "paper parse"))
        if parse.get("state_hint") != PAPER_PARSE_STATE_HINT:
            raise LiteratureAlgorithmLabError("paper parse has invalid state_hint")
        if parse.get("parse_quality", {}).get("quality") != "usable":
            extraction_audit["low_quality_parse_count"] += 1
            extraction_audit["skipped_parses"].append(
                {
                    "paper_parse_path": str(Path(parse_path)),
                    "source_id": parse.get("source_id"),
                    "reason_code": "parse_quality_not_usable",
                    "quality": parse.get("parse_quality", {}).get("quality"),
                    "quality_flags": parse.get("parse_quality", {}).get(
                        "quality_flags",
                        [],
                    ),
                }
            )
            continue
        extraction_audit["usable_parse_count"] += 1
        markdown_file = _require_file(
            Path(_nonempty_text(parse.get("markdown_path"), "markdown_path")),
            "parsed markdown",
        )
        markdown = markdown_file.read_text(encoding="utf-8")
        source_id = _nonempty_text(parse.get("source_id"), "source_id")
        field_blocks, skipped_blocks = _scan_algorithm_card_field_blocks(markdown)
        extraction_audit["scanned_block_count"] += len(field_blocks) + len(
            skipped_blocks
        )
        extraction_audit["extracted_block_count"] += len(field_blocks)
        extraction_audit["inferred_block_count"] += sum(
            1
            for field_block in field_blocks
            if field_block.get("source_shape") == "inferred_section"
        )
        for skipped_block in skipped_blocks:
            extraction_audit["skipped_blocks"].append(
                {"source_id": source_id, **skipped_block}
            )
        if not field_blocks:
            field_blocks = [
                {
                    "fields": _extract_colon_fields(markdown),
                    "field_spans": _extract_colon_field_spans(
                        markdown,
                        line_offset=0,
                        section_title=None,
                    ),
                    "markdown": markdown,
                    "section_title": None,
                    "block_index": 0,
                    "source_shape": "whole_document",
                    "line_start": 1,
                    "line_end": len(markdown.splitlines()) or 1,
                }
            ]
        for field_block in field_blocks:
            fields = dict(field_block["fields"])
            block_markdown = str(field_block["markdown"])
            if "hook" not in fields:
                fields = _infer_algorithm_fields(block_markdown)
            hook = _nonempty_text(fields.get("hook"), "Hook")
            section_title = field_block.get("section_title")
            card = {
                "algorithm_card_id": _algorithm_card_id(
                    source_id=source_id,
                    hook=hook,
                    section_title=section_title
                    if isinstance(section_title, str)
                    else None,
                ),
                "source_ids": [source_id],
                "claim": _nonempty_text(fields.get("claim"), "Claim"),
                "core_mechanism": _nonempty_text(
                    fields.get("mechanism"),
                    "Mechanism",
                ),
                "operation": _nonempty_text(fields.get("operation"), "Operation"),
                "preconditions": ["PDEBench-style neural operator rollout"],
                "implementation_hooks": [hook],
                "expected_observables": [
                    _nonempty_text(fields.get("observable"), "Observable")
                ],
                "failure_modes": [_nonempty_text(fields.get("risk"), "Risk")],
                "minimal_experiment": _nonempty_text(
                    fields.get("minimal experiment"),
                    "Minimal Experiment",
                ),
                "evidence_spans": _evidence_spans_for_fields(
                    field_block.get("field_spans")
                    if isinstance(field_block.get("field_spans"), dict)
                    else {},
                    [
                        "claim",
                        "mechanism",
                        "operation",
                        "hook",
                        "observable",
                        "risk",
                        "minimal experiment",
                    ],
                ),
                "evidence_status": "extracted",
            }
            card.update(
                _algorithm_card_v2_fields(
                    fields=fields,
                    hook=hook,
                    markdown=block_markdown,
                    parse=parse,
                    parse_path=Path(parse_path),
                    markdown_path=markdown_file,
                    section_title=section_title
                    if isinstance(section_title, str)
                    else None,
                    block_index=int(field_block["block_index"]),
                    source_shape=str(field_block["source_shape"]),
                )
            )
            cards.append(card)
    if not cards:
        raise LiteratureAlgorithmLabError("no usable algorithm cards extracted")
    report = {
        "schema_version": 1,
        "state_hint": ALGORITHM_CARDS_STATE_HINT,
        "algorithm_cards_path": str(output_file),
        "algorithm_card_count": len(cards),
        "algorithm_cards": cards,
        "extraction_audit": extraction_audit,
        "next_command": "compile-algorithm-candidates",
    }
    _write_json(output_file, report)
    return report


def validate_algorithm_cards_v2(
    *,
    algorithm_cards_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    cards_report = _read_json(
        _require_file(Path(algorithm_cards_path), "algorithm cards")
    )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    issues: list[dict[str, Any]] = []
    if cards_report.get("state_hint") != ALGORITHM_CARDS_STATE_HINT:
        _append_algorithm_card_v2_issue(
            issues,
            code="invalid_state_hint",
            path="state_hint",
            failure_code="candidate_schema_invalid",
            message="algorithm cards report must have ALGORITHM_CARDS_EXTRACTED state",
        )
    cards_value = cards_report.get("algorithm_cards")
    if not isinstance(cards_value, list):
        _append_algorithm_card_v2_issue(
            issues,
            code="invalid_algorithm_cards",
            path="algorithm_cards",
            failure_code="candidate_schema_invalid",
            message="algorithm_cards must be a list",
        )
        cards: list[Any] = []
    else:
        cards = cards_value
    for index, card in enumerate(cards):
        issues.extend(_validate_algorithm_card_v2(card, index=index))
    decision = "approved" if not issues else "rejected"
    next_command = (
        "compile-algorithm-candidates"
        if decision == "approved"
        else "fix-algorithm-card-v2-schema"
    )
    report = {
        "schema_version": 1,
        "state_hint": ALGORITHM_CARDS_V2_VALIDATED_STATE_HINT,
        "algorithm_cards_path": str(Path(algorithm_cards_path)),
        "algorithm_card_v2_audit_path": str(output_file),
        "decision": decision,
        "card_count": len(cards),
        "issue_count": len(issues),
        "issues": issues,
        "next_command": next_command,
    }
    _write_json(output_file, report)
    return report


def compile_algorithm_candidates(
    *,
    algorithm_cards_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    cards_report = _read_json(
        _require_file(Path(algorithm_cards_path), "algorithm cards")
    )
    if cards_report.get("state_hint") != ALGORITHM_CARDS_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "algorithm cards report has invalid state_hint"
        )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    candidates = []
    for card in cards_report.get("algorithm_cards", []):
        if not isinstance(card, dict):
            raise LiteratureAlgorithmLabError("algorithm card must be an object")
        seen_families: set[str] = set()
        for hook in _text_list(card.get("implementation_hooks"), "implementation_hooks"):
            normalized_hook = _normalize_algorithm_hook(hook)
            family_spec = HOOK_TO_FAMILY.get(normalized_hook)
            if family_spec is None:
                candidates.append(_blocked_candidate(card=card, hook=hook))
                continue
            family = str(family_spec["candidate_family"])
            if family in seen_families:
                continue
            seen_families.add(family)
            candidate_id = f"candidate_{_slug(card['algorithm_card_id'])}_{family}"
            candidates.append(
                {
                    "algorithm_candidate_id": candidate_id,
                    "parent_algorithm_card_id": card["algorithm_card_id"],
                    "candidate_family": family_spec["candidate_family"],
                    "target_problem": "PDE_Burgers",
                    "target_task": "task1",
                    "model_family": family_spec["model_family"],
                    "hypothesis": card["claim"],
                    "mechanism": card["core_mechanism"],
                    "literature_hook": hook,
                    "normalized_hook": normalized_hook,
                    "executor_spec": _executor_spec(str(family_spec["cli_command"])),
                    "parameter_grid": _default_parameter_grid(family),
                    "benchmark_plan": {
                        "judge": "hermes_local_judge",
                        "primary_metric": "task1_total_score_proxy",
                        "sandbox_level": family_spec["sandbox_level"],
                    },
                    "ablation_plan": {
                        "baseline": "persistence_or_current_best",
                        "contrast": "candidate_disabled_or_weight_zero",
                    },
                    "promotion_criteria": {
                        "min_score_delta": 0.01,
                        "max_inference_seconds": 120.0,
                        "requires_ablation": True,
                    },
                    "risk_controls": [
                        "preserve first 10 context steps",
                        "do not execute external repo code",
                    ],
                    "source_grounding": {
                        "source_ids": card["source_ids"],
                        "algorithm_card_id": card["algorithm_card_id"],
                    },
                }
            )
    report = {
        "schema_version": 1,
        "state_hint": ALGORITHM_CANDIDATES_STATE_HINT,
        "algorithm_candidates_path": str(output_file),
        "algorithm_candidate_count": len(candidates),
        "algorithm_candidates": candidates,
        "next_command": "plan-algorithm-candidate-sandbox",
    }
    _write_json(output_file, report)
    return report


def plan_candidate_sandbox(
    *,
    algorithm_candidates_path: str | Path,
    algorithm_candidate_id: str,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    candidates_report = _read_json(
        _require_file(Path(algorithm_candidates_path), "algorithm candidates")
    )
    if candidates_report.get("state_hint") not in {
        ALGORITHM_CANDIDATES_STATE_HINT,
        FNO_CANDIDATE_ADAPTED_STATE_HINT,
    }:
        raise LiteratureAlgorithmLabError(
            "algorithm candidates report has invalid state_hint"
        )
    selected_id = _nonempty_text(algorithm_candidate_id, "algorithm_candidate_id")
    selected = None
    for candidate in candidates_report.get("algorithm_candidates", []):
        if (
            isinstance(candidate, dict)
            and candidate.get("algorithm_candidate_id") == selected_id
        ):
            selected = candidate
            break
    if selected is None:
        raise LiteratureAlgorithmLabError(
            f"algorithm candidate not found: {selected_id}"
        )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    executor = selected.get("executor_spec")
    if not isinstance(executor, dict):
        raise LiteratureAlgorithmLabError("candidate executor_spec must be an object")
    cli_command = _nonempty_text(executor.get("cli_command"), "cli_command")
    blocked = cli_command == "blocked_requires_executor_design"
    report = {
        "schema_version": 1,
        "state_hint": SANDBOX_PLAN_STATE_HINT,
        "sandbox_plan_path": str(output_file),
        "algorithm_candidate_id": selected_id,
        "candidate_family": selected.get("candidate_family"),
        "sandbox_level": _sandbox_level(str(selected.get("candidate_family"))),
        "executor": executor,
        "blocked": blocked,
        "reason_codes": ["requires_new_bounded_executor"] if blocked else [],
        "risk_controls": selected.get("risk_controls", []),
        "next_command": "create-executor-design-request" if blocked else cli_command,
    }
    _write_json(output_file, report)
    return report


def evaluate_algorithm_candidate_promotion(
    *,
    algorithm_candidates_path: str | Path,
    algorithm_candidate_id: str,
    sandbox_evidence_paths: list[str | Path],
    output_path: str | Path,
    current_best_score: float,
    force: bool,
) -> dict[str, Any]:
    candidates_report = _read_json(
        _require_file(Path(algorithm_candidates_path), "algorithm candidates")
    )
    if candidates_report.get("state_hint") not in {
        ALGORITHM_CANDIDATES_STATE_HINT,
        FNO_CANDIDATE_ADAPTED_STATE_HINT,
    }:
        raise LiteratureAlgorithmLabError(
            "algorithm candidates report has invalid state_hint"
        )
    selected_id = _nonempty_text(algorithm_candidate_id, "algorithm_candidate_id")
    selected = _find_algorithm_candidate(candidates_report, selected_id)
    if not sandbox_evidence_paths:
        raise LiteratureAlgorithmLabError("sandbox_evidence_paths must be non-empty")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    evidence_records = [
        _read_json(_require_file(Path(path), "sandbox evidence"))
        for path in sandbox_evidence_paths
    ]
    best_evidence = _select_best_candidate_evidence(
        evidence_records=evidence_records,
        candidate_id=selected_id,
    )
    passed_checks, failed_checks = _evaluate_candidate_promotion_checks(
        candidate=selected,
        evidence=best_evidence,
        current_best_score=current_best_score,
    )
    reason_codes = [
        str(check["failure_code"]) for check in failed_checks.values()
    ] or ["all_rubric_checks_passed"]
    decision = _candidate_promotion_decision(
        candidate=selected,
        failed_checks=failed_checks,
        evidence=best_evidence,
    )
    promote = decision == "promote"
    ledger_event = {
        "event_type": f"candidate_{'promoted' if promote else decision}",
        "algorithm_candidate_id": selected_id,
        "candidate_family": selected.get("candidate_family"),
        "evidence_id": best_evidence.get("evidence_id"),
        "decision": decision,
        "reason_codes": reason_codes,
    }
    report = {
        "schema_version": 1,
        "state_hint": ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT,
        "algorithm_candidate_promotion_path": str(output_file),
        "algorithm_candidates_path": str(Path(algorithm_candidates_path)),
        "algorithm_candidate_id": selected_id,
        "candidate_family": selected.get("candidate_family"),
        "sandbox_evidence_paths": [str(Path(path)) for path in sandbox_evidence_paths],
        "evidence_count": len(evidence_records),
        "best_evidence_id": best_evidence.get("evidence_id"),
        "current_best_score": current_best_score,
        "decision": decision,
        "promote_to_strategy_growth": promote,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "reason_codes": reason_codes,
        "metric_results": _candidate_promotion_metric_results(
            evidence=best_evidence,
            current_best_score=current_best_score,
        ),
        "ledger_event": ledger_event,
        "next_command": _candidate_promotion_next_command(decision),
    }
    _write_json(output_file, report)
    return report


def build_executor_taxonomy(
    *,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    hook_mappings: dict[str, dict[str, Any]] = {}
    for hook, family_spec in HOOK_TO_FAMILY.items():
        hook_mappings[hook] = {
            "route": "existing_executor",
            "candidate_family": family_spec["candidate_family"],
            "model_family": family_spec["model_family"],
            "cli_command": family_spec["cli_command"],
            "sandbox_level": family_spec["sandbox_level"],
        }
    hook_mappings.update(EXECUTOR_DESIGN_HOOKS)
    report = {
        "schema_version": 1,
        "state_hint": EXECUTOR_TAXONOMY_STATE_HINT,
        "executor_taxonomy_path": str(output_file),
        "hook_mappings": hook_mappings,
        "route_counts": {
            "existing_executor": len(HOOK_TO_FAMILY),
            "requires_executor_design": len(EXECUTOR_DESIGN_HOOKS),
        },
        "next_command": "create-executor-design-request",
    }
    _write_json(output_file, report)
    return report


def adapt_fno_architecture_candidate(
    *,
    algorithm_candidates_path: str | Path,
    algorithm_candidate_id: str,
    checkpoint_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    candidates_report = _read_json(
        _require_file(Path(algorithm_candidates_path), "algorithm candidates")
    )
    if candidates_report.get("state_hint") != ALGORITHM_CANDIDATES_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "algorithm candidates report has invalid state_hint"
        )
    selected_id = _nonempty_text(algorithm_candidate_id, "algorithm_candidate_id")
    candidate = _find_by_id(
        candidates_report.get("algorithm_candidates"),
        selected_id,
        "algorithm_candidate_id",
    )
    executor = candidate.get("executor_spec")
    if not isinstance(executor, dict):
        raise LiteratureAlgorithmLabError("candidate executor_spec must be an object")
    if executor.get("unsupported_hook") != "fourier_neural_operator_architecture":
        raise LiteratureAlgorithmLabError(
            "FNO architecture adapter requires a fourier_neural_operator_architecture candidate"
        )
    checkpoint_file = _require_file(Path(checkpoint_path), "checkpoint resource")
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    source_grounding = candidate.get("source_grounding")
    if not isinstance(source_grounding, dict):
        source_grounding = {}
    adapted_source_grounding = {
        **source_grounding,
        "adapter_source_candidate_id": selected_id,
        "adapter_route": "official_checkpoint_family",
    }
    adapted_candidate = {
        "algorithm_candidate_id": f"{selected_id}_official_checkpoint_adapter",
        "parent_algorithm_card_id": candidate.get("parent_algorithm_card_id"),
        "candidate_family": "official_checkpoint_family",
        "target_problem": candidate.get("target_problem", "PDE_Burgers"),
        "target_task": candidate.get("target_task", "task1"),
        "model_family": "checkpoint",
        "hypothesis": candidate.get("hypothesis"),
        "mechanism": (
            f"{candidate.get('mechanism')} Adapted through a local official "
            "checkpoint-family sandbox instead of unbounded architecture code."
        ),
        "executor_spec": _executor_spec("run-checkpoint-family-sandbox"),
        "parameter_grid": _default_parameter_grid("official_checkpoint_family"),
        "resource_bindings": {"checkpoint_paths": [str(checkpoint_file)]},
        "benchmark_plan": {
            "judge": "hermes_local_judge",
            "primary_metric": "task1_total_score_proxy",
            "sandbox_level": 1,
        },
        "ablation_plan": {
            "baseline": "persistence_or_current_best",
            "contrast": "checkpoint_adapter_disabled",
        },
        "promotion_criteria": {
            "min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
            "requires_ablation": True,
        },
        "risk_controls": [
            "preserve first 10 context steps",
            "do not execute external repo code",
            "verify checkpoint tensor convention before promotion",
        ],
        "source_grounding": adapted_source_grounding,
    }
    report = {
        "schema_version": 1,
        "state_hint": FNO_CANDIDATE_ADAPTED_STATE_HINT,
        "adapted_candidate_path": str(output_file),
        "algorithm_candidates_path": str(output_file),
        "source_algorithm_candidate_id": selected_id,
        "algorithm_candidate_count": 1,
        "adapted_candidate": adapted_candidate,
        "algorithm_candidates": [adapted_candidate],
        "next_command": "plan-algorithm-candidate-sandbox",
    }
    _write_json(output_file, report)
    return report


def create_executor_design_request(
    *,
    algorithm_candidates_path: str | Path,
    algorithm_candidate_id: str,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    candidates_report = _read_json(
        _require_file(Path(algorithm_candidates_path), "algorithm candidates")
    )
    if candidates_report.get("state_hint") != ALGORITHM_CANDIDATES_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "algorithm candidates report has invalid state_hint"
        )
    candidate = _find_by_id(
        candidates_report.get("algorithm_candidates"),
        _nonempty_text(algorithm_candidate_id, "algorithm_candidate_id"),
        "algorithm_candidate_id",
    )
    executor = candidate.get("executor_spec")
    if not isinstance(executor, dict):
        raise LiteratureAlgorithmLabError("candidate executor_spec must be an object")
    if executor.get("cli_command") != "blocked_requires_executor_design":
        raise LiteratureAlgorithmLabError(
            "executor design request requires a blocked candidate"
        )
    hook = _nonempty_text(executor.get("unsupported_hook"), "unsupported_hook")
    hook_spec = EXECUTOR_DESIGN_HOOKS.get(hook, {})
    requested_executor_id = str(
        hook_spec.get("requested_executor_id", f"bounded_{_slug(hook)}_executor")
    )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    strategy_id = f"external_fno_executor_design_{_slug(str(candidate['algorithm_candidate_id']))}"
    hypothesis = _nonempty_text(candidate.get("hypothesis"), "hypothesis")
    mechanism = _nonempty_text(candidate.get("mechanism"), "mechanism")
    source_grounding = candidate.get("source_grounding")
    if not isinstance(source_grounding, dict):
        source_grounding = {}
    report = {
        "schema_version": 1,
        "state_hint": EXECUTOR_DESIGN_REQUEST_STATE_HINT,
        "executor_design_request_path": str(output_file),
        "algorithm_candidate_id": candidate["algorithm_candidate_id"],
        "candidate_family": candidate.get("candidate_family"),
        "unsupported_hook": hook,
        "requested_executor_id": requested_executor_id,
        "hypothesis": hypothesis,
        "mechanism": mechanism,
        "source_grounding": source_grounding,
        "reason_codes": [
            "unsupported_algorithm_hook",
            "requires_bounded_executor",
        ],
        "executor_contract": {
            "inputs": [
                "algorithm_candidate_path",
                "validation_windows_path",
                "optional_checkpoint_path",
                "output_dir",
            ],
            "outputs": [
                "candidate_prediction_path",
                "candidate_execution_report_path",
                "sandbox_result_path",
            ],
            "forbidden_actions": [
                "execute external repository code directly",
                "mutate source training data",
                "skip first-10-step identity validation",
            ],
        },
        "minimal_experiment": {
            "level": 2,
            "description": (
                "Implement a bounded executor adapter, run it on local Burgers "
                "validation windows, then compare against persistence and current best."
            ),
        },
        "risk_controls": list(candidate.get("risk_controls", []))
        + [
            "require local sandbox score before promotion",
            "record failure even when executor design is abandoned",
        ],
        "recommended_intermediate_actions": [
            "search official FNO checkpoint resource",
            "map checkpoint resource to run-checkpoint-family-sandbox when available",
            "otherwise design bounded_fno_architecture_executor",
        ],
        "strategy_proposal_fields": {
            "strategy_id": strategy_id,
            "title": "Design bounded FNO executor from literature candidate",
            "rationale": hypothesis,
            "implementation_plan": [
                "Translate the literature FNO mechanism into a bounded local executor contract.",
                "Prefer official checkpoint-family execution if a compatible checkpoint is available.",
                "Run a local Burgers sandbox and require score, runtime, and first-context identity evidence before promotion.",
            ],
            "expected_observables": [
                "higher local judge score than persistence or current best",
                "stable inference under the two-minute task budget",
            ],
            "known_risks": [
                "FNO tensor conventions may not match official checkpoints",
                "architecture reproduction may consume budget without improving long-horizon score",
            ],
            "source_type": "external_retrieval",
            "source_grounding": source_grounding,
        },
        "next_command": "build-blocked-candidate-handoff",
    }
    _write_json(output_file, report)
    return report


def build_blocked_candidate_handoff(
    *,
    executor_design_request_path: str | Path,
    output_path: str | Path,
    force: bool,
) -> dict[str, Any]:
    request = _read_json(
        _require_file(Path(executor_design_request_path), "executor design request")
    )
    if request.get("state_hint") != EXECUTOR_DESIGN_REQUEST_STATE_HINT:
        raise LiteratureAlgorithmLabError(
            "executor design request has invalid state_hint"
        )
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    proposal_fields = request.get("strategy_proposal_fields")
    if not isinstance(proposal_fields, dict):
        raise LiteratureAlgorithmLabError(
            "executor design request missing strategy_proposal_fields"
        )
    report = {
        "schema_version": 1,
        "state_hint": BLOCKED_CANDIDATE_HANDOFF_STATE_HINT,
        "blocked_candidate_handoff_path": str(output_file),
        "algorithm_candidate_id": request.get("algorithm_candidate_id"),
        "record_strategy_proposal_fields": proposal_fields,
        "blocked_evidence_summary": {
            "blocked": True,
            "requires_executor": request.get("requested_executor_id"),
            "reason_codes": request.get("reason_codes", []),
            "source_grounding": request.get("source_grounding", {}),
        },
        "next_command": "record-strategy-proposal",
    }
    _write_json(output_file, report)
    return report


def _blocked_candidate(*, card: dict[str, Any], hook: str) -> dict[str, Any]:
    return {
        "algorithm_candidate_id": f"candidate_{_slug(card['algorithm_card_id'])}_blocked",
        "parent_algorithm_card_id": card["algorithm_card_id"],
        "candidate_family": "unsupported",
        "target_problem": "PDE_Burgers",
        "target_task": "task1",
        "model_family": "hybrid",
        "hypothesis": card["claim"],
        "mechanism": card["core_mechanism"],
        "executor_spec": {
            "cli_command": "blocked_requires_executor_design",
            "unsupported_hook": hook,
            "required_options": [],
        },
        "parameter_grid": {},
        "benchmark_plan": {
            "judge": "hermes_local_judge",
            "primary_metric": "task1_total_score_proxy",
            "sandbox_level": 2,
        },
        "ablation_plan": {
            "baseline": "persistence_or_current_best",
            "contrast": "candidate_disabled",
        },
        "promotion_criteria": {
            "min_score_delta": 0.01,
            "max_inference_seconds": 120.0,
            "requires_ablation": True,
        },
        "risk_controls": ["blocked until a bounded executor is designed"],
        "source_grounding": {
            "source_ids": card["source_ids"],
            "algorithm_card_id": card["algorithm_card_id"],
        },
    }


def _empty_literature_registry(registry_file: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state_hint": LITERATURE_REGISTRY_UPDATED_STATE_HINT,
        "registry_path": str(registry_file),
        "sources": [],
        "algorithm_nodes": [],
        "candidate_nodes": [],
        "events": [],
        "next_graph_model": {
            "node_semantics": (
                "algorithm_nodes represent reproducible or attempted algorithmic "
                "mechanisms; paper sources are provenance, not the graph center"
            ),
            "edge_types": [
                "source_supports_algorithm",
                "algorithm_generates_candidate",
                "candidate_verified_by_sandbox",
                "candidate_blocked_by_executor_gap",
                "candidate_promoted_to_strategy",
            ],
        },
    }


def _upsert_registry_source(
    registry: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    source_id = _nonempty_text(record.get("source_id"), "source_id")
    sources = registry.setdefault("sources", [])
    if not isinstance(sources, list):
        raise LiteratureAlgorithmLabError("literature registry sources must be a list")
    for source in sources:
        if isinstance(source, dict) and source.get("source_id") == source_id:
            target = source
            break
    else:
        target = {
            "source_id": source_id,
            "source_type": record.get("source_type", "unknown"),
            "title": record.get("title", source_id),
            "url": record.get("url"),
            "retrieval_query": record.get("retrieval_query"),
            "license_note": record.get("license_note"),
            "download_status": "seen",
            "parse_status": "not_parsed",
            "algorithm_card_ids": [],
        }
        sources.append(target)
    for key in [
        "source_type",
        "title",
        "url",
        "retrieval_query",
        "license_note",
        "content_hash",
        "local_raw_path",
        "markdown_path",
    ]:
        value = record.get(key)
        if value is not None:
            if key == "source_type" and value == "unknown" and target.get(key):
                continue
            if key == "title" and value == source_id and target.get(key):
                continue
            target[key] = value
    return target


def _upsert_algorithm_node(
    registry: dict[str, Any],
    card: dict[str, Any],
) -> dict[str, Any]:
    card_id = _nonempty_text(card.get("algorithm_card_id"), "algorithm_card_id")
    node_id = f"algorithm_{card_id}"
    nodes = registry.setdefault("algorithm_nodes", [])
    if not isinstance(nodes, list):
        raise LiteratureAlgorithmLabError(
            "literature registry algorithm_nodes must be a list"
        )
    for node in nodes:
        if isinstance(node, dict) and node.get("node_id") == node_id:
            target = node
            break
    else:
        target = {
            "node_id": node_id,
            "algorithm_card_id": card_id,
            "candidate_ids": [],
            "strategy_ids": [],
            "strategy_status": "extracted",
        }
        nodes.append(target)
    source_ids = _text_list(card.get("source_ids", []), "source_ids")
    target["source_ids"] = source_ids
    target["claim"] = card.get("claim")
    target["mechanism"] = card.get("core_mechanism")
    target["implementation_hooks"] = [
        item
        for item in card.get("implementation_hooks", [])
        if isinstance(item, str)
    ]
    return target


def _upsert_candidate_node(
    registry: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = _nonempty_text(
        candidate.get("algorithm_candidate_id"),
        "algorithm_candidate_id",
    )
    nodes = registry.setdefault("candidate_nodes", [])
    if not isinstance(nodes, list):
        raise LiteratureAlgorithmLabError(
            "literature registry candidate_nodes must be a list"
        )
    for node in nodes:
        if isinstance(node, dict) and node.get("candidate_id") == candidate_id:
            target = node
            break
    else:
        target = {"candidate_id": candidate_id, "status": "compiled"}
        nodes.append(target)
    target["candidate_family"] = candidate.get("candidate_family")
    target["parent_algorithm_card_id"] = candidate.get("parent_algorithm_card_id")
    target["source_ids"] = _candidate_source_ids(candidate)
    executor = candidate.get("executor_spec")
    if isinstance(executor, dict):
        target["executor"] = executor.get("cli_command")
        if executor.get("cli_command") == "blocked_requires_executor_design":
            target["status"] = "blocked"
    resource_bindings = candidate.get("resource_bindings")
    if isinstance(resource_bindings, dict):
        target["resource_bindings"] = resource_bindings
    return target


def _candidate_source_ids(candidate: dict[str, Any]) -> list[str]:
    grounding = candidate.get("source_grounding")
    if not isinstance(grounding, dict):
        return []
    return [
        item for item in grounding.get("source_ids", []) if isinstance(item, str)
    ]


def _candidate_hooks_for_registry(candidate: dict[str, Any]) -> list[str]:
    executor = candidate.get("executor_spec")
    if not isinstance(executor, dict):
        return []
    hook = executor.get("unsupported_hook")
    if isinstance(hook, str) and hook.strip():
        return [hook.strip()]
    hooks = executor.get("implementation_hooks")
    if isinstance(hooks, list):
        return [item for item in hooks if isinstance(item, str)]
    return []


def _apply_literature_registry_event_artifact(
    registry: dict[str, Any],
    *,
    artifact_file: Path,
    payload: dict[str, Any],
) -> None:
    state = payload.get("state_hint")
    if state == "STRATEGY_PROPOSAL_RECORDED":
        strategy_id = _nonempty_text(payload.get("strategy_id"), "strategy_id")
        implementation_plan = (
            payload.get("implementation_plan")
            if isinstance(payload.get("implementation_plan"), dict)
            else {}
        )
        candidate_id = _nonempty_text(
            implementation_plan.get("algorithm_candidate_id") or strategy_id,
            "algorithm_candidate_id",
        )
        grounding = (
            implementation_plan.get("source_grounding")
            if isinstance(implementation_plan.get("source_grounding"), dict)
            else {}
        )
        source_ids = [
            item for item in grounding.get("source_ids", []) if isinstance(item, str)
        ]
        algorithm_card_id = grounding.get("algorithm_card_id")
        _set_candidate_status(registry, candidate_id, "promoted")
        algorithm_node_id = None
        if isinstance(algorithm_card_id, str) and algorithm_card_id.strip():
            algorithm = _upsert_algorithm_node(
                registry,
                {
                    "algorithm_card_id": algorithm_card_id,
                    "source_ids": source_ids,
                    "claim": payload.get("hypothesis", algorithm_card_id),
                    "core_mechanism": payload.get("mechanism", ""),
                    "implementation_hooks": [],
                },
            )
            algorithm["strategy_status"] = "strategy_proposed"
            _append_unique_value(algorithm, "candidate_ids", candidate_id)
            _append_unique_value(algorithm, "strategy_ids", strategy_id)
            algorithm_node_id = algorithm["node_id"]
        _registry_event(
            registry,
            artifact_file=artifact_file,
            event_type="strategy_proposal_recorded",
            source_ids=source_ids,
            algorithm_candidate_id=candidate_id,
            algorithm_node_id=algorithm_node_id,
            reason_codes=[
                "source_type_external_retrieval"
                if payload.get("source_type") == "external_retrieval"
                else "strategy_proposal"
            ],
        )
        return
    if state == "CHECKPOINT_FAMILY_SANDBOX_RAN":
        best_candidate = (
            payload.get("best_candidate")
            if isinstance(payload.get("best_candidate"), dict)
            else {}
        )
        checkpoint_path = best_candidate.get("checkpoint_path")
        matched_candidates = [
            candidate
            for candidate in registry.get("candidate_nodes", [])
            if isinstance(candidate, dict)
            and candidate.get("executor") == "run-checkpoint-family-sandbox"
            and _candidate_uses_checkpoint(candidate, checkpoint_path)
        ]
        if not matched_candidates:
            _registry_event(
                registry,
                artifact_file=artifact_file,
                event_type="checkpoint_family_sandbox_unlinked",
                source_ids=[],
            )
            return
        rules_passed = bool(best_candidate.get("rules_passed"))
        status = "sandbox_verified" if rules_passed else "sandbox_failed"
        event_type = (
            "candidate_sandbox_verified"
            if rules_passed
            else "candidate_sandbox_failed"
        )
        for candidate in matched_candidates:
            candidate_id = _nonempty_text(candidate.get("candidate_id"), "candidate_id")
            _set_candidate_status(registry, candidate_id, status)
            candidate = _find_registry_candidate(registry, candidate_id)
            evidence = {
                "artifact_path": str(artifact_file),
                "best_score": best_candidate.get("score"),
                "rules_passed": rules_passed,
                "inference_time": best_candidate.get("inference_time"),
                "calibration_weight": best_candidate.get("calibration_weight"),
            }
            _append_unique_object(candidate, "sandbox_evidence", evidence)
            algorithm_node_id = None
            parent_card_id = candidate.get("parent_algorithm_card_id")
            source_ids = [
                item for item in candidate.get("source_ids", []) if isinstance(item, str)
            ]
            if isinstance(parent_card_id, str) and parent_card_id.strip():
                algorithm = _upsert_algorithm_node(
                    registry,
                    {
                        "algorithm_card_id": parent_card_id,
                        "source_ids": source_ids,
                        "claim": parent_card_id,
                        "core_mechanism": "",
                        "implementation_hooks": [],
                    },
                )
                algorithm["strategy_status"] = status
                _append_unique_value(algorithm, "candidate_ids", candidate_id)
                algorithm_node_id = algorithm["node_id"]
            _registry_event(
                registry,
                artifact_file=artifact_file,
                event_type=event_type,
                source_ids=source_ids,
                algorithm_candidate_id=candidate_id,
                algorithm_node_id=algorithm_node_id,
                reason_codes=[f"score={best_candidate.get('score')}"],
            )
        return
    if state == BLOCKED_CANDIDATE_HANDOFF_STATE_HINT:
        candidate_id = _nonempty_text(
            payload.get("algorithm_candidate_id"),
            "algorithm_candidate_id",
        )
        summary = (
            payload.get("blocked_evidence_summary")
            if isinstance(payload.get("blocked_evidence_summary"), dict)
            else {}
        )
        grounding = (
            summary.get("source_grounding")
            if isinstance(summary.get("source_grounding"), dict)
            else {}
        )
        source_ids = [
            item for item in grounding.get("source_ids", []) if isinstance(item, str)
        ]
        algorithm_card_id = grounding.get("algorithm_card_id")
        _set_candidate_status(registry, candidate_id, "blocked")
        if isinstance(algorithm_card_id, str) and algorithm_card_id.strip():
            algorithm = _upsert_algorithm_node(
                registry,
                {
                    "algorithm_card_id": algorithm_card_id,
                    "source_ids": source_ids,
                    "claim": algorithm_card_id,
                    "core_mechanism": "",
                    "implementation_hooks": [],
                },
            )
            algorithm["strategy_status"] = "blocked"
            _append_unique_value(algorithm, "candidate_ids", candidate_id)
        _registry_event(
            registry,
            artifact_file=artifact_file,
            event_type="candidate_blocked",
            source_ids=source_ids,
            algorithm_candidate_id=candidate_id,
            algorithm_node_id=f"algorithm_{algorithm_card_id}"
            if isinstance(algorithm_card_id, str)
            else None,
            reason_codes=[
                item for item in summary.get("reason_codes", []) if isinstance(item, str)
            ],
        )
        return
    if state == LITERATURE_REGISTRY_EVENT_STATE_HINT:
        candidate_id = _nonempty_text(
            payload.get("algorithm_candidate_id"),
            "algorithm_candidate_id",
        )
        event_type = _nonempty_text(payload.get("event_type"), "event_type")
        status = _candidate_status_for_registry_event(event_type)
        source_ids = [
            item for item in payload.get("source_ids", []) if isinstance(item, str)
        ]
        _set_candidate_status(registry, candidate_id, status)
        candidate = _find_registry_candidate(registry, candidate_id)
        failure_codes = _normalize_failure_codes(payload.get("failure_codes", []))
        if failure_codes:
            _append_unique_object(
                candidate,
                "failure_history",
                {
                    "artifact_path": str(artifact_file),
                    "failure_codes": failure_codes,
                    "recommended_next": payload.get("recommended_next"),
                    "retry_allowed": bool(payload.get("retry_allowed")),
                },
            )
        algorithm_node_id = None
        algorithm_card_id = payload.get("algorithm_card_id")
        if not isinstance(algorithm_card_id, str) or not algorithm_card_id.strip():
            parent_card_id = candidate.get("parent_algorithm_card_id")
            algorithm_card_id = parent_card_id if isinstance(parent_card_id, str) else None
        if isinstance(algorithm_card_id, str) and algorithm_card_id.strip():
            algorithm = _upsert_algorithm_node(
                registry,
                {
                    "algorithm_card_id": algorithm_card_id,
                    "source_ids": source_ids
                    or [
                        item
                        for item in candidate.get("source_ids", [])
                        if isinstance(item, str)
                    ],
                    "claim": algorithm_card_id,
                    "core_mechanism": "",
                    "implementation_hooks": [],
                },
            )
            algorithm["strategy_status"] = status
            _append_unique_value(algorithm, "candidate_ids", candidate_id)
            algorithm_node_id = algorithm["node_id"]
        _registry_event(
            registry,
            artifact_file=artifact_file,
            event_type=event_type,
            source_ids=source_ids,
            algorithm_candidate_id=candidate_id,
            algorithm_node_id=algorithm_node_id,
            reason_codes=failure_codes,
        )
        return
    _registry_event(
        registry,
        artifact_file=artifact_file,
        event_type=str(state or "unknown_event"),
        source_ids=[],
    )


def _set_candidate_status(
    registry: dict[str, Any],
    candidate_id: str,
    status: str,
) -> None:
    for candidate in registry.get("candidate_nodes", []):
        if isinstance(candidate, dict) and candidate.get("candidate_id") == candidate_id:
            candidate["status"] = status
            return
    registry.setdefault("candidate_nodes", []).append(
        {"candidate_id": candidate_id, "status": status, "source_ids": []}
    )


def _find_registry_candidate(
    registry: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    for candidate in registry.get("candidate_nodes", []):
        if isinstance(candidate, dict) and candidate.get("candidate_id") == candidate_id:
            return candidate
    raise LiteratureAlgorithmLabError(f"candidate not found in registry: {candidate_id}")


def _candidate_uses_checkpoint(
    candidate: dict[str, Any],
    checkpoint_path: object,
) -> bool:
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        return False
    bindings = candidate.get("resource_bindings")
    if not isinstance(bindings, dict):
        return False
    paths = bindings.get("checkpoint_paths")
    return isinstance(paths, list) and checkpoint_path in paths


def _normalize_failure_codes(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise LiteratureAlgorithmLabError("failure codes must be a list")
    codes = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise LiteratureAlgorithmLabError("failure_code must be a non-empty string")
        code = item.strip()
        if code == "all_rubric_checks_passed":
            continue
        if code not in FAILURE_TAXONOMY:
            raise LiteratureAlgorithmLabError(f"unknown failure_code: {code}")
        if code not in codes:
            codes.append(code)
    return codes


def _registry_event_type_for_candidate_decision(decision: str) -> str:
    mapping = {
        "promote": "candidate_promoted",
        "refine": "candidate_refined",
        "deprioritize": "candidate_deprioritized",
        "reject": "candidate_rejected",
        "blocked": "candidate_blocked",
    }
    return mapping.get(decision, f"candidate_{decision}")


def _candidate_status_for_registry_event(event_type: str) -> str:
    mapping = {
        "candidate_promoted": "promoted",
        "candidate_refined": "needs_refinement",
        "candidate_deprioritized": "deprioritized",
        "candidate_rejected": "rejected",
        "candidate_blocked": "blocked",
    }
    return mapping.get(event_type, event_type.removeprefix("candidate_"))


def _recommended_next_for_candidate_decision(decision: str) -> str:
    mapping = {
        "promote": "enter_strategy_growth",
        "refine": "refine_with_new_ablation_or_variant",
        "deprioritize": "avoid_repeating_without_new_evidence",
        "reject": "avoid_repeating_without_new_evidence",
        "blocked": "design_executor_before_retry",
    }
    return mapping.get(decision, "review-literature-registry")


def _retry_allowed_for_candidate_decision(decision: str) -> bool:
    return decision in {"refine", "blocked"}


def _registry_event(
    registry: dict[str, Any],
    *,
    artifact_file: Path,
    event_type: str,
    source_ids: list[str],
    algorithm_node_id: str | None = None,
    algorithm_candidate_id: str | None = None,
    reason_codes: list[str] | None = None,
) -> None:
    events = registry.setdefault("events", [])
    event = {
        "event_id": _sha256_text(
            "|".join(
                [
                    event_type,
                    str(artifact_file),
                    algorithm_node_id or "",
                    algorithm_candidate_id or "",
                ]
            )
        )[:16],
        "event_type": event_type,
        "artifact_path": str(artifact_file),
        "source_ids": source_ids,
        "recorded_at": _utc_now(),
    }
    if algorithm_node_id is not None:
        event["algorithm_node_id"] = algorithm_node_id
    if algorithm_candidate_id is not None:
        event["algorithm_candidate_id"] = algorithm_candidate_id
    if reason_codes:
        event["reason_codes"] = reason_codes
    if not any(
        isinstance(existing, dict) and existing.get("event_id") == event["event_id"]
        for existing in events
    ):
        events.append(event)


def _append_unique_value(target: dict[str, Any], key: str, value: str) -> None:
    values = target.setdefault(key, [])
    if not isinstance(values, list):
        values = []
        target[key] = values
    if value not in values:
        values.append(value)


def _append_unique_object(
    target: dict[str, Any],
    key: str,
    value: dict[str, Any],
) -> None:
    values = target.setdefault(key, [])
    if not isinstance(values, list):
        values = []
        target[key] = values
    if value not in values:
        values.append(value)


def _finalize_literature_registry(registry: dict[str, Any]) -> None:
    for key in ["sources", "algorithm_nodes", "candidate_nodes"]:
        items = registry.get(key)
        if isinstance(items, list):
            items.sort(key=lambda item: json.dumps(item, sort_keys=True))
    registry["source_count"] = len(registry.get("sources", []))
    registry["algorithm_node_count"] = len(registry.get("algorithm_nodes", []))
    registry["candidate_node_count"] = len(registry.get("candidate_nodes", []))
    registry["event_count"] = len(registry.get("events", []))
    registry["updated_at"] = _utc_now()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _executor_spec(cli_command: str) -> dict[str, Any]:
    if cli_command == "run-checkpoint-family-sandbox":
        return {
            "cli_command": cli_command,
            "required_options": [
                "--checkpoint-path",
                "--task-id",
                "--test-path",
                "--target-path",
                "--output-dir",
            ],
        }
    if cli_command == "execute-fno-finetune-candidate":
        return {
            "cli_command": cli_command,
            "required_options": [
                "--candidate-plan-path",
                "--windows-path",
                "--checkpoint-path",
                "--max-train-steps",
            ],
        }
    return {
        "cli_command": cli_command,
        "required_options": [
            "--blueprint-path",
            "--test-path",
            "--target-path",
            "--output-dir",
        ],
    }


def _default_parameter_grid(candidate_family: str) -> dict[str, Any]:
    if candidate_family == "official_checkpoint_family":
        return {
            "calibration_weights": [0.0, 0.25, 0.5, 0.75, 1.0],
            "fno_finetune_steps": [0, 1],
        }
    if candidate_family == "lightweight_residual_corrector":
        return {"residual_shrinkage": [0.25, 0.5, 0.75], "horizon_decay": [40.0, 80.0]}
    if candidate_family == "spectral_rollout_stabilization":
        return {"spectral_strength": [0.25, 0.5, 1.0], "cutoff_ratio": [0.5, 0.75]}
    if candidate_family == "physics_residual_regularization":
        return {"physics_weight": [0.001, 0.01, 0.1], "spectral_weight": [0.0, 0.1]}
    if candidate_family == "scheduled_rollout_training":
        return {"rollout_steps": [1, 2, 4], "teacher_forcing_ratio": [0.5, 0.75]}
    if candidate_family == "operator_correction_postprocess":
        return {
            "correction_strength": [0.25, 0.5, 0.75],
            "spectral_strength": [0.0, 0.5],
        }
    return {"candidate_enabled": [True]}


def _sandbox_level(candidate_family: str) -> int:
    if candidate_family in {
        "lightweight_residual_corrector",
        "spectral_rollout_stabilization",
        "operator_correction_postprocess",
    }:
        return 0
    if candidate_family in {
        "official_checkpoint_family",
        "physics_residual_regularization",
        "scheduled_rollout_training",
    }:
        return 1
    return 2


def _normalize_algorithm_hook(hook: str) -> str:
    normalized = hook.strip()
    return HOOK_ALIASES.get(normalized, normalized)


def _section_index(markdown: str) -> dict[str, int]:
    sections: dict[str, int] = {}
    for line_number, line in enumerate(markdown.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            name = stripped.lstrip("#").strip().lower()
            if name:
                sections[name] = line_number
    return sections


def _parse_quality_flags(
    *,
    markdown: str,
    section_index: dict[str, int],
    formula_count: int,
) -> list[str]:
    flags = []
    lowered_sections = set(section_index)
    if "abstract" not in lowered_sections:
        flags.append("missing_abstract")
    has_method_section = any(
        keyword in section
        for section in lowered_sections
        for keyword in ["method", "approach", "algorithm", "neural operator"]
    )
    if not has_method_section:
        flags.append("missing_method")
    if len(markdown.strip()) < 200:
        flags.append("markdown_too_short")
    if formula_count == 0:
        flags.append("no_formula_blocks")
    return flags


def _extract_colon_fields(markdown: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in markdown.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        key = label.strip().lower()
        text = value.strip()
        if key and text:
            fields[key] = text
    return fields


def _extract_colon_field_spans(
    markdown: str,
    *,
    line_offset: int,
    section_title: str | None,
) -> dict[str, dict[str, Any]]:
    spans: dict[str, dict[str, Any]] = {}
    for local_line_number, line in enumerate(markdown.splitlines(), start=1):
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        key = label.strip().lower()
        if not key or not value.strip():
            continue
        absolute_line = line_offset + local_line_number
        span = {
            "text": line.strip(),
            "value": value.strip(),
            "line_start": absolute_line,
            "line_end": absolute_line,
        }
        if section_title:
            span["section_title"] = section_title
        spans[key] = span
    return spans


def _extract_algorithm_card_field_blocks(markdown: str) -> list[dict[str, Any]]:
    extracted, _ = _scan_algorithm_card_field_blocks(markdown)
    return extracted


def _scan_algorithm_card_field_blocks(
    markdown: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks = _markdown_heading_blocks(markdown)
    algorithm_blocks = []
    skipped_blocks = []
    for block in blocks:
        fields = _extract_colon_fields(str(block["markdown"]))
        missing_fields = _missing_algorithm_card_fields(fields)
        if _has_complete_algorithm_card_fields(fields):
            algorithm_blocks.append(
                {
                    "fields": fields,
                    "field_spans": _extract_colon_field_spans(
                        str(block["markdown"]),
                        line_offset=int(block["line_start"]) - 1,
                        section_title=block["section_title"]
                        if isinstance(block["section_title"], str)
                        else None,
                    ),
                    "markdown": block["markdown"],
                    "section_title": block["section_title"],
                    "block_index": block["block_index"],
                    "source_shape": "structured_section",
                    "line_start": block["line_start"],
                    "line_end": block["line_end"],
                }
            )
            continue
        inferred_block = _infer_nonstructured_algorithm_field_block(block)
        if inferred_block is not None:
            algorithm_blocks.append(inferred_block)
            continue
        skipped_blocks.append(
            {
                "section_title": block["section_title"],
                "block_index": block["block_index"],
                "line_start": block["line_start"],
                "line_end": block["line_end"],
                "reason_code": "missing_algorithm_card_fields",
                "missing_fields": missing_fields,
            }
        )
    return algorithm_blocks, skipped_blocks


def _infer_nonstructured_algorithm_field_block(
    block: dict[str, Any],
) -> dict[str, Any] | None:
    section_title = (
        block["section_title"] if isinstance(block.get("section_title"), str) else ""
    )
    markdown = str(block["markdown"])
    text_lines = _non_heading_lines(markdown)
    joined = " ".join(text for _, text in text_lines)
    if not joined.strip():
        return None
    lowered = f"{section_title} {joined}".lower()
    inferred = _infer_nonstructured_algorithm_fields(
        lowered=lowered,
        joined=joined,
        text_lines=text_lines,
    )
    if inferred is None:
        return None
    fields, field_spans = inferred
    return {
        "fields": fields,
        "field_spans": _attach_section_to_spans(
            field_spans,
            section_title=section_title or None,
            line_offset=int(block["line_start"]) - 1,
        ),
        "markdown": markdown,
        "section_title": section_title or None,
        "block_index": block["block_index"],
        "source_shape": "inferred_section",
        "line_start": block["line_start"],
        "line_end": block["line_end"],
    }


def _infer_nonstructured_algorithm_fields(
    *,
    lowered: str,
    joined: str,
    text_lines: list[tuple[int, str]],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]] | None:
    if _looks_like_spectral_rollout_section(lowered):
        claim_line = _first_line_matching(
            text_lines,
            ["stabilize", "damping", "damp", "spectral", "fourier"],
        )
        mechanism_line = _first_line_matching(
            text_lines,
            ["filter", "coefficients"],
        )
        risk_line = _first_line_matching(
            text_lines,
            ["oversmooth", "shock", "discontinu"],
        )
        fields = {
            "claim": (
                "Spectral damping can stabilize Burgers autoregressive rollouts "
                "by suppressing high-frequency drift after the context window."
            ),
            "mechanism": (
                "Apply a cutoff spectral filter to Fourier coefficients during "
                "rollout so high modes are damped before the next predicted step."
            ),
            "operation": (
                "Apply a cutoff spectral filter after the context window during "
                "autoregressive Burgers rollout prediction."
            ),
            "hook": "spectral_rollout_stabilization",
            "observable": "lower late-horizon RMSE and reduced high-frequency drift",
            "risk": "spectral damping can oversmooth shocks or discontinuities",
            "minimal experiment": (
                "Sweep spectral cutoff and damping strength on held-out Burgers "
                "windows and compare late-horizon RMSE against the current best."
            ),
        }
        spans = {
            "claim": _line_span_from_match(claim_line, fallback_text=joined),
            "mechanism": _line_span_from_match(
                mechanism_line or claim_line,
                fallback_text=joined,
            ),
            "operation": _line_span_from_match(
                mechanism_line or claim_line,
                fallback_text=joined,
            ),
            "risk": _line_span_from_match(risk_line, fallback_text=joined),
        }
        return fields, spans
    if _looks_like_residual_correction_section(lowered):
        claim_line = _first_line_matching(text_lines, ["residual", "bias", "correct"])
        mechanism_line = _first_line_matching(text_lines, ["learn", "fit", "map"])
        risk_line = _first_line_matching(text_lines, ["overfit", "validation"])
        fields = {
            "claim": (
                "A learned residual correction can reduce systematic rollout bias "
                "in Burgers neural-operator forecasts."
            ),
            "mechanism": (
                "Fit a lightweight residual operator on local validation windows "
                "and apply the correction as a postprocess."
            ),
            "operation": (
                "Fit and apply a horizon-aware residual correction to checkpoint "
                "or baseline rollout predictions."
            ),
            "hook": "learned_residual_correction",
            "observable": "lower early and middle Rel-MSE without material runtime cost",
            "risk": "residual correction can overfit local validation windows",
            "minimal experiment": (
                "Compare baseline rollout against residual-corrected rollout on "
                "held-out Burgers validation windows."
            ),
        }
        spans = {
            "claim": _line_span_from_match(claim_line, fallback_text=joined),
            "mechanism": _line_span_from_match(
                mechanism_line or claim_line,
                fallback_text=joined,
            ),
            "operation": _line_span_from_match(
                mechanism_line or claim_line,
                fallback_text=joined,
            ),
            "risk": _line_span_from_match(risk_line, fallback_text=joined),
        }
        return fields, spans
    if _looks_like_physics_residual_section(lowered):
        claim_line = _first_line_matching(text_lines, ["physics", "residual", "pde"])
        mechanism_line = _first_line_matching(text_lines, ["loss", "regular"])
        fields = {
            "claim": (
                "Physics residual regularization can improve Burgers rollout "
                "consistency by penalizing PDE violations."
            ),
            "mechanism": (
                "Evaluate a Burgers residual term and add it as a training or "
                "fine-tuning loss."
            ),
            "operation": (
                "Add a Burgers PDE residual penalty to the candidate training or "
                "fine-tuning objective."
            ),
            "hook": "physics_residual_regularization",
            "observable": "lower Burgers residual without score regression",
            "risk": "physics loss can hurt data-fit accuracy or increase runtime",
            "minimal experiment": (
                "Run a small residual-loss sweep and compare score, runtime, and "
                "Burgers residual against the data-only baseline."
            ),
        }
        spans = {
            "claim": _line_span_from_match(claim_line, fallback_text=joined),
            "mechanism": _line_span_from_match(
                mechanism_line or claim_line,
                fallback_text=joined,
            ),
            "operation": _line_span_from_match(
                mechanism_line or claim_line,
                fallback_text=joined,
            ),
        }
        return fields, spans
    if _looks_like_scheduled_rollout_section(lowered):
        claim_line = _first_line_matching(
            text_lines,
            ["rollout", "autoregressive", "teacher", "scheduled"],
        )
        fields = {
            "claim": (
                "Scheduled rollout training can reduce train-inference mismatch "
                "for autoregressive Burgers prediction."
            ),
            "mechanism": (
                "Gradually expose the model to its own predicted states during "
                "multi-step rollout training."
            ),
            "operation": (
                "Fine-tune with scheduled multi-step autoregressive rollout windows."
            ),
            "hook": "scheduled_rollout_training",
            "observable": "lower middle and late-horizon error",
            "risk": "rollout fine-tuning can be slower and destabilize short-horizon fit",
            "minimal experiment": (
                "Run a bounded scheduled-rollout fine-tune and compare segment "
                "scores against one-step training."
            ),
        }
        spans = {
            "claim": _line_span_from_match(claim_line, fallback_text=joined),
            "mechanism": _line_span_from_match(claim_line, fallback_text=joined),
            "operation": _line_span_from_match(claim_line, fallback_text=joined),
        }
        return fields, spans
    return None


def _non_heading_lines(markdown: str) -> list[tuple[int, str]]:
    lines = []
    for local_line_number, line in enumerate(markdown.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append((local_line_number, stripped))
    return lines


def _looks_like_spectral_rollout_section(lowered: str) -> bool:
    return (
        ("spectral" in lowered or "fourier" in lowered)
        and any(keyword in lowered for keyword in ["damp", "filter", "viscosity"])
        and any(keyword in lowered for keyword in ["rollout", "predict", "horizon"])
    )


def _looks_like_residual_correction_section(lowered: str) -> bool:
    return "residual" in lowered and any(
        keyword in lowered for keyword in ["correct", "bias", "postprocess", "fit"]
    )


def _looks_like_physics_residual_section(lowered: str) -> bool:
    return (
        ("physics" in lowered or "pde" in lowered or "burgers" in lowered)
        and "residual" in lowered
        and any(keyword in lowered for keyword in ["loss", "regular", "penalty"])
    )


def _looks_like_scheduled_rollout_section(lowered: str) -> bool:
    return "rollout" in lowered and any(
        keyword in lowered
        for keyword in ["scheduled", "teacher", "autoregressive", "multi-step"]
    )


def _first_line_matching(
    text_lines: list[tuple[int, str]],
    keywords: list[str],
) -> tuple[int, str] | None:
    for local_line_number, text in text_lines:
        lowered = text.lower()
        if any(keyword in lowered for keyword in keywords):
            return local_line_number, text
    return text_lines[0] if text_lines else None


def _line_span_from_match(
    match: tuple[int, str] | None,
    *,
    fallback_text: str,
) -> dict[str, Any]:
    if match is None:
        return {
            "text": fallback_text.strip(),
            "value": fallback_text.strip(),
            "line_start": 1,
            "line_end": 1,
        }
    local_line_number, text = match
    return {
        "text": text,
        "value": text,
        "line_start": local_line_number,
        "line_end": local_line_number,
    }


def _attach_section_to_spans(
    spans: dict[str, dict[str, Any]],
    *,
    section_title: str | None,
    line_offset: int,
) -> dict[str, dict[str, Any]]:
    grounded = {}
    for key, span in spans.items():
        line_start = int(span["line_start"]) + line_offset
        line_end = int(span["line_end"]) + line_offset
        grounded_span = {
            **span,
            "line_start": line_start,
            "line_end": line_end,
        }
        if section_title:
            grounded_span["section_title"] = section_title
        grounded[key] = grounded_span
    return grounded


def _markdown_heading_blocks(markdown: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    current_start_line = 1
    current_index = 0
    for line_number, line in enumerate(markdown.splitlines(), start=1):
        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            if current_lines:
                blocks.append(
                    {
                        "section_title": current_title,
                        "markdown": "\n".join(current_lines).strip(),
                        "block_index": current_index,
                        "line_start": current_start_line,
                        "line_end": line_number - 1,
                    }
                )
                current_index += 1
            current_title = heading.group(2).strip()
            current_lines = [line]
            current_start_line = line_number
            continue
        current_lines.append(line)
    if current_lines:
        line_count = len(markdown.splitlines())
        blocks.append(
            {
                "section_title": current_title,
                "markdown": "\n".join(current_lines).strip(),
                "block_index": current_index,
                "line_start": current_start_line,
                "line_end": line_count,
            }
        )
    return [block for block in blocks if str(block["markdown"]).strip()]


def _has_complete_algorithm_card_fields(fields: dict[str, str]) -> bool:
    return not _missing_algorithm_card_fields(fields)


def _missing_algorithm_card_fields(fields: dict[str, str]) -> list[str]:
    return [
        key
        for key in ["claim", "mechanism", "operation", "hook"]
        if key not in fields
    ]


def _evidence_spans_for_fields(
    spans: dict[str, Any],
    field_names: list[str],
) -> dict[str, Any]:
    evidence = {}
    for field_name in field_names:
        span = spans.get(field_name)
        if isinstance(span, dict):
            evidence[field_name.replace(" ", "_")] = span
    return evidence


def _algorithm_card_id(
    *,
    source_id: str,
    hook: str,
    section_title: str | None,
) -> str:
    parts = ["card", source_id]
    if section_title:
        parts.append(_slug(section_title))
    parts.append(_slug(hook))
    return "_".join(parts)


def _infer_algorithm_fields(markdown: str) -> dict[str, str]:
    lowered = markdown.lower()
    if "fourier neural operator" in lowered:
        return {
            "claim": (
                "Fourier Neural Operator learns parametric PDE solution "
                "operators by mapping functions to functions across discretizations."
            ),
            "mechanism": (
                "The kernel integral operator is parameterized in Fourier space "
                "so low-frequency global modes can be learned efficiently."
            ),
            "operation": (
                "Implement a bounded FNO rollout or adapt an official FNO "
                "checkpoint, then evaluate Burgers long-horizon prediction."
            ),
            "hook": "fourier_neural_operator_architecture",
            "observable": (
                "higher local judge score than persistence with stable inference "
                "under the two-minute budget"
            ),
            "risk": (
                "architecture reproduction may require a new executor and may not "
                "match official checkpoint tensor conventions"
            ),
            "minimal experiment": (
                "Design a bounded FNO architecture executor or map the source to "
                "an existing checkpoint-family adapter before sandbox execution."
            ),
        }
    raise LiteratureAlgorithmLabError(
        "parsed markdown does not contain structured algorithm fields or a supported inference pattern"
    )


def _algorithm_card_v2_fields(
    *,
    fields: dict[str, str],
    hook: str,
    markdown: str,
    parse: dict[str, Any],
    parse_path: Path,
    markdown_path: Path,
    section_title: str | None,
    block_index: int,
    source_shape: str,
) -> dict[str, Any]:
    normalized_hook = _normalize_algorithm_hook(hook)
    defaults = HOOK_TO_ALGORITHM_CARD_V2.get(
        normalized_hook,
        _infer_algorithm_card_v2_defaults(markdown),
    )
    provenance = {
        "source_id": _nonempty_text(parse.get("source_id"), "source_id"),
        "paper_parse_path": str(parse_path),
        "markdown_path": str(markdown_path),
        "extractor": "heuristic_algorithm_card_v2",
        "normalized_hook": normalized_hook,
        "section_index": parse.get("section_index", []),
        "source_shape": source_shape,
        "block_index": block_index,
    }
    if section_title:
        provenance["section_title"] = section_title
    return {
        "operator_target": _algorithm_card_v2_enum_or_default(
            fields.get("operator target") or fields.get("operator_target"),
            allowed=ALGORITHM_CARD_V2_OPERATOR_TARGETS,
            default=str(defaults["operator_target"]),
        ),
        "mechanism_type": _algorithm_card_v2_enum_or_default(
            fields.get("mechanism type") or fields.get("mechanism_type"),
            allowed=ALGORITHM_CARD_V2_MECHANISM_TYPES,
            default=str(defaults["mechanism_type"]),
        ),
        "mathematical_object": _nonempty_or_default(
            fields.get("mathematical object") or fields.get("mathematical_object"),
            str(defaults["mathematical_object"]),
        ),
        "implementation_requirements": _implementation_requirements_from_fields(
            fields=fields,
            defaults=defaults["implementation_requirements"],
        ),
        "provenance": provenance,
    }


def _infer_algorithm_card_v2_defaults(markdown: str) -> dict[str, Any]:
    lowered = markdown.lower()
    if "fourier" in lowered:
        return HOOK_TO_ALGORITHM_CARD_V2["fourier_neural_operator_architecture"]
    if "residual" in lowered:
        return HOOK_TO_ALGORITHM_CARD_V2["learned_residual_correction"]
    if "spectral" in lowered or "viscosity" in lowered:
        return HOOK_TO_ALGORITHM_CARD_V2["spectral_rollout_stabilization"]
    if "physics" in lowered or "pinn" in lowered:
        return HOOK_TO_ALGORITHM_CARD_V2["physics_residual_regularization"]
    return {
        "operator_target": "ic_to_trajectory",
        "mechanism_type": "hybrid",
        "mathematical_object": "operator_mapping",
        "implementation_requirements": {
            "requires_training": True,
            "requires_checkpoint": False,
            "requires_target_data": True,
            "requires_new_executor": True,
            "requires_external_code": False,
            "can_run_as_postprocess": False,
        },
    }


def _algorithm_card_v2_enum_or_default(
    value: str | None,
    *,
    allowed: set[str],
    default: str,
) -> str:
    if isinstance(value, str) and value.strip() in allowed:
        return value.strip()
    return default


def _nonempty_or_default(value: str | None, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _implementation_requirements_from_fields(
    *,
    fields: dict[str, str],
    defaults: Any,
) -> dict[str, bool]:
    if not isinstance(defaults, dict):
        raise LiteratureAlgorithmLabError(
            "algorithm card v2 defaults missing implementation requirements"
        )
    requirements = {
        key: bool(defaults.get(key))
        for key in sorted(ALGORITHM_CARD_V2_IMPLEMENTATION_REQUIREMENTS)
    }
    field_aliases = {
        "requires_training": ["requires training", "requires_training"],
        "requires_checkpoint": ["requires checkpoint", "requires_checkpoint"],
        "requires_target_data": ["requires target data", "requires_target_data"],
        "requires_new_executor": ["requires new executor", "requires_new_executor"],
        "requires_external_code": ["requires external code", "requires_external_code"],
        "can_run_as_postprocess": [
            "can run as postprocess",
            "can_run_as_postprocess",
        ],
    }
    for key, aliases in field_aliases.items():
        for alias in aliases:
            if alias in fields:
                requirements[key] = _parse_bool_field(fields[alias], alias)
                break
    return requirements


def _parse_bool_field(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "y", "1"}:
        return True
    if normalized in {"false", "no", "n", "0"}:
        return False
    raise LiteratureAlgorithmLabError(f"{label} must be a boolean-like value")


def _source_type(value: Any) -> str:
    text = _nonempty_text(value, "source_type")
    if text not in VALID_SOURCE_TYPES:
        raise LiteratureAlgorithmLabError(f"unsupported source_type: {text}")
    return text


def _source_ids_from_source_artifact(payload: dict[str, Any]) -> set[Any]:
    if payload.get("state_hint") == SOURCE_DOCUMENT_STATE_HINT:
        record = payload.get("source_record")
        if isinstance(record, dict):
            return {record.get("source_id")}
        return set()
    return {
        record.get("source_id")
        for record in payload.get("source_records", [])
        if isinstance(record, dict)
    }


def _write_source_search_report(
    *,
    provider: str,
    retrieval_query: str,
    candidates: list[dict[str, Any]],
    output_file: Path,
) -> dict[str, Any]:
    if not candidates:
        raise LiteratureAlgorithmLabError("source search produced no candidates")
    report = {
        "schema_version": 1,
        "state_hint": SOURCE_SEARCH_STATE_HINT,
        "provider": provider,
        "retrieval_query": retrieval_query,
        "source_candidate_count": len(candidates),
        "source_candidates": candidates,
        "source_candidates_path": str(output_file),
        "next_command": "download-source-document",
    }
    _write_json(output_file, report)
    return report


class MinerUHttpClient:
    def post_json(
        self,
        endpoint: str,
        payload: dict[str, object],
        token: str | None,
    ) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def get_json(self, endpoint: str, token: str | None) -> dict[str, Any]:
        headers = {"Accept": "*/*"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(endpoint, headers=headers, method="GET")
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def download_bytes(self, url: str) -> bytes:
        return _download_bytes(url)


def _mineru_mode(value: Any) -> str:
    mode = _nonempty_text(value, "mineru_mode")
    if mode not in {"agent_url", "precise_url"}:
        raise LiteratureAlgorithmLabError(f"unsupported MinerU mode: {mode}")
    return mode


def _mineru_data(response: dict[str, Any]) -> dict[str, Any]:
    if response.get("code") != 0:
        raise LiteratureAlgorithmLabError(f"MinerU API error: {response.get('msg')}")
    data = response.get("data")
    if not isinstance(data, dict):
        raise LiteratureAlgorithmLabError("MinerU response missing data object")
    return data


def _find_by_id(value: Any, selected_id: str, id_key: str) -> dict[str, Any]:
    for item in _object_list(value, "items"):
        if item.get(id_key) == selected_id:
            return item
    raise LiteratureAlgorithmLabError(f"{id_key} not found: {selected_id}")


def _extension_for_url(url: str) -> str:
    lowered = url.lower().split("?", 1)[0].rstrip("/")
    for extension in [".pdf", ".zip", ".md", ".html", ".docx", ".doc", ".pptx", ".ppt"]:
        if lowered.endswith(extension):
            return extension
    if "/pdf/" in lowered or "arxiv.org/pdf" in lowered:
        return ".pdf"
    return ".bin"


def _download_bytes(url: str) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "pde-agent-literature-algorithm-lab/0.1"},
    )
    with urlopen(request, timeout=60) as response:
        return response.read()


def _source_candidate(
    *,
    source_id: str,
    source_type: str,
    title: str,
    url: str,
    document_url: str,
    retrieval_query: str,
    discovery_provider: str,
    license_note: str,
    external_ids: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_type": source_type,
        "title": title,
        "url": url,
        "document_url": document_url,
        "retrieval_query": retrieval_query,
        "discovery_provider": discovery_provider,
        "license_note": license_note,
        "external_ids": {
            key: value for key, value in external_ids.items() if value is not None
        },
    }


def _validate_algorithm_card_v2(card: Any, *, index: int) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    base = f"algorithm_cards[{index}]"
    if not isinstance(card, dict):
        _append_algorithm_card_v2_issue(
            issues,
            code="invalid_algorithm_card",
            path=base,
            failure_code="candidate_schema_invalid",
            message="algorithm card must be an object",
        )
        return issues
    for field in (
        "algorithm_card_id",
        "claim",
        "mathematical_object",
        "operation",
        "minimal_experiment",
    ):
        failure_code = (
            "no_algorithmic_claim" if field == "operation" else "candidate_schema_invalid"
        )
        if not _is_nonempty_string(card.get(field)):
            _append_algorithm_card_v2_issue(
                issues,
                code=f"missing_{field}",
                path=f"{base}.{field}",
                failure_code=failure_code,
                message=f"{field} must be a non-empty string",
            )
    for field in (
        "source_ids",
        "implementation_hooks",
        "preconditions",
        "expected_observables",
        "failure_modes",
    ):
        value = card.get(field)
        if not isinstance(value, list) or not value:
            _append_algorithm_card_v2_issue(
                issues,
                code=f"empty_{field}",
                path=f"{base}.{field}",
                failure_code="candidate_schema_invalid",
                message=f"{field} must be a non-empty list of strings",
            )
            continue
        for item_index, item in enumerate(value):
            if not _is_nonempty_string(item):
                _append_algorithm_card_v2_issue(
                    issues,
                    code=f"invalid_{field}_item",
                    path=f"{base}.{field}[{item_index}]",
                    failure_code="candidate_schema_invalid",
                    message=f"{field} items must be non-empty strings",
                )
    _validate_algorithm_card_v2_enum(
        issues,
        value=card.get("operator_target"),
        allowed=ALGORITHM_CARD_V2_OPERATOR_TARGETS,
        code="invalid_operator_target",
        path=f"{base}.operator_target",
    )
    _validate_algorithm_card_v2_enum(
        issues,
        value=card.get("mechanism_type"),
        allowed=ALGORITHM_CARD_V2_MECHANISM_TYPES,
        code="invalid_mechanism_type",
        path=f"{base}.mechanism_type",
    )
    _validate_algorithm_card_v2_enum(
        issues,
        value=card.get("evidence_status"),
        allowed=ALGORITHM_CARD_V2_EVIDENCE_STATUSES,
        code="invalid_evidence_status",
        path=f"{base}.evidence_status",
    )
    requirements = card.get("implementation_requirements")
    if not isinstance(requirements, dict):
        _append_algorithm_card_v2_issue(
            issues,
            code="missing_implementation_requirements",
            path=f"{base}.implementation_requirements",
            failure_code="candidate_schema_invalid",
            message="implementation_requirements must be an object",
        )
    else:
        for key in sorted(ALGORITHM_CARD_V2_IMPLEMENTATION_REQUIREMENTS):
            if key not in requirements:
                _append_algorithm_card_v2_issue(
                    issues,
                    code="missing_implementation_requirement",
                    path=f"{base}.implementation_requirements.{key}",
                    failure_code="candidate_schema_invalid",
                    message=f"implementation requirement missing: {key}",
                )
            elif not isinstance(requirements[key], bool):
                _append_algorithm_card_v2_issue(
                    issues,
                    code="implementation_requirement_not_bool",
                    path=f"{base}.implementation_requirements.{key}",
                    failure_code="candidate_schema_invalid",
                    message=f"implementation requirement must be boolean: {key}",
                )
    provenance = card.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        _append_algorithm_card_v2_issue(
            issues,
            code="missing_provenance",
            path=f"{base}.provenance",
            failure_code="candidate_schema_invalid",
            message="provenance must be a non-empty object",
        )
    return issues


def _validate_algorithm_card_v2_enum(
    issues: list[dict[str, Any]],
    *,
    value: Any,
    allowed: set[str],
    code: str,
    path: str,
) -> None:
    if not isinstance(value, str) or value not in allowed:
        _append_algorithm_card_v2_issue(
            issues,
            code=code,
            path=path,
            failure_code="candidate_schema_invalid",
            message=f"{path} must be one of {sorted(allowed)}",
        )


def _append_algorithm_card_v2_issue(
    issues: list[dict[str, Any]],
    *,
    code: str,
    path: str,
    failure_code: str,
    message: str,
) -> None:
    issues.append(
        {
            "code": code,
            "path": path,
            "failure_code": failure_code,
            "message": message,
        }
    )


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize_executable_blueprint(
    blueprint: Any,
    *,
    index: int,
    issues: list[dict[str, Any]],
) -> dict[str, Any] | None:
    base = f"blueprints[{index}]"
    if not isinstance(blueprint, dict):
        _append_blueprint_issue(
            issues,
            code="invalid_blueprint",
            path=base,
            failure_code="blueprint_schema_invalid",
            message="blueprint must be an object",
        )
        return None
    if _has_native_executable_blueprint_shape(blueprint):
        return dict(blueprint)
    blueprint_id = blueprint.get("blueprint_id")
    candidate_family = blueprint.get("candidate_family")
    sandbox = blueprint.get("sandbox") if isinstance(blueprint.get("sandbox"), dict) else {}
    parameter_grid = (
        blueprint.get("parameter_grid")
        if isinstance(blueprint.get("parameter_grid"), dict)
        else {}
    )
    executor_id = sandbox.get("allowed_executor")
    source_candidate_id = blueprint.get("source_candidate_id") or blueprint.get(
        "hypothesis_id"
    )
    baseline = _blueprint_baseline_for_family(str(candidate_family or ""))
    transforms = _blueprint_transforms_for_family(
        family=str(candidate_family or ""),
        parameter_grid=parameter_grid,
    )
    return {
        "blueprint_id": blueprint_id,
        "source_candidate_id": source_candidate_id,
        "candidate_family": candidate_family,
        "executor_id": executor_id,
        "baseline": baseline,
        "transforms": transforms,
        "training_plan": {
            "mode": "finetune"
            if candidate_family == "official_checkpoint_calibration"
            else "none"
        },
        "parameter_grid": parameter_grid,
        "metrics": _blueprint_metrics_from_sandbox(sandbox),
        "safety_policy": {
            "preserve_context_steps": 10,
            "forbid_target_leakage": True,
            "forbid_external_code": True,
        },
        "expected_artifacts": _blueprint_expected_artifacts(str(candidate_family or "")),
        "source_blueprint_shape": "hypothesis_lab_legacy",
    }


def _materialize_executable_blueprints_from_candidates(
    *,
    candidates: dict[str, Any],
    output_path: Path,
    force: bool,
) -> dict[str, Any]:
    _guard_output(output_path, force=force)
    blueprints = [
        _executable_blueprint_from_candidate(candidate)
        for candidate in _object_list(
            candidates.get("algorithm_candidates"),
            "algorithm_candidates",
        )
    ]
    report = {
        "schema_version": 1,
        "state_hint": "EXECUTABLE_BLUEPRINTS_DESIGNED",
        "executable_blueprints_path": str(output_path),
        "blueprint_count": len(blueprints),
        "blueprints": blueprints,
        "next_command": "validate-executable-blueprints",
    }
    _write_json(output_path, report)
    return report


def _executable_blueprint_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_id = _nonempty_text(
        candidate.get("algorithm_candidate_id"),
        "algorithm_candidate_id",
    )
    family = str(candidate.get("candidate_family") or "unknown_family")
    executor_id = _executor_id_for_candidate(candidate)
    parameter_grid = (
        candidate.get("parameter_grid")
        if isinstance(candidate.get("parameter_grid"), dict)
        and candidate.get("parameter_grid")
        else {"enabled": [True]}
    )
    return {
        "blueprint_id": f"bp_{_slug(candidate_id)}_001",
        "source_candidate_id": candidate_id,
        "candidate_family": family,
        "executor_id": executor_id,
        "baseline": _blueprint_baseline_for_family(family),
        "transforms": _blueprint_transforms_for_family(
            family=_blueprint_transform_family_alias(family),
            parameter_grid=parameter_grid,
        ),
        "training_plan": {
            "mode": "finetune" if family == "official_checkpoint_family" else "none"
        },
        "parameter_grid": parameter_grid,
        "metrics": ["score", "runtime", "burgers_residual"],
        "safety_policy": {
            "preserve_context_steps": 10,
            "forbid_target_leakage": True,
            "forbid_external_code": True,
        },
        "expected_artifacts": _blueprint_expected_artifacts(family),
    }


def _executor_id_for_candidate(candidate: dict[str, Any]) -> str:
    family = str(candidate.get("candidate_family") or "")
    if family == "official_checkpoint_family":
        return "official_checkpoint_candidate_pipeline"
    if family == "lightweight_residual_corrector":
        return "controlled_numpy_residual_operator"
    if family in {"task1_retrieval_prior"}:
        return "controlled_numpy_retrieval_prior"
    if family in {"task1_retrieval_refinement"}:
        return "controlled_numpy_retrieval_refinement"
    return "controlled_numpy_operator"


def _blueprint_transform_family_alias(family: str) -> str:
    if family == "lightweight_residual_corrector":
        return "learned_residual_correction"
    if family == "official_checkpoint_family":
        return "official_checkpoint_calibration"
    if family in {
        "spectral_rollout_stabilization",
        "operator_correction_postprocess",
        "physics_residual_regularization",
        "scheduled_rollout_training",
    }:
        return "linear_trend_spectral"
    return family


def _first_candidate_id(candidates: dict[str, Any]) -> str:
    items = _object_list(candidates.get("algorithm_candidates"), "algorithm_candidates")
    if not items:
        raise LiteratureAlgorithmLabError("algorithm candidates must be non-empty")
    return _nonempty_text(
        items[0].get("algorithm_candidate_id"),
        "algorithm_candidate_id",
    )


def _has_native_executable_blueprint_shape(blueprint: dict[str, Any]) -> bool:
    return all(
        key in blueprint
        for key in [
            "source_candidate_id",
            "executor_id",
            "baseline",
            "transforms",
            "training_plan",
            "metrics",
            "safety_policy",
            "expected_artifacts",
        ]
    )


def _blueprint_baseline_for_family(family: str) -> str:
    if family in {"task1_retrieval_prior", "task1_retrieval_refinement"}:
        return "retrieval_prior"
    if family == "official_checkpoint_calibration":
        return "fno_checkpoint"
    return "persistence"


def _blueprint_transforms_for_family(
    *, family: str, parameter_grid: dict[str, Any]
) -> list[dict[str, Any]]:
    if family == "learned_residual_correction":
        return [{"name": "residual_fit", "params": parameter_grid}]
    if family == "task1_retrieval_prior":
        return [{"name": "retrieval_prior", "params": parameter_grid}]
    if family == "task1_retrieval_refinement":
        return [{"name": "retrieval_refinement", "params": parameter_grid}]
    if family == "official_checkpoint_calibration":
        return [{"name": "checkpoint_calibration", "params": parameter_grid}]
    transforms = [{"name": "linear_trend", "params": parameter_grid}]
    if "spectral_strength" in parameter_grid:
        transforms.append({"name": "spectral_smooth", "params": parameter_grid})
    return transforms


def _blueprint_metrics_from_sandbox(sandbox: dict[str, Any]) -> list[str]:
    metrics = ["score", "runtime"]
    primary = sandbox.get("primary_metric")
    if isinstance(primary, str) and primary.strip() and primary not in metrics:
        metrics.append(primary)
    return metrics


def _blueprint_expected_artifacts(family: str) -> list[str]:
    artifacts = ["sandbox_evidence"]
    if family == "official_checkpoint_calibration":
        artifacts.append("checkpoint_sandbox_report")
    else:
        artifacts.append("prediction_hdf5")
    return artifacts


def _validate_executable_blueprint(
    blueprint: dict[str, Any],
    *,
    index: int,
    issues: list[dict[str, Any]],
) -> None:
    base = f"blueprints[{index}]"
    for field in [
        "blueprint_id",
        "source_candidate_id",
        "candidate_family",
        "executor_id",
        "baseline",
    ]:
        if not _is_nonempty_string(blueprint.get(field)):
            _append_blueprint_issue(
                issues,
                code=f"missing_{field}",
                path=f"{base}.{field}",
                failure_code="blueprint_schema_invalid",
                message=f"{field} must be a non-empty string",
            )
    baseline = blueprint.get("baseline")
    if isinstance(baseline, str) and baseline not in EXECUTABLE_BLUEPRINT_BASELINES:
        _append_blueprint_issue(
            issues,
            code="invalid_baseline",
            path=f"{base}.baseline",
            failure_code="blueprint_schema_invalid",
            message=f"baseline must be one of {sorted(EXECUTABLE_BLUEPRINT_BASELINES)}",
        )
    executor_id = blueprint.get("executor_id")
    if isinstance(executor_id, str) and executor_id not in EXECUTABLE_BLUEPRINT_EXECUTORS:
        _append_blueprint_issue(
            issues,
            code="invalid_executor",
            path=f"{base}.executor_id",
            failure_code="unsupported_executor",
            message=f"executor_id is not approved: {executor_id}",
        )
    _validate_blueprint_transforms(blueprint.get("transforms"), base=base, issues=issues)
    _validate_blueprint_training_plan(
        blueprint.get("training_plan"), base=base, issues=issues
    )
    if not isinstance(blueprint.get("parameter_grid"), dict) or not blueprint.get(
        "parameter_grid"
    ):
        _append_blueprint_issue(
            issues,
            code="empty_parameter_grid",
            path=f"{base}.parameter_grid",
            failure_code="blueprint_schema_invalid",
            message="parameter_grid must be a non-empty object",
        )
    for field in ["metrics", "expected_artifacts"]:
        value = blueprint.get(field)
        if not isinstance(value, list) or not value:
            _append_blueprint_issue(
                issues,
                code=f"empty_{field}",
                path=f"{base}.{field}",
                failure_code="blueprint_schema_invalid",
                message=f"{field} must be a non-empty list",
            )
        elif not all(_is_nonempty_string(item) for item in value):
            _append_blueprint_issue(
                issues,
                code=f"invalid_{field}",
                path=f"{base}.{field}",
                failure_code="blueprint_schema_invalid",
                message=f"{field} items must be non-empty strings",
            )
    _validate_blueprint_safety_policy(
        blueprint.get("safety_policy"), base=base, issues=issues
    )


def _validate_blueprint_transforms(
    transforms: Any,
    *,
    base: str,
    issues: list[dict[str, Any]],
) -> None:
    if not isinstance(transforms, list) or not transforms:
        _append_blueprint_issue(
            issues,
            code="empty_transforms",
            path=f"{base}.transforms",
            failure_code="blueprint_schema_invalid",
            message="transforms must be a non-empty list",
        )
        return
    for item_index, transform in enumerate(transforms):
        path = f"{base}.transforms[{item_index}]"
        if not isinstance(transform, dict):
            _append_blueprint_issue(
                issues,
                code="invalid_transform",
                path=path,
                failure_code="blueprint_schema_invalid",
                message="transform must be an object",
            )
            continue
        name = transform.get("name")
        if not isinstance(name, str) or name not in EXECUTABLE_BLUEPRINT_TRANSFORMS:
            _append_blueprint_issue(
                issues,
                code="invalid_transform_name",
                path=f"{path}.name",
                failure_code="blueprint_schema_invalid",
                message=f"transform name is not approved: {name}",
            )
        if not isinstance(transform.get("params"), dict):
            _append_blueprint_issue(
                issues,
                code="invalid_transform_params",
                path=f"{path}.params",
                failure_code="blueprint_schema_invalid",
                message="transform params must be an object",
            )


def _validate_blueprint_training_plan(
    training_plan: Any,
    *,
    base: str,
    issues: list[dict[str, Any]],
) -> None:
    if not isinstance(training_plan, dict):
        _append_blueprint_issue(
            issues,
            code="invalid_training_plan",
            path=f"{base}.training_plan",
            failure_code="blueprint_schema_invalid",
            message="training_plan must be an object",
        )
        return
    mode = training_plan.get("mode")
    if not isinstance(mode, str) or mode not in EXECUTABLE_BLUEPRINT_TRAINING_MODES:
        _append_blueprint_issue(
            issues,
            code="invalid_training_mode",
            path=f"{base}.training_plan.mode",
            failure_code="blueprint_schema_invalid",
            message=f"training mode must be one of {sorted(EXECUTABLE_BLUEPRINT_TRAINING_MODES)}",
        )


def _validate_blueprint_safety_policy(
    safety_policy: Any,
    *,
    base: str,
    issues: list[dict[str, Any]],
) -> None:
    if not isinstance(safety_policy, dict):
        _append_blueprint_issue(
            issues,
            code="invalid_safety_policy",
            path=f"{base}.safety_policy",
            failure_code="blueprint_schema_invalid",
            message="safety_policy must be an object",
        )
        return
    preserve_steps = safety_policy.get("preserve_context_steps")
    if not isinstance(preserve_steps, int) or preserve_steps < 10:
        _append_blueprint_issue(
            issues,
            code="unsafe_context_preservation",
            path=f"{base}.safety_policy.preserve_context_steps",
            failure_code="context_mismatch",
            message="preserve_context_steps must be at least 10",
        )
    if safety_policy.get("forbid_target_leakage") is not True:
        _append_blueprint_issue(
            issues,
            code="unsafe_target_leakage",
            path=f"{base}.safety_policy.forbid_target_leakage",
            failure_code="target_leakage_risk",
            message="forbid_target_leakage must be true",
        )
    if safety_policy.get("forbid_external_code") is not True:
        _append_blueprint_issue(
            issues,
            code="unsafe_external_code",
            path=f"{base}.safety_policy.forbid_external_code",
            failure_code="requires_external_code",
            message="forbid_external_code must be true",
        )


def _append_blueprint_issue(
    issues: list[dict[str, Any]],
    *,
    code: str,
    path: str,
    failure_code: str,
    message: str,
) -> None:
    issues.append(
        {
            "code": code,
            "path": path,
            "failure_code": failure_code,
            "message": message,
        }
    )


def _find_algorithm_candidate(
    candidates_report: dict[str, Any], algorithm_candidate_id: str
) -> dict[str, Any]:
    for candidate in candidates_report.get("algorithm_candidates", []):
        if (
            isinstance(candidate, dict)
            and candidate.get("algorithm_candidate_id") == algorithm_candidate_id
        ):
            return candidate
    raise LiteratureAlgorithmLabError(
        f"algorithm candidate not found: {algorithm_candidate_id}"
    )


def _select_best_candidate_evidence(
    *, evidence_records: list[dict[str, Any]], candidate_id: str
) -> dict[str, Any]:
    matching = [
        evidence
        for evidence in evidence_records
        if evidence.get("candidate_id") == candidate_id
    ]
    if not matching:
        raise LiteratureAlgorithmLabError(
            f"sandbox evidence does not reference candidate: {candidate_id}"
        )
    return max(matching, key=lambda item: _float_or_default(item.get("score"), -1e30))


def _evaluate_candidate_promotion_checks(
    *,
    candidate: dict[str, Any],
    evidence: dict[str, Any],
    current_best_score: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    passed: dict[str, Any] = {}
    failed: dict[str, Any] = {}
    criteria = candidate.get("promotion_criteria")
    if not isinstance(criteria, dict):
        criteria = {}
    min_score_delta = _float_or_default(criteria.get("min_score_delta"), 0.0)
    max_inference_seconds = _float_or_default(
        criteria.get("max_inference_seconds"), 120.0
    )
    requires_ablation = bool(criteria.get("requires_ablation", True))

    executor = candidate.get("executor_spec")
    cli_command = executor.get("cli_command") if isinstance(executor, dict) else None
    if (
        candidate.get("candidate_family") == "unsupported"
        or cli_command == "blocked_requires_executor_design"
    ):
        failed["executor"] = _promotion_check(
            failure_code="unsupported_executor",
            message="candidate cannot promote without an executable sandbox",
        )
    else:
        passed["executor"] = {"passed": True, "cli_command": cli_command}

    if evidence.get("rules_passed") is True:
        passed["validity"] = {"passed": True}
    else:
        failed["validity"] = _promotion_check(
            failure_code="sandbox_rule_failed",
            message="sandbox rules did not pass",
        )

    score_delta = _evidence_score_delta(evidence, current_best_score)
    if score_delta >= min_score_delta:
        passed["score"] = {
            "passed": True,
            "score_delta": score_delta,
            "threshold": min_score_delta,
        }
    else:
        failed["score"] = _promotion_check(
            failure_code="score_regressed",
            message="score delta did not meet promotion threshold",
            score_delta=score_delta,
            threshold=min_score_delta,
        )

    runtime_seconds = _evidence_runtime_seconds(evidence)
    if runtime_seconds <= max_inference_seconds:
        passed["runtime"] = {
            "passed": True,
            "inference_seconds": runtime_seconds,
            "threshold": max_inference_seconds,
        }
    else:
        failed["runtime"] = _promotion_check(
            failure_code="runtime_budget_exceeded",
            message="inference runtime exceeded candidate budget",
            inference_seconds=runtime_seconds,
            threshold=max_inference_seconds,
        )

    physics_delta = _evidence_physics_delta(evidence)
    if physics_delta <= 0.0:
        passed["physics"] = {"passed": True, "residual_delta": physics_delta}
    else:
        failed["physics"] = _promotion_check(
            failure_code="physics_regressed",
            message="physics diagnostic regressed",
            residual_delta=physics_delta,
        )

    ablation = evidence.get("ablation_metrics")
    ablation_delta = (
        _float_or_default(ablation.get("score_delta"), 0.0)
        if isinstance(ablation, dict)
        else 0.0
    )
    if not requires_ablation or ablation_delta > 0.0:
        passed["ablation"] = {
            "passed": True,
            "required": requires_ablation,
            "score_delta": ablation_delta,
        }
    else:
        failed["ablation"] = _promotion_check(
            failure_code="ablation_missing_or_weak",
            message="required ablation evidence did not support candidate",
            score_delta=ablation_delta,
        )

    reproducibility = evidence.get("reproducibility")
    deterministic = (
        bool(reproducibility.get("deterministic"))
        if isinstance(reproducibility, dict)
        else False
    )
    if deterministic:
        passed["reproducibility"] = {"passed": True}
    else:
        failed["reproducibility"] = _promotion_check(
            failure_code="insufficient_reproducibility",
            message="evidence is not marked deterministic",
        )
    return passed, failed


def _candidate_promotion_decision(
    *,
    candidate: dict[str, Any],
    failed_checks: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    if "executor" in failed_checks:
        return "blocked"
    hard_reject = {"validity", "runtime"}
    if hard_reject.intersection(failed_checks):
        return "reject"
    if "score" in failed_checks:
        return "deprioritize" if evidence.get("rules_passed") is True else "reject"
    if failed_checks:
        return "refine"
    return "promote"


def _candidate_promotion_next_command(decision: str) -> str:
    if decision == "promote":
        return "record-strategy-proposal"
    if decision == "refine":
        return "refine-algorithm-candidate"
    if decision == "blocked":
        return "create-executor-design-request"
    return "record-literature-registry-event"


def _candidate_promotion_metric_results(
    *, evidence: dict[str, Any], current_best_score: float
) -> dict[str, Any]:
    return {
        "score": evidence.get("score"),
        "current_best_score": current_best_score,
        "score_delta_vs_current_best": _evidence_score_delta(
            evidence, current_best_score
        ),
        "inference_seconds": _evidence_runtime_seconds(evidence),
        "physics_residual_delta": _evidence_physics_delta(evidence),
        "rules_passed": evidence.get("rules_passed"),
    }


def _promotion_check(
    *, failure_code: str, message: str, **extra: Any
) -> dict[str, Any]:
    result = {"passed": False, "failure_code": failure_code, "message": message}
    result.update(extra)
    return result


def _evidence_score_delta(evidence: dict[str, Any], current_best_score: float) -> float:
    if isinstance(evidence.get("score_delta_vs_current_best"), int | float):
        return float(evidence["score_delta_vs_current_best"])
    score = _float_or_default(evidence.get("score"), current_best_score)
    return score - current_best_score


def _evidence_runtime_seconds(evidence: dict[str, Any]) -> float:
    runtime = evidence.get("runtime")
    if isinstance(runtime, dict):
        return _float_or_default(runtime.get("inference_seconds"), 1e30)
    return _float_or_default(runtime, 1e30)


def _evidence_physics_delta(evidence: dict[str, Any]) -> float:
    physics = evidence.get("physics_metrics")
    if isinstance(physics, dict):
        return _float_or_default(physics.get("residual_delta"), 0.0)
    return 0.0


def _float_or_default(value: Any, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def _object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise LiteratureAlgorithmLabError(f"{label} must be a list")
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise LiteratureAlgorithmLabError(f"{label} items must be objects")
        result.append(item)
    return result


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise LiteratureAlgorithmLabError("expected at least one non-empty text value")


def _xml_text(element: ET.Element, path: str, namespace: dict[str, str]) -> str | None:
    child = element.find(path, namespace)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _dedupe_id(base: str, seen: set[str]) -> str:
    if base not in seen:
        return base
    index = 2
    while f"{base}_{index}" in seen:
        index += 1
    return f"{base}_{index}"


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return text[:80] or "source"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LiteratureAlgorithmLabError(f"JSON artifact must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _guard_output(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise LiteratureAlgorithmLabError(f"output file already exists: {path}")


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise LiteratureAlgorithmLabError(f"{label} not found: {path}")
    return path


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LiteratureAlgorithmLabError(f"{label} must be a non-empty string")
    return value.strip()


def _text_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise LiteratureAlgorithmLabError(f"{label} must be a list")
    result = []
    for item in value:
        result.append(_nonempty_text(item, label))
    return result
