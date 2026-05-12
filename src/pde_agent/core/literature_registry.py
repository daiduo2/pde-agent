"""Literature source and algorithm registry state management."""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pde_agent.core.literature_algorithm_common import LiteratureAlgorithmLabError

SOURCE_RECORDS_STATE_HINT = "LITERATURE_SOURCE_RECORDS_BUILT"
SOURCE_DOCUMENT_STATE_HINT = "SOURCE_DOCUMENT_DOWNLOADED"
PAPER_PARSE_STATE_HINT = "MINERU_PARSE_INGESTED"
ALGORITHM_CARDS_STATE_HINT = "ALGORITHM_CARDS_EXTRACTED"
ALGORITHM_CANDIDATES_STATE_HINT = "ALGORITHM_CANDIDATES_COMPILED"
ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT = "ALGORITHM_CANDIDATE_PROMOTION_EVALUATED"
LITERATURE_REGISTRY_EVENT_STATE_HINT = "LITERATURE_REGISTRY_EVENT_RECORDED"
FNO_CANDIDATE_ADAPTED_STATE_HINT = "FNO_ARCHITECTURE_CANDIDATE_ADAPTED"
BLOCKED_CANDIDATE_HANDOFF_STATE_HINT = "BLOCKED_CANDIDATE_HANDOFF_BUILT"
LITERATURE_REGISTRY_UPDATED_STATE_HINT = "LITERATURE_REGISTRY_UPDATED"

VALID_SOURCE_TYPES = {
    "openalex_work",
    "arxiv_paper",
    "github_repo",
    "competition_resource",
    "manual_source",
}

FAILURE_TAXONOMY = {
    "source_download_failed",
    "parse_failed",
    "parse_quality_insufficient",
    "no_algorithmic_claim",
    "unsupported_executor",
    "data_contract_incompatible",
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
        cards_value = payload.get("algorithm_cards")
        card_count = len(cards_value) if isinstance(cards_value, list) else int(
            payload.get("algorithm_card_count", 0)
        )
        if card_count == 0:
            source_ids = _source_ids_from_algorithm_cards_report(payload)
            for source_id in source_ids:
                source = _upsert_registry_source(
                    registry,
                    {
                        "source_id": source_id,
                        "source_type": "unknown",
                        "title": source_id,
                    },
                )
                source["algorithm_extraction_status"] = "empty"
                source["algorithm_extraction_failure_code"] = payload.get(
                    "failure_code",
                    "no_algorithmic_claim",
                )
                source["algorithm_extraction_audit"] = payload.get(
                    "extraction_audit",
                    {},
                )
            if source_ids:
                _registry_event(
                    registry,
                    artifact_file=artifact_file,
                    event_type="algorithm_extraction_empty",
                    source_ids=source_ids,
                    reason_codes=[
                        str(payload.get("failure_code", "no_algorithmic_claim"))
                    ],
                )
            continue
        for card in _object_list(cards_value, "algorithm_cards"):
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
    mutation_evidence = candidate.get("mutation_evidence")
    if isinstance(mutation_evidence, dict):
        target["mutation_evidence"] = dict(mutation_evidence)
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
        candidate_family = payload.get("candidate_family")
        if isinstance(candidate_family, str) and candidate_family.strip():
            candidate["candidate_family"] = candidate_family.strip()
        mutation_evidence = payload.get("mutation_evidence")
        if isinstance(mutation_evidence, dict):
            candidate["mutation_evidence"] = dict(mutation_evidence)
        failure_codes = _normalize_failure_codes(payload.get("failure_codes", []))
        if failure_codes:
            failure_entry = {
                "artifact_path": str(artifact_file),
                "failure_codes": failure_codes,
                "recommended_next": payload.get("recommended_next"),
                "retry_allowed": bool(payload.get("retry_allowed")),
            }
            metric_results = payload.get("metric_results")
            if isinstance(metric_results, dict) and metric_results:
                failure_entry["metric_results"] = metric_results
            _append_unique_object(
                candidate,
                "failure_history",
                failure_entry,
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

def _source_type(value: Any) -> str:
    text = _nonempty_text(value, "source_type")
    if text not in VALID_SOURCE_TYPES:
        raise LiteratureAlgorithmLabError(f"unsupported source_type: {text}")
    return text


def _source_ids_from_algorithm_cards_report(payload: dict[str, Any]) -> list[str]:
    source_ids = []
    for source_id in payload.get("source_ids", []):
        if isinstance(source_id, str) and source_id.strip():
            source_ids.append(source_id)
    if source_ids:
        return sorted(set(source_ids))
    audit = payload.get("extraction_audit")
    if not isinstance(audit, dict):
        return []
    for key in ["skipped_blocks", "skipped_parses"]:
        value = audit.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            source_id = item.get("source_id")
            if isinstance(source_id, str) and source_id.strip():
                source_ids.append(source_id)
    return sorted(set(source_ids))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise LiteratureAlgorithmLabError(f"{label} must be a list")
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise LiteratureAlgorithmLabError(f"{label} entries must be objects")
        result.append(item)
    return result
