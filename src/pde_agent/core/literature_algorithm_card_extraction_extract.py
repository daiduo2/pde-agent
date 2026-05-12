"""Literature algorithm card extraction extract functions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from pde_agent.core.literature_algorithm_common import LiteratureAlgorithmLabError
from pde_agent.core.literature_algorithm_card_extraction_helpers import (
    _guard_output,
    _nonempty_text,
    _read_json,
    _require_file,
    _write_json,
)

ALGORITHM_CARD_V2_MECHANISM_TYPES = {
    "architecture",
    "loss",
    "rollout",
    "postprocess",
    "conditioning",
    "checkpoint",
    "hybrid",
}
ALGORITHM_CARDS_STATE_HINT = "ALGORITHM_CARDS_EXTRACTED"
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
PAPER_PARSE_STATE_HINT = "MINERU_PARSE_INGESTED"
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
ALGORITHM_CARD_V2_IMPLEMENTATION_REQUIREMENTS = {
    "requires_training",
    "requires_checkpoint",
    "requires_target_data",
    "requires_new_executor",
    "requires_external_code",
    "can_run_as_postprocess",
}

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
    extracted_source_ids: list[str] = []
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
        if source_id not in extracted_source_ids:
            extracted_source_ids.append(source_id)
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
                try:
                    fields = _infer_algorithm_fields(block_markdown)
                except LiteratureAlgorithmLabError:
                    extraction_audit["skipped_blocks"].append(
                        {
                            "source_id": source_id,
                            "section_title": field_block.get("section_title"),
                            "block_index": field_block.get("block_index"),
                            "line_start": field_block.get("line_start"),
                            "line_end": field_block.get("line_end"),
                            "reason_code": "no_supported_inference_pattern",
                            "missing_fields": _missing_algorithm_card_fields(fields),
                        }
                    )
                    continue
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
    extraction_status = "extracted" if cards else "empty"
    report = {
        "schema_version": 1,
        "state_hint": ALGORITHM_CARDS_STATE_HINT,
        "algorithm_cards_path": str(output_file),
        "algorithm_card_count": len(cards),
        "source_ids": extracted_source_ids,
        "algorithm_cards": cards,
        "extraction_status": extraction_status,
        "extraction_audit": extraction_audit,
        "next_command": "compile-algorithm-candidates"
        if cards
        else "update-literature-registry",
    }
    if not cards:
        report["failure_code"] = "no_algorithmic_claim"
    _write_json(output_file, report)
    return report

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

def _algorithm_card_v2_enum_or_default(
    value: str | None,
    *,
    allowed: set[str],
    default: str,
) -> str:
    if isinstance(value, str) and value.strip() in allowed:
        return value.strip()
    return default

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

def _first_line_matching(
    text_lines: list[tuple[int, str]],
    keywords: list[str],
) -> tuple[int, str] | None:
    for local_line_number, text in text_lines:
        lowered = text.lower()
        if any(keyword in lowered for keyword in keywords):
            return local_line_number, text
    return text_lines[0] if text_lines else None

def _has_complete_algorithm_card_fields(fields: dict[str, str]) -> bool:
    return not _missing_algorithm_card_fields(fields)

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

def _infer_nonstructured_algorithm_field_block(
    block: dict[str, Any],
) -> dict[str, Any] | None:
    section_title = (
        block["section_title"] if isinstance(block.get("section_title"), str) else ""
    )
    if _is_low_signal_algorithm_section(section_title):
        return None
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
    if _looks_like_fno_architecture_section(lowered):
        claim_line = _first_line_matching(
            text_lines,
            ["fourier neural operator", "infinite-dimensional", "discretization"],
        )
        mechanism_line = _first_line_matching(
            text_lines,
            ["kernel integral", "fourier space", "fourier transform"],
        )
        operation_line = _first_line_matching(
            text_lines,
            ["truncates", "low-frequency", "learned weights", "transforms back"],
        )
        fields = _infer_algorithm_fields("Fourier Neural Operator")
        spans = {
            "claim": _line_span_from_match(claim_line, fallback_text=joined),
            "mechanism": _line_span_from_match(
                mechanism_line or claim_line,
                fallback_text=joined,
            ),
            "operation": _line_span_from_match(
                operation_line or mechanism_line or claim_line,
                fallback_text=joined,
            ),
        }
        return fields, spans
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

def _is_low_signal_algorithm_section(section_title: str) -> bool:
    lowered = section_title.lower()
    return any(
        marker in lowered
        for marker in [
            "abstract",
            "introduction",
            "references",
            "acknowledgement",
            "discussion",
            "conclusion",
        ]
    )

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

def _looks_like_fno_architecture_section(lowered: str) -> bool:
    return (
        "fourier neural operator" in lowered
        and any(
            phrase in lowered
            for phrase in [
                "kernel integral",
                "fourier space",
                "fourier layer",
                "fourier transform",
            ]
        )
    )

def _looks_like_physics_residual_section(lowered: str) -> bool:
    return (
        ("physics" in lowered or "pde" in lowered or "burgers" in lowered)
        and "residual" in lowered
        and any(keyword in lowered for keyword in ["loss", "regular", "penalty"])
    )

def _looks_like_residual_correction_section(lowered: str) -> bool:
    return "residual" in lowered and any(
        keyword in lowered for keyword in ["correct", "bias", "postprocess", "fit"]
    )

def _looks_like_scheduled_rollout_section(lowered: str) -> bool:
    return "rollout" in lowered and any(
        keyword in lowered
        for keyword in ["scheduled", "teacher", "autoregressive", "multi-step"]
    )

def _looks_like_spectral_rollout_section(lowered: str) -> bool:
    return (
        ("spectral" in lowered or "fourier" in lowered)
        and any(keyword in lowered for keyword in ["damp", "filter", "viscosity"])
        and any(keyword in lowered for keyword in ["rollout", "predict", "horizon"])
    )

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

def _missing_algorithm_card_fields(fields: dict[str, str]) -> list[str]:
    return [
        key
        for key in ["claim", "mechanism", "operation", "hook"]
        if key not in fields
    ]

def _non_heading_lines(markdown: str) -> list[tuple[int, str]]:
    lines = []
    for local_line_number, line in enumerate(markdown.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append((local_line_number, stripped))
    return lines

def _nonempty_or_default(value: str | None, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default

def _normalize_algorithm_hook(hook: str) -> str:
    normalized = hook.strip()
    return HOOK_ALIASES.get(normalized, normalized)

def _parse_bool_field(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "y", "1"}:
        return True
    if normalized in {"false", "no", "n", "0"}:
        return False
    raise LiteratureAlgorithmLabError(f"{label} must be a boolean-like value")

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

def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return text[:80] or "source"
