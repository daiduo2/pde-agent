"""Agent-facing PDE literature extraction skill task scaffold."""


import json
from pathlib import Path
from typing import Any


STATE_HINT = "PDE_LITERATURE_EXTRACTION_SKILL_PREPARED"


class PdeLiteratureExtractionSkillError(RuntimeError):
    pass


def prepare_pde_literature_extraction_skill(
    *,
    literature_extraction_context_path: str | Path,
    output_path: str | Path,
    operator_family: str,
    target_problem: str,
    force: bool,
) -> dict[str, Any]:
    output_file = Path(output_path)
    _guard_output(output_file, force=force)
    context = _read_json(Path(literature_extraction_context_path))
    if context.get("state_hint") != "LITERATURE_EXTRACTION_CONTEXT_PREPARED":
        raise PdeLiteratureExtractionSkillError(
            "literature extraction context has invalid state_hint"
        )
    source_id = _text(context.get("source_id"), "source_id")
    units = context.get("context_units")
    if not isinstance(units, list) or not units:
        raise PdeLiteratureExtractionSkillError("context_units must be a non-empty list")
    selected_units = [
        unit for unit in units if isinstance(unit, dict) and _unit_text(unit)
    ][:8]
    if not selected_units:
        raise PdeLiteratureExtractionSkillError("no usable context units found")
    prompt = _agent_prompt(
        source_id=source_id,
        operator_family=_text(operator_family, "operator_family"),
        target_problem=_text(target_problem, "target_problem"),
        selected_units=selected_units,
    )
    task = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "source_id": source_id,
        "operator_family": operator_family,
        "target_problem": target_problem,
        "literature_extraction_context_path": str(literature_extraction_context_path),
        "selected_context_unit_count": len(selected_units),
        "agent_contract": {
            "role": "pde_literature_algorithm_extractor",
            "output_state_hint": "ALGORITHM_CARDS_EXTRACTED",
            "required_outputs": [
                "algorithm_cards_v2",
                "candidate_mechanism_summary",
                "evidence_hooks",
                "sandbox_observables",
                "failure_risks",
            ],
            "forbidden_outputs": [
                "invented benchmark results",
                "unbounded code execution",
                "claims without source_unit_ids",
            ],
            "candidate_novelty_policy": {
                "new_family": "Use normal AlgorithmCard fields for a new mechanism family.",
                "same_family_retry": (
                    "Include mutation_evidence when the card is a deliberate variant "
                    "of a previously failed or deprioritized candidate family."
                ),
            },
            "optional_card_fields": {
                "mutation_evidence": {
                    "required_when_present": [
                        "parent_candidate_family",
                        "mutation_type",
                        "reason",
                    ],
                    "use_when": (
                        "Only when the algorithm is intended to retry a known "
                        "candidate family with a concrete algorithmic change."
                    ),
                }
            },
        },
        "agent_prompt": prompt,
        "next_command": "agent-extract-algorithm-cards-v2",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(task, indent=2, sort_keys=True), encoding="utf-8")
    return task


def _agent_prompt(
    *,
    source_id: str,
    operator_family: str,
    target_problem: str,
    selected_units: list[dict[str, Any]],
) -> str:
    unit_lines = []
    for unit in selected_units:
        unit_id = unit.get("unit_id") or unit.get("id") or "unknown_unit"
        heading = unit.get("heading") or unit.get("title") or "Untitled"
        text = _unit_text(unit)
        unit_lines.append(f"### {unit_id}: {heading}\n{text}")
    units_text = "\n\n".join(unit_lines)
    return (
        "You are a PDE literature algorithm extraction skill running inside an "
        "agent-controlled research workflow.\n\n"
        f"Source: {source_id}\n"
        f"Target problem: {target_problem}\n"
        f"Operator family focus: {operator_family}\n\n"
        "Task: read the provided minerU-derived context units and extract "
        "AlgorithmCard v2 candidates that could become immediately testable PDE "
        "algorithm blueprints.\n\n"
        "Hard rules:\n"
        "- Do not invent results, datasets, equations, or implementation details.\n"
        "- Every algorithmic claim must cite source_unit_ids from the context.\n"
        "- Prefer mechanisms that can be tested in a bounded sandbox.\n"
        "- Separate architecture, loss, rollout, postprocess, conditioning, and "
        "checkpoint mechanisms.\n"
        "- Classify each candidate as either a new algorithm family or a same-family "
        "retry of a failed/deprioritized family.\n"
        "- For a same-family retry, include mutation_evidence with "
        "parent_candidate_family, mutation_type, and reason; mutation_type must "
        "name the concrete algorithmic change.\n"
        "- If the paper gives only a vague idea, emit a blocked candidate with "
        "clear missing evidence instead of pretending it is executable.\n\n"
        "Return a JSON object with state_hint ALGORITHM_CARDS_EXTRACTED and "
        "AlgorithmCard v2 entries. Each card must include operator_target, "
        "mechanism_type, mathematical_object, implementation_requirements, "
        "evidence_status, source_unit_ids, expected_observables, and known_risks.\n\n"
        "Context units:\n\n"
        f"{units_text}"
    )


def _unit_text(unit: dict[str, Any]) -> str:
    value = unit.get("text") or unit.get("content") or unit.get("summary")
    if not isinstance(value, str):
        return ""
    return value.strip()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PdeLiteratureExtractionSkillError(f"{name} must be a non-empty string")
    return value.strip()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PdeLiteratureExtractionSkillError(f"input does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PdeLiteratureExtractionSkillError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise PdeLiteratureExtractionSkillError("input must be a JSON object")
    return payload


def _guard_output(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise PdeLiteratureExtractionSkillError(f"output already exists: {path}")
