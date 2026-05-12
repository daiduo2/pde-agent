"""Local strategy-skill registry for Hermes strategy reuse."""


import json
from pathlib import Path
from typing import Any


REGISTRY_STATE_HINT = "STRATEGY_SKILL_REGISTRY_UPDATED"
ENTRY_STATE_HINT = "STRATEGY_SKILL_REGISTERED"
LOOKUP_STATE_HINT = "STRATEGY_SKILL_LOOKUP_COMPLETED"


class StrategySkillRegistryError(RuntimeError):
    pass


def register_strategy_skill(
    *,
    install_manifest_path: str | Path,
    evidence_summary_path: str | Path,
    promotion_report_path: str | Path,
    registry_path: str | Path,
    source: str,
) -> dict[str, Any]:
    install_file = Path(install_manifest_path)
    evidence_file = Path(evidence_summary_path)
    promotion_file = Path(promotion_report_path)
    registry_file = Path(registry_path)
    install = _read_json(install_file, "skill install manifest")
    evidence = _read_json(evidence_file, "strategy evidence")
    promotion = _read_json(promotion_file, "promotion report")
    _validate_inputs(install=install, evidence=evidence, promotion=promotion)

    entry = {
        "state_hint": ENTRY_STATE_HINT,
        "strategy_id": install["strategy_id"],
        "skill_name": install["skill_name"],
        "installed_skill_dir": install["installed_skill_dir"],
        "installed_skill_md_path": install["installed_skill_md_path"],
        "skill_install_manifest_path": str(install_file),
        "evidence_summary_path": str(evidence_file),
        "promotion_report_path": str(promotion_file),
        "source": source,
        "model_family": evidence.get("model_family"),
        "source_state_hint": evidence.get("source_state_hint"),
        "metrics": {
            "evaluation_window_count": evidence.get("evaluation_window_count"),
            "mean_score_delta": evidence.get("mean_score_delta"),
            "score_delta_std": evidence.get("score_delta_std"),
            "failure_rate": evidence.get("failure_rate"),
        },
        "routing": {
            "applicable_pde": "1D Burgers",
            "preferred_when": [
                "real-HDF5 Burgers rollout evidence is required",
                "spectral residual strategy has positive multi-window evidence",
                "window-stratified variance gate passes promotion",
            ],
        },
    }
    registry = _read_registry(registry_file)
    entries = [
        item
        for item in registry["entries"]
        if not (
            item.get("strategy_id") == entry["strategy_id"]
            and item.get("skill_name") == entry["skill_name"]
        )
    ]
    entries.append(entry)
    updated = {
        "schema_version": 1,
        "state_hint": REGISTRY_STATE_HINT,
        "strategy_registry_path": str(registry_file),
        "entry_count": len(entries),
        "entries": entries,
        "next_command": "strategy-growth-complete",
    }
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(
        json.dumps(updated, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "state_hint": ENTRY_STATE_HINT,
        "strategy_id": entry["strategy_id"],
        "skill_name": entry["skill_name"],
        "registry_path": str(registry_file),
        "registered_entry": entry,
        "next_command": "strategy-growth-complete",
    }


def lookup_strategy_skills(
    *,
    registry_path: str | Path,
    applicable_pde: str,
    model_family: str,
    max_results: int,
    output_path: str | Path | None,
) -> dict[str, Any]:
    if max_results < 1:
        raise StrategySkillRegistryError("max_results must be positive")
    registry_file = Path(registry_path)
    registry = _read_registry(registry_file)
    matches = []
    for entry in registry["entries"]:
        if not isinstance(entry, dict):
            continue
        score = _selection_score(
            entry=entry,
            applicable_pde=applicable_pde,
            model_family=model_family,
        )
        if score <= 0:
            continue
        row = dict(entry)
        row["selection_score"] = score
        matches.append(row)
    matches.sort(
        key=lambda item: (
            -float(item["selection_score"]),
            float(item.get("metrics", {}).get("score_delta_std", 999.0)),
            -float(item.get("metrics", {}).get("mean_score_delta", 0.0)),
        )
    )
    result = {
        "schema_version": 1,
        "state_hint": LOOKUP_STATE_HINT,
        "strategy_registry_path": str(registry_file),
        "lookup": {
            "applicable_pde": applicable_pde,
            "model_family": model_family,
            "max_results": max_results,
        },
        "match_count": min(len(matches), max_results),
        "matches": matches[:max_results],
        "next_command": "use-registered-strategy-skill",
    }
    if output_path is not None:
        output_file = Path(output_path)
        result["lookup_report_path"] = str(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def _selection_score(
    *,
    entry: dict[str, Any],
    applicable_pde: str,
    model_family: str,
) -> int:
    score = 0
    if entry.get("model_family") == model_family:
        score += 2
    routing = entry.get("routing")
    if isinstance(routing, dict) and routing.get("applicable_pde") == applicable_pde:
        score += 2
    metrics = entry.get("metrics")
    if isinstance(metrics, dict):
        if metrics.get("failure_rate") == 0.0:
            score += 1
        std = metrics.get("score_delta_std")
        if isinstance(std, (int, float)) and std <= 2.0:
            score += 1
        mean_delta = metrics.get("mean_score_delta")
        if isinstance(mean_delta, (int, float)) and mean_delta > 0.0:
            score += 1
    return score


def _validate_inputs(
    *,
    install: dict[str, Any],
    evidence: dict[str, Any],
    promotion: dict[str, Any],
) -> None:
    if install.get("state_hint") != "SKILL_INSTALLED":
        raise StrategySkillRegistryError("install manifest has invalid state_hint")
    if evidence.get("state_hint") != "STRATEGY_EVIDENCE_SUMMARY_BUILT":
        raise StrategySkillRegistryError("evidence summary has invalid state_hint")
    if promotion.get("state_hint") != "STRATEGY_PROMOTION_EVALUATED":
        raise StrategySkillRegistryError("promotion report has invalid state_hint")
    if promotion.get("decision") != "promote" or not promotion.get(
        "promote_to_skill_candidate"
    ):
        raise StrategySkillRegistryError("promotion report did not promote strategy")
    strategy_id = install.get("strategy_id")
    if not isinstance(strategy_id, str) or not strategy_id:
        raise StrategySkillRegistryError("install manifest must include strategy_id")
    if promotion.get("strategy_id") != strategy_id:
        raise StrategySkillRegistryError("strategy_id mismatch across artifacts")
    if evidence.get("best_strategy_id") != strategy_id:
        raise StrategySkillRegistryError("evidence best_strategy_id mismatch")
    skill_name = install.get("skill_name")
    if not isinstance(skill_name, str) or not skill_name:
        raise StrategySkillRegistryError("install manifest must include skill_name")
    skill_md = install.get("installed_skill_md_path")
    if not isinstance(skill_md, str) or not Path(skill_md).is_file():
        raise StrategySkillRegistryError("installed skill markdown is missing")


def _read_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "state_hint": REGISTRY_STATE_HINT,
            "strategy_registry_path": str(path),
            "entry_count": 0,
            "entries": [],
        }
    registry = _read_json(path, "strategy skill registry")
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise StrategySkillRegistryError("registry entries must be a list")
    return registry


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise StrategySkillRegistryError(f"missing {label} path: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise StrategySkillRegistryError(f"{label} must be a JSON object")
    return data
