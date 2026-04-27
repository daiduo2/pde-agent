from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_strategy_growth_audit_script_parses_run_state_commands() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        "audit_hermes_strategy_growth",
        repo_root / "scripts" / "audit_hermes_strategy_growth.py",
    )
    run_state = """
required_reads:
  - /repo/run_state.yaml
  - /repo/bootstrap_prompt.md
required_artifacts:
  - /tmp/out/strategy_evidence_summary.json
  - /tmp/out/strategy_ledger.jsonl
allowed_commands:
  - name: convert_fno_evidence_summary
    command: cd /repo && uv run --frozen pde-agent convert-fno-evidence-summary --sweep-summary-path /tmp/fno.json --output-path /tmp/out/strategy_evidence_summary.json --long-window-threshold 31 --force
  - name: append_strategy_ledger
    command: cd /repo && uv run --frozen pde-agent append-strategy-ledger --proposal-path /tmp/out/strategy_proposal.json --rubric-path /tmp/out/experiment_rubric.json --evidence-summary-path /tmp/out/strategy_evidence_summary.json --promotion-report-path /tmp/out/promotion_report.json --ledger-path /tmp/out/strategy_ledger.jsonl
success_transition:
  on_complete: STRATEGY_LEDGER_APPENDED
"""

    assert module._yaml_list(run_state, "required_artifacts") == [
        "/tmp/out/strategy_evidence_summary.json",
        "/tmp/out/strategy_ledger.jsonl",
    ]
    assert module._yaml_command_list(run_state) == [
        "cd /repo && uv run --frozen pde-agent convert-fno-evidence-summary --sweep-summary-path /tmp/fno.json --output-path /tmp/out/strategy_evidence_summary.json --long-window-threshold 31 --force",
        "cd /repo && uv run --frozen pde-agent append-strategy-ledger --proposal-path /tmp/out/strategy_proposal.json --rubric-path /tmp/out/experiment_rubric.json --evidence-summary-path /tmp/out/strategy_evidence_summary.json --promotion-report-path /tmp/out/promotion_report.json --ledger-path /tmp/out/strategy_ledger.jsonl",
    ]


def test_strategy_growth_audit_script_derives_arm_from_fixture_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        "audit_hermes_strategy_growth",
        repo_root / "scripts" / "audit_hermes_strategy_growth.py",
    )

    assert (
        module._arm_from_fixture_root(
            repo_root / "integration_tests" / "hermes_deeponet_strategy_growth"
        )
        == "deeponet_strategy_growth"
    )


def test_run_script_renders_strategy_growth_prompt() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        "run_hermes_strategy_growth",
        repo_root / "scripts" / "run_hermes_strategy_growth.py",
    )
    fixture_root = repo_root / "integration_tests" / "hermes_fno_strategy_growth"

    prompt = module._render_prompt(fixture_root)

    assert str(fixture_root / "run_state.yaml") in prompt
    assert str(fixture_root / "bootstrap_prompt.md") in prompt
    assert str(fixture_root / "task_spec.json") in prompt
    assert "STRATEGY_LEDGER_APPENDED" in prompt


def test_run_script_accepts_protocol_complete_terminal_message() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        "run_hermes_strategy_growth",
        repo_root / "scripts" / "run_hermes_strategy_growth.py",
    )

    output = """
Protocol complete. Final command returned `STRATEGY_LEDGER_APPENDED`.
Summary:
- append_strategy_ledger: appended ledger entry.
"""

    assert module._success_transition_seen(output) is True
