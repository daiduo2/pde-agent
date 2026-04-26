# Strategy Growth Loop V0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic strategy-growth gate that formalizes agent-generated PDE strategies, assigns evaluation rubrics, evaluates promotion evidence, and records attempts in an append-only ledger.

**Architecture:** Add a model-family-agnostic `strategy_growth` core module plus thin CLI wrappers. V0 does not search the web, patch code, or create skills; it only creates auditable JSON/JSONL artifacts that future retrieval and skill-creation fixtures must pass through.

**Tech Stack:** Python 3.14, Typer CLI, JSON/JSONL artifacts, pytest, existing `success_payload` protocol.

---

## File Structure

- Create `src/pde_agent/core/strategy_growth.py`: validation, artifact creation, promotion evaluation, ledger append.
- Create `src/pde_agent/commands/record_strategy_proposal.py`: CLI payload wrapper for proposal artifacts.
- Create `src/pde_agent/commands/build_strategy_rubric.py`: CLI payload wrapper for rubric artifacts.
- Create `src/pde_agent/commands/evaluate_strategy_promotion.py`: CLI payload wrapper for promotion reports.
- Create `src/pde_agent/commands/append_strategy_ledger.py`: CLI payload wrapper for ledger rows.
- Modify `src/pde_agent/cli.py`: expose four commands.
- Add `tests/test_strategy_growth.py`: core and CLI behavior coverage.
- Update `docs/hermes_integration_discipline.md`: document that retrieval/skill growth must pass through this gate.

---

### Task 1: Strategy Proposal Artifact

**Files:**
- Create: `src/pde_agent/core/strategy_growth.py`
- Create: `src/pde_agent/commands/record_strategy_proposal.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_strategy_growth.py`

- [ ] **Step 1: Write failing tests**

Add tests that define the desired API:

```python
def test_record_strategy_proposal_writes_valid_artifact(tmp_path: Path) -> None:
    from pde_agent.core.strategy_growth import record_strategy_proposal

    output_path = tmp_path / "strategy_proposal.json"
    report = record_strategy_proposal(
        strategy_id="fno_spectral_filter_v0",
        title="Spectral filtering before rollout fine-tuning",
        source_type="agent_hypothesis",
        problem_context="1D Burgers fixed-nu short-window FNO rollout",
        hypothesis="Removing unstable high-frequency residual modes improves long-window stability.",
        mechanism="Apply a low-pass residual filter before fine-tuning rollout predictions.",
        implementation_plan={
            "model_family": "FNO",
            "allowed_commands": ["run-fno-iteration", "run-fno-evidence-sweep"],
            "expected_artifacts": ["candidate_comparison.json"],
        },
        expected_observables=[
            "positive mean score delta across windows",
            "lower high-frequency residual energy",
        ],
        known_risks=[
            "over-smoothing shocks",
            "single-window improvement may not generalize",
        ],
        output_path=output_path,
    )

    assert output_path.exists()
    assert report["state_hint"] == "STRATEGY_PROPOSAL_RECORDED"
    assert report["strategy_id"] == "fno_spectral_filter_v0"
    assert report["source_type"] == "agent_hypothesis"
    assert report["next_command"] == "build-strategy-rubric"
```

Also add:

```python
def test_record_strategy_proposal_rejects_invalid_source_type(tmp_path: Path) -> None:
    from pde_agent.core.strategy_growth import (
        StrategyGrowthError,
        record_strategy_proposal,
    )

    with pytest.raises(StrategyGrowthError, match="unsupported source_type"):
        record_strategy_proposal(
            strategy_id="bad_strategy",
            title="Bad",
            source_type="random_blog_copy",
            problem_context="context",
            hypothesis="hypothesis",
            mechanism="mechanism",
            implementation_plan={"model_family": "FNO"},
            expected_observables=["metric"],
            known_risks=["risk"],
            output_path=tmp_path / "proposal.json",
        )
```

Add CLI test:

```python
def test_record_strategy_proposal_cli_returns_protocol_payload(tmp_path: Path) -> None:
    from pde_agent.cli import app
    from typer.testing import CliRunner

    output_path = tmp_path / "strategy_proposal.json"
    result = CliRunner().invoke(
        app,
        [
            "record-strategy-proposal",
            "--strategy-id", "fno_spectral_filter_v0",
            "--title", "Spectral filtering before rollout fine-tuning",
            "--source-type", "agent_hypothesis",
            "--problem-context", "1D Burgers fixed-nu short-window FNO rollout",
            "--hypothesis", "Removing unstable high-frequency residual modes improves long-window stability.",
            "--mechanism", "Apply a low-pass residual filter before fine-tuning rollout predictions.",
            "--implementation-plan-json", '{"model_family":"FNO","allowed_commands":["run-fno-iteration"]}',
            "--expected-observable", "positive mean score delta across windows",
            "--known-risk", "over-smoothing shocks",
            "--output-path", str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["command"] == "record-strategy-proposal"
    assert payload["state_hint"] == "STRATEGY_PROPOSAL_RECORDED"
    assert payload["artifacts"]["strategy_proposal_path"] == str(output_path)
    assert payload["next_hint"] == "build-strategy-rubric"
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
uv run --frozen pytest tests/test_strategy_growth.py -q
```

Expected: fail because `pde_agent.core.strategy_growth` does not exist.

- [ ] **Step 3: Implement proposal recording**

Implement:

```python
VALID_SOURCE_TYPES = {"local_library", "external_retrieval", "agent_hypothesis"}

class StrategyGrowthError(ValueError):
    pass

def record_strategy_proposal(...):
    ...
```

Write JSON with:

- `schema_version: 1`
- `state_hint: STRATEGY_PROPOSAL_RECORDED`
- all supplied fields
- `strategy_proposal_path`
- `next_command: build-strategy-rubric`

Reject empty required strings, invalid source type, empty observables, empty
risks, non-dict implementation plan, and existing output path.

- [ ] **Step 4: Implement command and CLI**

`record_strategy_proposal_command` should wrap the core result with
`success_payload`, using:

- command: `record-strategy-proposal`
- state: `STRATEGY_PROPOSAL_RECORDED`
- artifacts: `strategy_proposal_path`
- metrics: `strategy_id`, `source_type`, `observable_count`, `risk_count`
- next hint: `build-strategy-rubric`

CLI should parse `--implementation-plan-json` with `json.loads`.

- [ ] **Step 5: Verify and commit**

Run:

```bash
uv run --frozen pytest tests/test_strategy_growth.py -q
git diff --check
git add src/pde_agent/core/strategy_growth.py src/pde_agent/commands/record_strategy_proposal.py src/pde_agent/cli.py tests/test_strategy_growth.py
git commit -m "feat: add strategy proposal recorder"
```

---

### Task 2: Experiment Rubric Artifact

**Files:**
- Modify: `src/pde_agent/core/strategy_growth.py`
- Create: `src/pde_agent/commands/build_strategy_rubric.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_strategy_growth.py`

- [ ] **Step 1: Write failing rubric tests**

Add:

```python
def test_build_strategy_rubric_requires_multi_window_evidence(tmp_path: Path) -> None:
    from pde_agent.core.strategy_growth import (
        build_strategy_rubric,
        record_strategy_proposal,
    )

    proposal_path = tmp_path / "strategy_proposal.json"
    rubric_path = tmp_path / "experiment_rubric.json"
    record_strategy_proposal(..., output_path=proposal_path)

    rubric = build_strategy_rubric(
        proposal_path=proposal_path,
        output_path=rubric_path,
        min_evaluation_windows=3,
        max_inference_time_regression=0.10,
    )

    assert rubric_path.exists()
    assert rubric["state_hint"] == "STRATEGY_RUBRIC_BUILT"
    assert rubric["strategy_id"] == "fno_spectral_filter_v0"
    assert rubric["promotion_thresholds"]["min_evaluation_windows"] == 3
    assert "mean_score_delta" in rubric["required_metrics"]
    assert "long_window_score_delta" in rubric["required_metrics"]
    assert "physics_residual_delta" in rubric["required_metrics"]
    assert rubric["next_command"] == "evaluate-strategy-promotion"
```

Add CLI payload test for `build-strategy-rubric`.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
uv run --frozen pytest tests/test_strategy_growth.py -q
```

Expected: fail because `build_strategy_rubric` does not exist.

- [ ] **Step 3: Implement rubric builder**

Read proposal JSON, validate `state_hint`, and write:

- `schema_version: 1`
- `state_hint: STRATEGY_RUBRIC_BUILT`
- `strategy_id`
- `strategy_proposal_path`
- `required_metrics`: `evaluation_window_count`, `mean_score_delta`,
  `score_delta_std`, `long_window_score_delta`, `physics_residual_delta`,
  `inference_time_regression`, `failure_rate`
- `promotion_thresholds`
- `next_command: evaluate-strategy-promotion`

Reject `min_evaluation_windows < 2` and negative inference-time regression
allowance.

- [ ] **Step 4: Implement command and CLI**

Command wrapper emits `success_payload`:

- command: `build-strategy-rubric`
- state: `STRATEGY_RUBRIC_BUILT`
- artifacts: `experiment_rubric_path`, `strategy_proposal_path`
- metrics: `strategy_id`, `min_evaluation_windows`, `required_metric_count`
- next hint: `evaluate-strategy-promotion`

- [ ] **Step 5: Verify and commit**

Run:

```bash
uv run --frozen pytest tests/test_strategy_growth.py -q
git diff --check
git add src/pde_agent/core/strategy_growth.py src/pde_agent/commands/build_strategy_rubric.py src/pde_agent/cli.py tests/test_strategy_growth.py
git commit -m "feat: add strategy experiment rubric"
```

---

### Task 3: Promotion Gate

**Files:**
- Modify: `src/pde_agent/core/strategy_growth.py`
- Create: `src/pde_agent/commands/evaluate_strategy_promotion.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_strategy_growth.py`

- [ ] **Step 1: Write failing promotion tests**

Add a passing evidence test:

```python
def test_evaluate_strategy_promotion_promotes_strong_evidence(tmp_path: Path) -> None:
    from pde_agent.core.strategy_growth import evaluate_strategy_promotion

    evidence_path = tmp_path / "evidence_summary.json"
    evidence_path.write_text(json.dumps({
        "evaluation_window_count": 4,
        "mean_score_delta": 2.5,
        "score_delta_std": 0.8,
        "long_window_score_delta": 1.1,
        "physics_residual_delta": -0.05,
        "inference_time_regression": 0.02,
        "failure_rate": 0.0
    }), encoding="utf-8")

    report = evaluate_strategy_promotion(
        proposal_path=proposal_path,
        rubric_path=rubric_path,
        evidence_summary_path=evidence_path,
        output_path=tmp_path / "promotion_report.json",
    )

    assert report["state_hint"] == "STRATEGY_PROMOTION_EVALUATED"
    assert report["decision"] == "promote"
    assert report["promote_to_skill_candidate"] is True
    assert report["next_command"] == "append-strategy-ledger"
```

Add a failing single-window test:

```python
def test_evaluate_strategy_promotion_rejects_single_window_evidence(tmp_path: Path) -> None:
    ...
    evidence_path.write_text(json.dumps({
        "evaluation_window_count": 1,
        "mean_score_delta": 5.0,
        "score_delta_std": 0.0,
        "long_window_score_delta": 0.0,
        "physics_residual_delta": 0.0,
        "inference_time_regression": 0.0,
        "failure_rate": 0.0
    }), encoding="utf-8")

    report = evaluate_strategy_promotion(...)

    assert report["decision"] == "reject"
    assert report["promote_to_skill_candidate"] is False
    assert "insufficient_evaluation_windows" in report["reason_codes"]
```

Add CLI payload test for `evaluate-strategy-promotion`.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
uv run --frozen pytest tests/test_strategy_growth.py -q
```

Expected: fail because `evaluate_strategy_promotion` does not exist.

- [ ] **Step 3: Implement promotion evaluation**

Pass/fail rules:

- reject if any required metric is missing,
- reject if `evaluation_window_count < min_evaluation_windows`,
- reject if `mean_score_delta < 0`,
- reject if `long_window_score_delta < 0`,
- reject if `failure_rate > 0`,
- reject if `inference_time_regression > max_inference_time_regression`,
- revise if only `physics_residual_delta > 0` or `score_delta_std > max_score_delta_std`,
- promote otherwise.

Use reason codes:

- `insufficient_evaluation_windows`
- `negative_mean_score_delta`
- `negative_long_window_score_delta`
- `nonzero_failure_rate`
- `inference_time_regression_too_high`
- `physics_residual_regressed`
- `score_delta_variance_too_high`
- `all_thresholds_passed`

- [ ] **Step 4: Implement command and CLI**

Command wrapper emits:

- command: `evaluate-strategy-promotion`
- state: `STRATEGY_PROMOTION_EVALUATED`
- artifacts: proposal, rubric, evidence, promotion report
- metrics: strategy id, decision, promote boolean, reason count
- next hint: `append-strategy-ledger`

- [ ] **Step 5: Verify and commit**

Run:

```bash
uv run --frozen pytest tests/test_strategy_growth.py -q
git diff --check
git add src/pde_agent/core/strategy_growth.py src/pde_agent/commands/evaluate_strategy_promotion.py src/pde_agent/cli.py tests/test_strategy_growth.py
git commit -m "feat: add strategy promotion gate"
```

---

### Task 4: Strategy Ledger And Docs

**Files:**
- Modify: `src/pde_agent/core/strategy_growth.py`
- Create: `src/pde_agent/commands/append_strategy_ledger.py`
- Modify: `src/pde_agent/cli.py`
- Modify: `docs/hermes_integration_discipline.md`
- Test: `tests/test_strategy_growth.py`

- [ ] **Step 1: Write failing ledger tests**

Add:

```python
def test_append_strategy_ledger_records_attempt(tmp_path: Path) -> None:
    from pde_agent.core.strategy_growth import append_strategy_ledger

    ledger_path = tmp_path / "strategy_ledger.jsonl"
    row = append_strategy_ledger(
        proposal_path=proposal_path,
        rubric_path=rubric_path,
        evidence_summary_path=evidence_path,
        promotion_report_path=promotion_path,
        ledger_path=ledger_path,
    )

    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == row
    assert row["state_hint"] == "STRATEGY_LEDGER_APPENDED"
    assert row["decision"] == "promote"
    assert row["promote_to_skill_candidate"] is True
```

Add CLI payload test for `append-strategy-ledger`.

- [ ] **Step 2: Verify tests fail**

Run:

```bash
uv run --frozen pytest tests/test_strategy_growth.py -q
```

Expected: fail because `append_strategy_ledger` does not exist.

- [ ] **Step 3: Implement ledger append**

Read all artifacts, validate their state hints, then append one JSON object with:

- `schema_version: 1`
- `state_hint: STRATEGY_LEDGER_APPENDED`
- `strategy_id`
- `proposal_path`
- `rubric_path`
- `evidence_summary_path`
- `promotion_report_path`
- `decision`
- `promote_to_skill_candidate`
- `reason_codes`
- `next_command`: `create-skill-candidate` if promoted, else `record-strategy-proposal`

- [ ] **Step 4: Implement command and CLI**

Command wrapper emits:

- command: `append-strategy-ledger`
- state: `STRATEGY_LEDGER_APPENDED`
- artifacts: `strategy_ledger_path`
- metrics: strategy id, decision, promote boolean
- next hint from ledger row.

- [ ] **Step 5: Update docs**

Add to `docs/hermes_integration_discipline.md`:

- retrieval and skill creation must not bypass strategy-growth artifacts,
- promotion does not equal automatic skill installation,
- failed strategies must remain in the ledger.

- [ ] **Step 6: Verify and commit**

Run:

```bash
uv run --frozen pytest tests/test_strategy_growth.py -q
uv run --frozen pytest tests -q
git diff --check
git add src/pde_agent/core/strategy_growth.py src/pde_agent/commands/append_strategy_ledger.py src/pde_agent/cli.py tests/test_strategy_growth.py docs/hermes_integration_discipline.md
git commit -m "feat: add strategy growth ledger"
```

---

## Final Verification

Run:

```bash
uv run --frozen pytest tests -q
git status --short
```

Expected:

- all tests pass,
- working tree is clean.
