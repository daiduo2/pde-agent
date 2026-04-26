# FNO Iteration Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `run-fno-iteration`, a one-command single-window FNO loop that runs dry-run planning, audit, all V1 candidates, evaluation, comparison, and writes an iteration report.

**Architecture:** Reuse the existing core functions instead of duplicating scientific logic. Add a small orchestration module that owns artifact layout and report synthesis, a command wrapper that converts it to the standard JSON payload, and a Typer CLI entry.

**Tech Stack:** Python, Typer, pytest, torch-backed existing FNO executor.

---

## File Structure

- Create `src/pde_agent/core/fno_iteration_runner.py`: orchestration and report schema.
- Create `src/pde_agent/commands/run_fno_iteration.py`: standard JSON payload wrapper.
- Modify `src/pde_agent/cli.py`: import wrapper and expose `run-fno-iteration`.
- Create `tests/test_fno_iteration_runner.py`: core and CLI behavior tests.
- Modify `README.md`: document the new default Hermes entrypoint.

## Task 1: Core Runner

**Files:**
- Create: `src/pde_agent/core/fno_iteration_runner.py`
- Test: `tests/test_fno_iteration_runner.py`

- [ ] **Step 1: Write failing core test**

Create `tests/test_fno_iteration_runner.py` with tiny official-FNO fixtures and:

```python
def test_run_fno_iteration_writes_report_for_all_candidates(tmp_path: Path) -> None:
    windows_path = tmp_path / "windows.npz"
    checkpoint_path = tmp_path / "official_fno.pt"
    output_dir = tmp_path / "iteration"
    _write_windows(windows_path)
    _write_checkpoint(checkpoint_path)

    report = run_fno_iteration(
        windows_path=windows_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        max_train_steps=1,
        learning_rate=1e-4,
        nu=0.001,
        dt=1.0,
        dx=1.0,
        force=False,
    )

    assert report["state_hint"] == "FNO_ITERATION_COMPLETED"
    assert report["candidate_count"] == 2
    assert report["winner"]["experiment_id"]
    assert report["artifacts"]["iteration_report_path"] == str(output_dir / "fno_iteration_report.json")
    assert Path(report["artifacts"]["comparison_report_path"]).exists()
    assert all(candidate["evaluation_report_path"] for candidate in report["candidates"])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_iteration_runner.py::test_run_fno_iteration_writes_report_for_all_candidates -q
```

Expected: fail because `pde_agent.core.fno_iteration_runner` does not exist.

- [ ] **Step 3: Implement core runner**

Implement `run_fno_iteration(...)` that:

1. validates input files,
2. rejects non-empty `output_dir` unless `force=True`,
3. calls `run_fno_iteration_dryrun`,
4. audits `experiment_config.json`,
5. prepares and executes every approved candidate,
6. evaluates each candidate against the same windows and baseline score,
7. compares candidate evaluation reports,
8. writes `fno_iteration_report.json`.

- [ ] **Step 4: Run test to verify it passes**

Run the same focused test. Expected: pass.

- [ ] **Step 5: Add overwrite guard test**

Add:

```python
def test_run_fno_iteration_rejects_non_empty_output_without_force(tmp_path: Path) -> None:
    output_dir = tmp_path / "iteration"
    output_dir.mkdir()
    existing = output_dir / "keep.txt"
    existing.write_text("previous", encoding="utf-8")

    with pytest.raises(FNOIterationRunnerError, match="output_dir already contains files"):
        run_fno_iteration(...)

    assert existing.read_text(encoding="utf-8") == "previous"
```

Run the focused test. Expected: pass after implementation.

- [ ] **Step 6: Commit**

```bash
git add src/pde_agent/core/fno_iteration_runner.py tests/test_fno_iteration_runner.py
git commit -m "feat: add fno iteration runner core"
```

## Task 2: CLI Wrapper

**Files:**
- Create: `src/pde_agent/commands/run_fno_iteration.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_fno_iteration_runner.py`

- [ ] **Step 1: Write failing CLI test**

Add a test invoking `run-fno-iteration` with the same tiny fixtures. Assert:

```python
assert payload["command"] == "run-fno-iteration"
assert payload["state_hint"] == "FNO_ITERATION_COMPLETED"
assert payload["metrics"]["candidate_count"] == 2
assert payload["metrics"]["winner_experiment_id"]
assert payload["artifacts"]["iteration_report_path"] == str(output_dir / "fno_iteration_report.json")
assert payload["next_hint"] == "run-fno-iteration"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_iteration_runner.py::test_run_fno_iteration_cli_writes_payload -q
```

Expected: fail because CLI command is not registered.

- [ ] **Step 3: Implement command wrapper and CLI entry**

Create `run_fno_iteration_command(...)` using `success_payload`. Add Typer
command options:

- `--windows-path`
- `--checkpoint-path`
- `--output-dir`
- `--max-train-steps`, default `16`
- `--learning-rate`, default `1e-4`
- `--nu`, default `0.001`
- `--dt`, default `1.0`
- `--dx`, default `1.0`
- `--force`

- [ ] **Step 4: Run CLI test to verify it passes**

Run the focused CLI test. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/pde_agent/commands/run_fno_iteration.py src/pde_agent/cli.py tests/test_fno_iteration_runner.py
git commit -m "feat: expose fno iteration runner cli"
```

## Task 3: Documentation And Smoke

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Add `run-fno-iteration` as the recommended single-window Hermes entrypoint
after `run-fno-iteration-dryrun`, and keep `run-fno-evidence-sweep` as the
broader validation path.

- [ ] **Step 2: Run verification**

Run:

```bash
uv run --frozen --extra torch pytest tests/test_fno_iteration_runner.py -q
uv run --frozen --extra torch pytest tests -q
git diff --check
```

- [ ] **Step 3: Run real smoke**

Run:

```bash
uv run --frozen --extra torch pde-agent run-fno-iteration \
  --windows-path /tmp/pde-agent-window-suite-task1-public/windows_f031.npz \
  --checkpoint-path data/pdebench/checkpoints/burgers_FNO.tar \
  --output-dir /tmp/pde-agent-fno-iteration-runner-smoke-20260426-001 \
  --max-train-steps 1 \
  --learning-rate 1e-4 \
  --force
```

Expected: JSON payload with `state_hint = FNO_ITERATION_COMPLETED`,
`candidate_count = 2`, and a non-empty `winner_experiment_id`.

- [ ] **Step 4: Commit docs**

```bash
git add README.md
git commit -m "docs: document fno iteration runner"
```
