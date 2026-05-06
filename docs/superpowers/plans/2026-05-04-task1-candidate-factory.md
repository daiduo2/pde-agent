# Task1 Candidate Factory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded Task1 candidate-family factory so Hermes can expand, sandbox, benchmark, and remember new strategy families after local candidate space saturates.

**Architecture:** Add a focused core module for `CandidateFamilyCard` validation, generation, sandbox planning, and capability evaluation. Expose the module through JSON-only CLI commands and the scientific tool catalog, then connect runtime expansion artifacts to the new factory reports without allowing arbitrary code execution.

**Tech Stack:** Python, Typer CLI, JSON artifacts, pytest, existing Hermes/Hypothesis Lab runtime conventions.

---

### Task 1: CandidateFamilyCard schema

**Files:**
- Create: `src/pde_agent/core/task1_candidate_factory.py`
- Test: `tests/test_task1_candidate_factory.py`

- [ ] Write tests for valid and invalid candidate cards.
- [ ] Implement validation helpers with stable state hints and clear errors.
- [ ] Verify with `uv run --frozen pytest -q tests/test_task1_candidate_factory.py`.

### Task 2: Candidate family generation

**Files:**
- Modify: `src/pde_agent/core/task1_candidate_factory.py`
- Test: `tests/test_task1_candidate_factory.py`

- [ ] Write tests that a stagnated Task1 capability benchmark produces diverse families.
- [ ] Implement bounded generation from benchmark/audit artifacts and prior registries.
- [ ] Verify generated cards validate and avoid exact prior IDs.

### Task 3: Candidate family sandbox and benchmark

**Files:**
- Modify: `src/pde_agent/core/task1_candidate_factory.py`
- Test: `tests/test_task1_candidate_factory.py`

- [ ] Write tests for sandbox planning and blocked unsupported hooks.
- [ ] Implement sandbox report generation without arbitrary code execution.
- [ ] Implement capability evaluation over family and sandbox reports.

### Task 4: CLI and tool catalog

**Files:**
- Create: `src/pde_agent/commands/task1_candidate_factory.py`
- Modify: `src/pde_agent/cli.py`
- Modify: `src/pde_agent/core/scientific_tool_router.py`
- Test: `tests/test_task1_candidate_factory.py`

- [ ] Add JSON-only command wrappers.
- [ ] Register `generate-task1-candidate-families`, `run-candidate-family-sandbox`, and `evaluate-candidate-factory-capability`.
- [ ] Verify CLI help and tool catalog visibility in tests.

### Task 5: Runtime handoff documentation

**Files:**
- Modify: `docs/hermes_hypothesis_lab_runbook.md`
- Test: full pytest

- [ ] Document candidate factory as the next command after stable candidate-space exhaustion.
- [ ] Run targeted tests.
- [ ] Run `uv run --frozen pytest -q`.

## Self-Review

- The plan maps directly to the approved five-step direction.
- No task requires internet, external LLM calls, arbitrary generated code execution, or hidden target access.
- The factory produces bounded artifacts that Hermes can use as agent-authored search surface.
