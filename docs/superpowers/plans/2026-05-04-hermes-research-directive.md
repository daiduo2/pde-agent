# Hermes Research Directive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a higher-level Hermes research guidance layer that turns recent evidence into auditable research directives, routes directives into runtime audit-chain execution, and evaluates whether Hermes followed the directive.

**Architecture:** Create a focused `hermes_research_directive` core module with three JSON artifacts: directive, runtime application evidence, and followthrough evaluation. Expose JSON-only CLI commands and scientific-tool catalog entries. Runtime integration should pass bounded directive fields into existing Hermes parameters instead of adding free-form execution.

**Tech Stack:** Python, Typer CLI, JSON artifacts, pytest, existing Hermes/Hypothesis Lab runtime conventions.

---

### Task 1: Directive Builder

**Files:**
- Create: `src/pde_agent/core/hermes_research_directive.py`
- Create: `src/pde_agent/commands/hermes_research_directive.py`
- Test: `tests/test_hermes_research_directive.py`

- [x] Write tests for building a directive from Task1 benchmark, candidate factory benchmark, and optional judge report paths.
- [x] Implement directive construction with bounded fields: bottleneck, objective, candidate generator, candidate options, constraints, experiment design, promotion gate, and expected trace tags.
- [x] Add CLI command `build-hermes-research-directive`.

### Task 2: Runtime Directive Ingestion

**Files:**
- Modify: `src/pde_agent/core/hermes_runtime_audit_chain.py`
- Modify: `src/pde_agent/core/hermes_runtime_closed_loop.py`
- Modify: `src/pde_agent/commands/hypothesis_lab_runtime.py`
- Modify: `src/pde_agent/cli.py`
- Test: `tests/test_hermes_research_directive.py`

- [x] Write tests proving `research_directive_path` overrides the runtime candidate generator, learning objective, and candidate options.
- [x] Implement directive loading and option merging without arbitrary execution.
- [x] Preserve existing behavior when no directive is provided.

### Task 3: Followthrough Evaluator

**Files:**
- Modify: `src/pde_agent/core/hermes_research_directive.py`
- Modify: `src/pde_agent/commands/hermes_research_directive.py`
- Modify: `src/pde_agent/cli.py`
- Modify: `src/pde_agent/core/scientific_tool_router.py`
- Test: `tests/test_hermes_research_directive.py`

- [x] Write tests for passing and failing followthrough.
- [x] Implement `evaluate-research-directive-followthrough`.
- [x] Register all new commands in the scientific tool catalog.

### Task 4: Verification

**Files:**
- Test only

- [x] Run targeted tests: `uv run --frozen pytest -q tests/test_hermes_research_directive.py tests/test_hypothesis_lab_runtime.py::test_hermes_runtime_closed_loop_forwards_task1_reference_to_hypothesis_runner`
- [x] Run full tests: `uv run --frozen pytest -q`.
