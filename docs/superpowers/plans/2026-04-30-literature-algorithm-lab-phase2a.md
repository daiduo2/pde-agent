# Literature Algorithm Lab Phase 2a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Connect the offline literature algorithm backbone to source discovery, document download, and MinerU parse task artifacts without requiring live network calls in unit tests.

**Architecture:** Extend `literature_algorithm_lab` with fixture-testable normalizers for OpenAlex/arXiv/GitHub, a cache-aware document downloader, and MinerU submit/poll/download task contracts. Keep API keys out of code and artifacts; live MinerU uses `MINERU_API_TOKEN` only when a command is run explicitly.

**Tech Stack:** Python standard library (`json`, `xml.etree.ElementTree`, `urllib.request`, `urllib.parse`, `zipfile`, `hashlib`), Typer CLI, existing JSON `success_payload`, pytest fixture tests.

---

## Task 1: Normalize OpenAlex, arXiv, and GitHub Search Results

**Files:**
- Modify: `src/pde_agent/core/literature_algorithm_lab.py`
- Modify: `tests/test_literature_algorithm_lab.py`

Steps:

1. Add tests for:
   - OpenAlex work JSON fixture -> `EXTERNAL_SOURCE_SEARCH_NORMALIZED`
   - arXiv Atom fixture -> source candidates with PDF URL
   - GitHub repository search fixture -> source candidates
2. Implement:
   - `normalize_openalex_search_results`
   - `normalize_arxiv_search_results`
   - `normalize_github_search_results`
3. Each source candidate must include:
   - `source_id`
   - `source_type`
   - `title`
   - `url`
   - `document_url` when available
   - `retrieval_query`
   - `external_ids`
   - `license_note`
   - `discovery_provider`
4. Run:
   - `uv run --frozen pytest -q tests/test_literature_algorithm_lab.py`
5. Commit:
   - `git commit -m "feat: normalize external literature sources"`

## Task 2: Download Source Documents Into Local Cache

**Files:**
- Modify: `src/pde_agent/core/literature_algorithm_lab.py`
- Modify: `tests/test_literature_algorithm_lab.py`

Steps:

1. Add a test that uses an injected fake downloader to fetch bytes from a source candidate into `data/literature_lab`-style cache.
2. Implement `download_source_document`.
3. Output state hint: `SOURCE_DOCUMENT_DOWNLOADED`.
4. The output record must include:
   - source metadata,
   - `local_raw_path`,
   - `content_hash`,
   - `download_url`,
   - `next_command: submit-mineru-parse`
5. Run:
   - `uv run --frozen pytest -q tests/test_literature_algorithm_lab.py`
6. Commit:
   - `git commit -m "feat: download literature source documents"`

## Task 3: MinerU Submit/Poll/Download Contracts

**Files:**
- Modify: `src/pde_agent/core/literature_algorithm_lab.py`
- Modify: `tests/test_literature_algorithm_lab.py`

Steps:

1. Add tests with a fake MinerU client for:
   - precise URL submit with token redacted from artifacts,
   - agent URL submit without token,
   - poll task result for `done`,
   - download markdown result to local path.
2. Implement:
   - `submit_mineru_parse_task`
   - `poll_mineru_parse_task`
   - `download_mineru_parse_result`
   - a small `MinerUHttpClient` using `urllib.request`
3. Output state hints:
   - `MINERU_PARSE_TASK_SUBMITTED`
   - `MINERU_PARSE_TASK_POLLED`
   - `MINERU_PARSE_RESULT_DOWNLOADED`
4. Do not write the API token to reports, logs, or tests.
5. Run:
   - `uv run --frozen pytest -q tests/test_literature_algorithm_lab.py`
6. Commit:
   - `git commit -m "feat: add mineru parse task contracts"`

## Task 4: CLI Commands

**Files:**
- Modify: `src/pde_agent/commands/literature_algorithm_lab.py`
- Modify: `src/pde_agent/cli.py`
- Modify: `tests/test_literature_algorithm_lab.py`

Steps:

1. Add CLI tests for one source normalizer and one MinerU task command.
2. Register commands:
   - `normalize-openalex-sources`
   - `normalize-arxiv-sources`
   - `normalize-github-sources`
   - `download-source-document`
   - `submit-mineru-parse`
   - `poll-mineru-parse`
   - `download-mineru-result`
3. `submit-mineru-parse` reads token from `MINERU_API_TOKEN` unless `--mode agent_url`.
4. Run:
   - `uv run --frozen pytest -q tests/test_literature_algorithm_lab.py`
5. Commit:
   - `git commit -m "feat: expose literature retrieval cli"`

## Task 5: Verification and Live Smoke Notes

**Files:**
- Modify: `docs/superpowers/specs/2026-04-30-literature-to-algorithm-lab-design.md`
- Modify: `docs/superpowers/plans/2026-04-30-literature-algorithm-lab-phase2a.md`

Steps:

1. Run full tests:
   - `uv run --frozen pytest -q`
2. Add a short note to the spec:
   - Phase 2a adds network-capable commands, but ordinary tests remain fixture-only.
   - MinerU tokens must be supplied through `MINERU_API_TOKEN`.
3. Mark this plan complete.
4. Commit:
   - `git commit -m "docs: record literature retrieval phase2a verification"`

## Live Smoke Policy

After implementation, run live smoke only if explicitly requested or if network escalation is approved:

1. Use OpenAlex/arXiv search for a small neural-operator query.
2. Download one arXiv PDF into `data/literature_lab`.
3. Submit MinerU precise parse with `MINERU_API_TOKEN`.
4. Poll until `done`.
5. Download markdown/zip.
6. Feed markdown into existing `ingest-mineru-parse`.

The token must never be committed, printed, or embedded in command history beyond environment variable usage.
