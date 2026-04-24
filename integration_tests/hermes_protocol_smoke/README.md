# Hermes Protocol Smoke Test

This directory contains the first non-invasive integration test between Hermes and `pde-agent`.

## Goal

Verify that Hermes, using Kimi 2.6, can obey:
- external `run_state.yaml`
- file-based task inputs
- `pde-agent` command boundaries
- absolute-path discipline

This test does not evaluate scientific quality or training quality.

## Test scope

Current scope:
- state = `TASK_PARSED`
- allowed transition = `TASK_PARSED -> RESULTS_COLLECTED`
- allowed command execution:
  - only the exact `init_run` command declared in `run_state.yaml`
  - only the exact `resolve_config` command declared in `run_state.yaml`
  - only the exact `eval_rollout` command declared in `run_state.yaml`
  - only the exact `record_timing` command declared in `run_state.yaml`
  - only the exact `collect_artifacts` command declared in `run_state.yaml`

Forbidden in this round:
- code patching
- training
- rollout evaluation
- spectrum analysis
- physics checks
- submission building
- searching outside the current directory
- command help probing

## Files

- `bootstrap_prompt.md`: first prompt entry for Hermes
- `run_state.yaml`: source of truth for allowed actions
- `task_spec.json`: minimal Burgers task spec
- `base_config.json`: minimal baseline config
- `override.json`: small override to validate merge behavior
- `expected_actions.md`: human-facing expected behavior checklist

## Success criteria

The run is successful if Hermes:
1. reads the required files first
2. uses the declared absolute paths instead of relying on cwd
3. calls only the exact allowed command strings
4. does not inspect command help or probe command surface
5. reaches `RESULTS_COLLECTED` without violating forbidden actions
6. produces `runs/hermes_protocol_smoke_003/artifacts/manifest.json`

## Manual run notes

Start Hermes with the dedicated `hermes-protocol-smoke` profile pointing at `hermes_home/` inside this directory.
Review its behavior through:
- `hermes_home/logs/agent.log`
- `hermes_home/logs/errors.log`
- `hermes_home/sessions/`
