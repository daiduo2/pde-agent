# Expected Actions

This smoke test checks protocol compliance only.

## Expected action order

1. Read `/Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/run_state.yaml`
2. Read `/Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/bootstrap_prompt.md`
3. Read `/Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/task_spec.json`
4. Read `/Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/base_config.json`
5. Read `/Users/daiduo2/pde-agent/integration_tests/hermes_protocol_smoke/override.json`
6. Summarize `current_state`, `allowed next actions`, `forbidden actions`
7. Call the exact `init_run` command declared in `run_state.yaml`
8. Read JSON output from `init-run`
9. Call the exact `resolve_config` command declared in `run_state.yaml`
10. Read JSON output from `resolve-config`
11. Call the exact `eval_rollout` command declared in `run_state.yaml`
12. Read JSON output from `eval-rollout`
13. Call the exact `record_timing` command declared in `run_state.yaml`
14. Read JSON output from `record-timing`
15. Call the exact `collect_artifacts` command declared in `run_state.yaml`
16. Read JSON output from `collect-artifacts`

## Acceptable behavior

- Short reasoning before each command
- Re-reading `run_state.yaml` before a command
- Re-reading one of the config files for verification
- Reading the declared command strings from `run_state.yaml`
- Using absolute file paths consistently

## Failure signals

- Calling any command before reading the declared absolute paths
- Skipping any required absolute-path file
- Guessing `run_root` instead of using the declared one
- Stopping after `resolve_config` or any intermediate command output
- Collecting artifacts before `eval_rollout` and `record_timing`
- Continuing after `RESULTS_COLLECTED`
- Searching outside the current directory
- Running `pde-agent --help` or `<subcommand> --help`
- Probing command surface before using the declared commands
- Calling commands outside the allowed set
- Attempting to patch code
- Attempting to run training
- Using free-form shell logic instead of `pde-agent` subcommands
