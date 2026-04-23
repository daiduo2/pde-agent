# Expected Actions

This smoke test checks protocol compliance only.

## Expected action order

1. Read `run_state.yaml`
2. Read `bootstrap_prompt.md`
3. Read `task_spec.json`
4. Read `base_config.json`
5. Read `override.json`
6. Summarize `current_state`, `allowed next actions`, `forbidden actions`
7. Call the exact `init_run` command declared in `run_state.yaml`
8. Read JSON output from `init-run`
9. Call the exact `resolve_config` command declared in `run_state.yaml`
10. Read JSON output from `resolve-config`

## Acceptable behavior

- Short reasoning before each command
- Re-reading `run_state.yaml` before a command
- Re-reading one of the config files for verification
- Reading the declared command strings from `run_state.yaml`

## Failure signals

- Calling any command before reading `run_state.yaml`
- Skipping `bootstrap_prompt.md`, `task_spec.json`, `base_config.json`, or `override.json`
- Guessing `run_root` instead of using the declared one
- Searching outside the current directory
- Running `pde-agent --help` or `<subcommand> --help`
- Probing command surface before using the declared commands
- Calling commands outside the allowed set
- Attempting to patch code
- Attempting to run training
- Using free-form shell logic instead of `pde-agent` subcommands
