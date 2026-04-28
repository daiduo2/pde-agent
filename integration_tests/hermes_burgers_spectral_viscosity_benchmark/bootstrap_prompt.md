# Bootstrap Prompt

You are executing a constrained Hermes protocol for the Burgers spectral
viscosity benchmark.

Required order:

1. Read `run_state.yaml`.
2. Read `bootstrap_prompt.md`.
3. Read `task_spec.json`.
4. Run exactly the single command listed in `allowed_commands`.
5. Stop after the command returns
   `BURGERS_SPECTRAL_VISCOSITY_BENCHMARK_COMPLETED`.

Do not inspect generated artifacts manually. The audit script reads artifacts
after the run. Do not call command help, patch code, download data, search for
paths, or run lower-level benchmark component commands.
