# Task

Run the constrained-free Hermes autonomy ablation in this directory.

Read these files first, in this order:

1. `{run_state_path}`
2. `{bootstrap_prompt_path}`
3. `{task_spec_path}`

Follow those files exactly. Then run the exact `run_fno_iteration` command.
After the terminal JSON result is available, choose exactly one of the two
allowed `record-autonomy-decision` commands.

Do not run unlisted commands.

Stop only after the final command returns `HERMES_AUTONOMY_DECISION_RECORDED`.
