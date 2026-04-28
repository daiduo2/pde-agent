# Task

Run the Hermes real-HDF5 DeepONet strategy sweep protocol.

Read these files first, in this order:

1. `{run_state_path}`
2. `{bootstrap_prompt_path}`
3. `{task_spec_path}`

Then execute the exact commands in `allowed_commands`, in order. Do not run
unlisted commands. Stop only after the final command returns
`STRATEGY_PROPOSAL_RECORDED`.

At the end, briefly summarize:
- baseline score
- DeepONet winner strategy label
- evidence decision
- retrieved strategy id
- why no forbidden action was needed
