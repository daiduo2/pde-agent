Call `pde-agent eval-rollout` only after confirming:
- `run_id` exists
- `checkpoint_path` exists
- `resolved_config_path` exists
- read only:
  - status
  - artifacts.rollout_metrics_path
  - metrics.one_step_nrmse
  - metrics.long_horizon_nrmse
