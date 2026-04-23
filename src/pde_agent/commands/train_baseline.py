from pde_agent.core.json_output import success_payload


def train_baseline(run_id: str, run_root: str) -> dict[str, object]:
    return success_payload(
        run_id=run_id,
        command="train-baseline",
        state_hint="RESULTS_COLLECTED",
        artifacts={
            "checkpoint_path": f"{run_root}/checkpoints/best.pt",
            "train_log_path": f"{run_root}/logs/train.log",
        },
        metrics={
            "best_epoch": 0,
            "train_loss_final": 0.0,
            "val_loss_final": 0.0,
        },
    )
