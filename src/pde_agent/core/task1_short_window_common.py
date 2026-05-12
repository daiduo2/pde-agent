from __future__ import annotations


STATE_HINT = "TASK1_SHORT_WINDOW_TRAINING_LOOP_RAN"
FAMILY_SELECTION_STATE_HINT = "TASK1_SHORT_WINDOW_FAMILY_SELECTION_RAN"
SUPPORTED_MODEL_FAMILIES = {"conv1d_residual", "fno_ar1", "fno_lite"}
SUPPORTED_FNO_LOSS_TYPES = {
    "mse",
    "early_weighted_mse",
    "horizon_weighted_mse",
    "spectral_mse",
}
SUPPORTED_NORMALIZATION_TYPES = {"none", "sample_standard"}


class Task1ShortWindowTrainingError(RuntimeError):
    pass
