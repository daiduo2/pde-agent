"""Task 1 FNO training policy metrics functions."""

from __future__ import annotations

from typing import Any
import numpy as np
def _short_window_metrics(
    *, prediction: np.ndarray, target: np.ndarray
) -> dict[str, Any]:
    early_stop = min(47, prediction.shape[1])
    middle_start = min(47, prediction.shape[1])
    middle_stop = min(95, prediction.shape[1])
    late_start = min(95, prediction.shape[1])
    late_stop = min(190, prediction.shape[1])
    early_rel = _rel_mse(prediction[:, :early_stop], target[:, :early_stop])
    metrics = {
        "early": {
            "evaluated": early_stop > 0,
            "future_start": 0,
            "future_stop": int(early_stop),
            "rel_mse": early_rel,
            "score": 100.0 * float(np.exp(-20.0 * early_rel)),
        },
        "middle_guardrail": {"evaluated": middle_stop > middle_start},
        "late_guardrail": {"evaluated": late_stop > late_start},
    }
    if middle_stop > middle_start:
        middle_rel = _rel_mse(
            prediction[:, middle_start:middle_stop],
            target[:, middle_start:middle_stop],
        )
        metrics["middle_guardrail"].update(
            {
                "future_start": int(middle_start),
                "future_stop": int(middle_stop),
                "rel_mse": middle_rel,
                "score": 100.0 * float(np.exp(-10.0 * middle_rel)),
            }
        )
    if late_stop > late_start:
        late_rel = _rel_mse(
            prediction[:, late_start:late_stop],
            target[:, late_start:late_stop],
        )
        metrics["late_guardrail"].update(
            {
                "future_start": int(late_start),
                "future_stop": int(late_stop),
                "rel_mse": late_rel,
                "score": 100.0 / (1.0 + 10.0 * late_rel),
            }
        )
    return metrics

def _rel_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    numerator = np.sum((prediction - target) ** 2, axis=(1, 2))
    denominator = np.sum(target**2, axis=(1, 2)) + 1e-12
    return float(np.mean(np.minimum(numerator / denominator, 5.0)))
