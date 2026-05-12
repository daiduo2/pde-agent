"""Competition timing normalization utilities."""


from typing import Any


GPU_TIME_FACTORS: dict[str, dict[str, Any]] = {
    "h100_80gb": {"factor": 2.0, "official_listed": True},
    "h800_80gb": {"factor": 1.8, "official_listed": True},
    "a100_80gb": {"factor": 1.0, "official_listed": True},
    "a100_40gb": {"factor": 1.0, "official_listed": True},
    "a800_80gb": {"factor": 0.9, "official_listed": True},
    "rtx_pro_6000": {"factor": 0.85, "official_listed": True},
    "v100_32gb": {"factor": 0.45, "official_listed": True},
    "rtx_5090": {"factor": 0.75, "official_listed": True},
    "rtx_4090": {"factor": 0.7, "official_listed": True},
    "rtx_5080": {"factor": 0.7, "official_listed": True},
    "rtx_4080": {"factor": 0.5, "official_listed": True},
    "a40": {"factor": 0.45, "official_listed": True},
    "rtx_3090": {"factor": 0.45, "official_listed": True},
    "rtx_3080": {"factor": 0.3, "official_listed": True},
    "a10": {"factor": 0.3, "official_listed": True},
    "t4": {"factor": 0.15, "official_listed": True},
    "rtx_5060ti": {"factor": 0.1, "official_listed": True},
    "mac_mini_m4_local_dev": {
        "factor": 0.03,
        "official_listed": False,
        "policy": "local_mac_mini_m4_estimated_lower_bound_0.03_0.08",
    },
}
DEFAULT_TIMING_PROFILE = "a100_80gb"


class CompetitionTimingError(RuntimeError):
    pass


def normalize_timing(
    *,
    raw_train_time: float,
    raw_inference_time: float | None,
    timing_profile: str | None = None,
    timing_factor: float | None = None,
) -> dict[str, Any]:
    profile = resolve_timing_profile(
        timing_profile=timing_profile,
        timing_factor=timing_factor,
    )
    factor = float(profile["factor"])
    inference_time = (
        None if raw_inference_time is None else _normalize_seconds(raw_inference_time, factor)
    )
    return {
        "timing_profile": profile["timing_profile"],
        "timing_factor": factor,
        "timing_profile_official_listed": bool(profile["official_listed"]),
        "timing_policy": profile.get("policy"),
        "timing_valid_for_submission": bool(profile["official_listed"]),
        "raw_train_time": _non_negative(raw_train_time, "raw_train_time"),
        "raw_inference_time": None
        if raw_inference_time is None
        else _non_negative(raw_inference_time, "raw_inference_time"),
        "train_time": _normalize_seconds(raw_train_time, factor),
        "inference_time": inference_time,
    }


def resolve_timing_profile(
    *,
    timing_profile: str | None = None,
    timing_factor: float | None = None,
) -> dict[str, Any]:
    profile_name = (timing_profile or DEFAULT_TIMING_PROFILE).strip().lower()
    if profile_name not in GPU_TIME_FACTORS:
        raise CompetitionTimingError(f"unknown timing profile: {timing_profile}")
    profile = {
        "timing_profile": profile_name,
        **GPU_TIME_FACTORS[profile_name],
    }
    if timing_factor is not None:
        if isinstance(timing_factor, bool) or timing_factor <= 0.0:
            raise CompetitionTimingError("timing_factor must be positive")
        profile["factor"] = float(timing_factor)
        profile["official_listed"] = False
        profile["policy"] = "explicit_override"
    return profile


def _normalize_seconds(value: float, factor: float) -> float:
    return float(_non_negative(value, "seconds") * factor)


def _non_negative(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CompetitionTimingError(f"{label} must be numeric")
    seconds = float(value)
    if seconds < 0.0:
        raise CompetitionTimingError(f"{label} must be non-negative")
    return seconds
