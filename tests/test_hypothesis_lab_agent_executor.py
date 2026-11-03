"""Tests for hypothesis_lab_agent_executor."""

from __future__ import annotations

import numpy as np
import pytest
import sys

from pde_agent.core.hypothesis_lab_agent_executor import (
    _audit_agent_executor_source,
    _module_cache,
    _prediction_from_agent_executor,
    _run_predict_isolated,
    load_agent_executor,
)
from pde_agent.core.hypothesis_lab_utils import HypothesisLabError


VALID_PREDICT_SOURCE = '''
import numpy as np

def predict(context, **kwargs):
    """Return persistence prediction."""
    n, t, x = context.shape
    forecast_steps = 200 - t
    result = np.zeros((n, 200, x), dtype=np.float32)
    result[:, :t, :] = context
    for step in range(forecast_steps):
        result[:, t + step, :] = context[:, -1, :]
    return result
'''

INVALID_SYNTAX_SOURCE = '''
import numpy as np

def predict(context, **kwargs)
    return context
'''

MISSING_PREDICT_SOURCE = '''
import numpy as np

def compute(context, **kwargs):
    return context
'''

DISALLOWED_IMPORT_SOURCE = '''
import os

def predict(context, **kwargs):
    return context
'''

FORBIDDEN_CALL_SOURCE = '''
import numpy as np

def predict(context, **kwargs):
    eval("1 + 1")
    return context
'''

FORBIDDEN_NUMPY_CALL_SOURCE = '''
import numpy as np

def predict(context, **kwargs):
    np.save("/tmp/out.npy", context)
    return context
'''

INVALID_SIGNATURE_SOURCE = '''
import numpy as np

def predict(data, **kwargs):
    return data
'''


class TestAuditAgentExecutorSource:
    def test_valid_source_passes(self):
        issues = _audit_agent_executor_source(VALID_PREDICT_SOURCE)
        assert issues == []

    def test_syntax_error_reported(self):
        issues = _audit_agent_executor_source(INVALID_SYNTAX_SOURCE)
        assert len(issues) == 1
        assert issues[0]["code"] == "syntax_error"

    def test_missing_predict_function(self):
        issues = _audit_agent_executor_source(MISSING_PREDICT_SOURCE)
        codes = {i["code"] for i in issues}
        assert "missing_predict_function" in codes

    def test_disallowed_import_rejected(self):
        issues = _audit_agent_executor_source(DISALLOWED_IMPORT_SOURCE)
        codes = {i["code"] for i in issues}
        assert "disallowed_import" in codes

    def test_forbidden_call_rejected(self):
        issues = _audit_agent_executor_source(FORBIDDEN_CALL_SOURCE)
        codes = {i["code"] for i in issues}
        assert "forbidden_call" in codes

    def test_forbidden_numpy_call_rejected(self):
        issues = _audit_agent_executor_source(FORBIDDEN_NUMPY_CALL_SOURCE)
        codes = {i["code"] for i in issues}
        assert "forbidden_numpy_call" in codes

    def test_invalid_predict_signature(self):
        issues = _audit_agent_executor_source(INVALID_SIGNATURE_SOURCE)
        codes = {i["code"] for i in issues}
        assert "invalid_predict_signature" in codes

    def test_numpy_fft_allowed(self):
        source = '''
import numpy as np

def predict(context, **kwargs):
    return np.fft.fft(context)
'''
        issues = _audit_agent_executor_source(source)
        assert issues == []

    def test_typing_import_allowed(self):
        source = '''
import numpy as np
from typing import Any

def predict(context, **kwargs) -> np.ndarray:
    return context
'''
        issues = _audit_agent_executor_source(source)
        assert issues == []

    def test_math_import_allowed(self):
        source = '''
import numpy as np
import math

def predict(context, **kwargs):
    return context * math.pi
'''
        issues = _audit_agent_executor_source(source)
        assert issues == []


class TestLoadAgentExecutor:
    def test_loads_valid_executor(self):
        predict_fn = load_agent_executor(VALID_PREDICT_SOURCE, audit=True)
        assert callable(predict_fn)
        context = np.ones((2, 10, 64), dtype=np.float32)
        result = predict_fn(context)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 200, 64)

    def test_rejects_invalid_source(self):
        with pytest.raises(HypothesisLabError) as exc_info:
            load_agent_executor(DISALLOWED_IMPORT_SOURCE, audit=True)
        assert "audit failed" in str(exc_info.value)

    def test_module_caching(self):
        # Clear cache for this source
        import hashlib
        source_hash = hashlib.sha256(VALID_PREDICT_SOURCE.encode("utf-8")).hexdigest()
        _module_cache.pop(source_hash, None)

        predict_fn_1 = load_agent_executor(VALID_PREDICT_SOURCE, audit=True)
        predict_fn_2 = load_agent_executor(VALID_PREDICT_SOURCE, audit=True)
        assert predict_fn_1 is predict_fn_2

    def test_audit_disabled_skips_checks(self):
        # Even with audit=False, syntax errors still fail at parse time
        # but disallowed imports would pass if we could exec them.
        # This test verifies audit=False path works for valid code.
        predict_fn = load_agent_executor(VALID_PREDICT_SOURCE, audit=False)
        assert callable(predict_fn)

    def test_missing_predict_raises(self):
        source = '''
import numpy as np

def run(context):
    return context
'''
        with pytest.raises(HypothesisLabError) as exc_info:
            load_agent_executor(source, audit=True)
        assert "missing_predict_function" in str(exc_info.value) or "audit failed" in str(exc_info.value)


class TestPredictionFromAgentExecutor:
    def test_executes_prediction(self):
        context = np.ones((2, 10, 64), dtype=np.float32)
        target = np.ones((2, 200, 64), dtype=np.float32)
        params = {"executor_source": VALID_PREDICT_SOURCE}
        result = _prediction_from_agent_executor(
            context=context,
            target=target,
            reference=None,
            params=params,
            executor_source=VALID_PREDICT_SOURCE,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 200, 64)

    def test_passes_reference(self):
        source = '''
import numpy as np

def predict(context, reference=None, **kwargs):
    if reference is not None:
        return reference
    return context
'''
        context = np.ones((2, 10, 64), dtype=np.float32)
        target = np.ones((2, 200, 64), dtype=np.float32)
        reference = np.zeros((2, 200, 64), dtype=np.float32)
        result = _prediction_from_agent_executor(
            context=context,
            target=target,
            reference=reference,
            params={},
            executor_source=source,
        )
        np.testing.assert_array_equal(result, reference)

    def test_non_ndarray_return_raises(self):
        source = '''
import numpy as np

def predict(context, **kwargs):
    return "not an array"
'''
        context = np.ones((2, 10, 64), dtype=np.float32)
        target = np.ones((2, 200, 64), dtype=np.float32)
        with pytest.raises(HypothesisLabError) as exc_info:
            _prediction_from_agent_executor(
                context=context,
                target=target,
                reference=None,
                params={},
                executor_source=source,
            )
        assert "must return np.ndarray" in str(exc_info.value)

    def test_params_passed_as_kwargs(self):
        source = '''
import numpy as np

def predict(context, gain=1.0, **kwargs):
    return context * float(gain)
'''
        context = np.ones((2, 10, 64), dtype=np.float32)
        target = np.ones((2, 200, 64), dtype=np.float32)
        result = _prediction_from_agent_executor(
            context=context,
            target=target,
            reference=None,
            params={"gain": 2.5},
            executor_source=source,
        )
        np.testing.assert_array_almost_equal(result, context * 2.5)

    def test_isolated_mode_executes_prediction(self):
        context = np.ones((2, 10, 64), dtype=np.float32)
        target = np.ones((2, 200, 64), dtype=np.float32)
        result = _prediction_from_agent_executor(
            context=context,
            target=target,
            reference=None,
            params={},
            executor_source=VALID_PREDICT_SOURCE,
            isolated=True,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 200, 64)

    def test_non_isolated_mode_still_works(self):
        context = np.ones((2, 10, 64), dtype=np.float32)
        target = np.ones((2, 200, 64), dtype=np.float32)
        result = _prediction_from_agent_executor(
            context=context,
            target=target,
            reference=None,
            params={},
            executor_source=VALID_PREDICT_SOURCE,
            isolated=False,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 200, 64)


class TestRunPredictIsolated:
    def test_valid_code_runs_in_isolated_mode(self):
        context = np.ones((2, 10, 64), dtype=np.float32)
        kwargs = {"target": np.ones((2, 200, 64), dtype=np.float32)}
        result = _run_predict_isolated(VALID_PREDICT_SOURCE, context, kwargs)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 200, 64)

    def test_timeout_on_infinite_loop(self):
        source = '''
import numpy as np

def predict(context, **kwargs):
    while True:
        pass
    return context
'''
        context = np.ones((2, 10, 64), dtype=np.float32)
        kwargs = {}
        with pytest.raises(HypothesisLabError) as exc_info:
            _run_predict_isolated(source, context, kwargs, timeout=1.0)
        assert "timed out" in str(exc_info.value)

    @pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="RLIMIT_AS memory limiting only enforced on Linux",
    )
    def test_memory_limit_crashes_huge_allocation(self):
        source = '''
import numpy as np

def predict(context, **kwargs):
    # Try to allocate ~100 MB (well above the 1 MB limit we'll set)
    huge = np.zeros((10000, 10000), dtype=np.float64)
    return context
'''
        context = np.ones((2, 10, 64), dtype=np.float32)
        kwargs = {}
        with pytest.raises(HypothesisLabError) as exc_info:
            _run_predict_isolated(
                source, context, kwargs, max_memory_bytes=1 * 1024 * 1024
            )
        assert "crashed" in str(exc_info.value)
