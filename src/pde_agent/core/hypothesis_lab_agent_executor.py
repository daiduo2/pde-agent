"""Agent-generated executor loader with AST security audit for Hypothesis Lab."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import multiprocessing
import os
import sys
import types
from typing import Any

import numpy as np

from pde_agent.core.hypothesis_lab_utils import HypothesisLabError

_AGENT_EXECUTOR_ALLOWED_IMPORTS = {"numpy", "math", "typing"}
_AGENT_EXECUTOR_FORBIDDEN_CALLS = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "getattr",
}
_AGENT_EXECUTOR_FORBIDDEN_NUMPY_CALLS = {
    "load",
    "save",
    "savez",
    "savez_compressed",
    "fromfile",
    "memmap",
}

# Module cache keyed by SHA256 of source code
_module_cache: dict[str, types.ModuleType] = {}


def load_agent_executor(source_code: str, *, audit: bool = True) -> Any:
    """Load a predict() function from agent-generated source code.

    Args:
        source_code: Python source code defining predict(context, **kwargs).
        audit: Whether to run AST security audit before loading.

    Returns:
        The predict function extracted from the loaded module.

    Raises:
        HypothesisLabError: If audit fails or module loading fails.
    """
    if audit:
        issues = _audit_agent_executor_source(source_code)
        if issues:
            messages = "; ".join(
                f"{i['code']}({i.get('line', '?')}): {i.get('message', i.get('call', i.get('module', '')))}"
                for i in issues
            )
            raise HypothesisLabError(f"agent executor audit failed: {messages}")

    source_hash = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
    if source_hash in _module_cache:
        module = _module_cache[source_hash]
        return getattr(module, "predict")

    module = _load_agent_executor_module(source_code, source_hash)
    _module_cache[source_hash] = module
    return getattr(module, "predict")


def _audit_agent_executor_source(source_code: str) -> list[dict[str, Any]]:
    """AST audit for security. Reuses patterns from literature_bounded_executor."""
    issues: list[dict[str, Any]] = []
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return [{"code": "syntax_error", "message": str(exc), "line": exc.lineno}]

    found_function = False
    import_aliases: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_name = alias.name.split(".", 1)[0]
                if root_name not in _AGENT_EXECUTOR_ALLOWED_IMPORTS:
                    issues.append(
                        {
                            "code": "disallowed_import",
                            "module": alias.name,
                            "line": node.lineno,
                        }
                    )
                import_aliases[alias.asname or root_name] = root_name
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if module == "__future__":
                continue
            if module not in _AGENT_EXECUTOR_ALLOWED_IMPORTS:
                issues.append(
                    {
                        "code": "disallowed_import",
                        "module": node.module,
                        "line": node.lineno,
                    }
                )
            for alias in node.names:
                if module == "numpy" and alias.name in _AGENT_EXECUTOR_FORBIDDEN_NUMPY_CALLS:
                    issues.append(
                        {
                            "code": "forbidden_numpy_import",
                            "module": node.module,
                            "name": alias.name,
                            "line": node.lineno,
                        }
                    )
                import_aliases[alias.asname or alias.name] = module
        elif isinstance(node, ast.FunctionDef) and node.name == "predict":
            found_function = True
            args = [arg.arg for arg in node.args.args]
            if not args or args[0] != "context":
                issues.append(
                    {
                        "code": "invalid_predict_signature",
                        "expected_args": ["context", "..."],
                        "actual_args": args,
                        "line": node.lineno,
                    }
                )
        elif isinstance(node, ast.Call):
            call_name = _ast_call_name(node.func)
            if call_name in _AGENT_EXECUTOR_FORBIDDEN_CALLS:
                issues.append(
                    {
                        "code": "forbidden_call",
                        "call": call_name,
                        "line": node.lineno,
                    }
                )
            if "." in call_name:
                receiver, attr = call_name.split(".", 1)
                root_module = import_aliases.get(receiver, receiver)
                if root_module == "numpy" and attr in _AGENT_EXECUTOR_FORBIDDEN_NUMPY_CALLS:
                    issues.append(
                        {
                            "code": "forbidden_numpy_call",
                            "call": call_name,
                            "line": node.lineno,
                        }
                    )
            else:
                root_module = import_aliases.get(call_name)
                if root_module == "numpy" and call_name in _AGENT_EXECUTOR_FORBIDDEN_NUMPY_CALLS:
                    issues.append(
                        {
                            "code": "forbidden_numpy_call",
                            "call": call_name,
                            "line": node.lineno,
                        }
                    )

    if not found_function:
        issues.append({"code": "missing_predict_function", "function_name": "predict"})

    return issues


def _ast_call_name(node: ast.AST) -> str:
    """Extract the dotted name of a call expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _ast_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _load_agent_executor_module(source_code: str, source_hash: str) -> types.ModuleType:
    """Dynamically load a module from source code string."""
    module_name = f"_pde_agent_agent_executor_{source_hash[:16]}"
    # Use a temporary file for importlib compatibility
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    if spec is None:
        raise HypothesisLabError("cannot create module spec for agent executor")
    module = importlib.util.module_from_spec(spec)
    exec(source_code, module.__dict__)  # noqa: S102
    if not hasattr(module, "predict"):
        raise HypothesisLabError("agent executor module missing predict function")
    sys.modules[module_name] = module
    return module


def _run_predict_isolated(
    executor_source: str,
    context: np.ndarray,
    kwargs: dict[str, Any],
    *,
    timeout: float = 60.0,
    max_memory_bytes: int = 512 * 1024 * 1024,
) -> np.ndarray:
    """Run predict() in a child process with resource limits.

    Args:
        executor_source: Python source code string defining predict().
        context: Input context array.
        kwargs: Keyword arguments to pass to predict().
        timeout: Maximum seconds to wait for the child process.
        max_memory_bytes: Maximum memory (RLIMIT_AS) for the child process.

    Returns:
        Prediction array from the child process.

    Raises:
        HypothesisLabError: On timeout, crash, or invalid return.
    """
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)

    p = multiprocessing.Process(
        target=_isolated_predict_worker,
        args=(executor_source, context, kwargs, child_conn, max_memory_bytes),
    )
    p.start()
    child_conn.close()

    try:
        if not parent_conn.poll(timeout):
            p.terminate()
            p.join(5)
            if p.is_alive():
                p.kill()
                p.join(5)
            raise HypothesisLabError("agent executor timed out")

        response = parent_conn.recv()
    finally:
        parent_conn.close()
        # Give child a moment to exit naturally after sending result
        p.join(0.5)
        if p.is_alive():
            p.terminate()
            p.join(2)
            if p.is_alive():
                p.kill()
                p.join(2)

    if p.exitcode != 0:
        raise HypothesisLabError(
            f"agent executor crashed: exitcode {p.exitcode}"
        )

    if isinstance(response, Exception):
        raise HypothesisLabError(f"agent executor error: {response}")
    if not isinstance(response, np.ndarray):
        raise HypothesisLabError("agent executor predict() must return np.ndarray")
    return response


def _isolated_predict_worker(
    executor_source: str,
    context: np.ndarray,
    kwargs: dict[str, Any],
    conn: multiprocessing.connection.Connection,
    max_memory_bytes: int,
) -> None:
    """Worker function executed in child process."""
    try:
        # Set memory limit on Unix (Linux only; macOS RLIMIT_AS is not enforced)
        if sys.platform.startswith("linux"):
            import resource

            try:
                resource.setrlimit(
                    resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes)
                )
            except (OSError, ValueError):
                pass  # Skip gracefully if rlimit fails

        # Minimal module loading without importing full pde_agent tree
        module_name = "_isolated_agent_executor"
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        if spec is None:
            conn.send(
                HypothesisLabError("cannot create module spec for agent executor")
            )
            return
        module = importlib.util.module_from_spec(spec)
        exec(executor_source, module.__dict__)  # noqa: S102
        predict_fn = getattr(module, "predict")
        result = predict_fn(context, **kwargs)
        conn.send(result)
    except Exception as exc:
        conn.send(exc)


def _prediction_from_agent_executor(
    *,
    context: np.ndarray,
    target: np.ndarray,
    reference: np.ndarray | None,
    params: dict[str, Any],
    executor_source: str,
    isolated: bool = True,
) -> np.ndarray:
    """Load executor and call predict().

    Args:
        context: Input context array [N, T_context, X].
        target: Target array (passed through to predict as keyword).
        reference: Optional reference array (passed through to predict as keyword).
        params: Blueprint parameters.
        executor_source: Python source code string defining predict().
        isolated: If True, run in a separate process with resource limits.

    Returns:
        Prediction array [N, T, X].
    """
    kwargs = dict(params)
    kwargs["target"] = target
    if reference is not None:
        kwargs["reference"] = reference

    if isolated:
        return _run_predict_isolated(executor_source, context, kwargs)

    predict_fn = load_agent_executor(executor_source, audit=True)
    result = predict_fn(context, **kwargs)
    if not isinstance(result, np.ndarray):
        raise HypothesisLabError("agent executor predict() must return np.ndarray")
    return result
