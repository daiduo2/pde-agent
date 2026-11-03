"""Tool registry for agent-driven experimentation.

Provides a unified discovery and invocation surface for all capabilities
available to the agent.  Every tool exposes:

* ``name`` — machine-readable identifier
* ``description`` — natural-language summary for LLM consumption
* ``parameters`` — JSON-Schema-like contract
* ``fn`` — the actual callable

Usage::

    from pde_agent.core.tool_registry import registry

    # Register a new tool
    @registry.register(
        name="train_model",
        description="Train a neural-operator model on Burgers data",
        parameters={"model_family": {"type": "str"}, ...},
    )
    def train_model(model_family: str, ...): ...

    # Agent queries available tools
    schemas = registry.get_schemas()

    # Agent invokes a tool by name
    result = registry.call("train_model", {"model_family": "fno_lite", ...})
"""
from __future__ import annotations

import json
import traceback
from typing import Any, Callable, Dict, List


class ToolRegistry:
    """Unified tool discovery and invocation registry."""

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._schemas: List[Dict[str, Any]] = []
        self._descriptions: Dict[str, str] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: register a callable as a named tool.

        Args:
            name: Machine-readable tool identifier (kebab-case recommended).
            description: Natural-language summary for LLM/agent consumption.
            parameters: JSON-Schema-like dict describing accepted arguments.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._tools:
                raise RuntimeError(f"tool '{name}' is already registered")
            self._tools[name] = func
            self._descriptions[name] = description
            self._schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )
            return func

        return decorator

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a registered tool by name with keyword arguments.

        Returns:
            A dict with ``status`` ("ok" or "error"), ``result`` or
            ``error``/``traceback``.
        """
        if name not in self._tools:
            return {
                "status": "error",
                "error": f"tool '{name}' not found. Available: {', '.join(self.list_tools())}",
            }
        try:
            result = self._tools[name](**arguments)
            if not isinstance(result, dict):
                result = {"result": result}
            return {"status": "ok", **result}
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return all tool schemas (name + description + parameters)."""
        return list(self._schemas)

    def list_tools(self) -> List[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def get_description(self, name: str) -> str:
        """Return the description for a registered tool."""
        return self._descriptions.get(name, "")


# Global registry instance — all core tools register here.
registry = ToolRegistry()
