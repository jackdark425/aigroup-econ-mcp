"""Lightweight tool registry for the MCP server.

Every econometrics tool exposed over MCP calls :func:`register` exactly once —
typically at import time in the adapter module that defines it. The MCP server
then iterates :data:`REGISTRY.tools` and hands each entry to FastMCP.

Design choices:

* No inheritance / no per-tool wrapper class. The adapter function itself is
  the handler; we attach metadata externally.
* ``include_traceback`` is controlled by ``AIGROUP_ECON_MCP_DEBUG=1`` so
  production clients get terse error payloads by default.
* Lazy groups are supported via :func:`register_module`: the module is only
  imported the first time any of its tools would be registered on the server.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

from .errors import format_exception

log = logging.getLogger(__name__)

_DEBUG = os.environ.get("AIGROUP_ECON_MCP_DEBUG", "").lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    group: str
    handler: Callable[..., Any]


@dataclass
class Registry:
    tools: dict[str, ToolSpec] = field(default_factory=dict)
    _lazy_modules: list[str] = field(default_factory=list)
    _loaded_modules: set[str] = field(default_factory=set)

    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str,
        *,
        group: str,
    ) -> Callable[..., Any]:
        if name in self.tools:
            raise ValueError(f"duplicate tool name: {name!r}")
        wrapped = _wrap_errors(handler)
        self.tools[name] = ToolSpec(
            name=name, description=description, group=group, handler=wrapped
        )
        return wrapped

    def register_module(self, dotted_path: str) -> None:
        """Declare a module whose import side-effects register tools.

        The module is imported in :meth:`load_all`; until then only its path is
        stored. This keeps cold-start fast — heavy libs like statsmodels or
        xgboost only load when the server is actually started.
        """
        if dotted_path not in self._lazy_modules:
            self._lazy_modules.append(dotted_path)

    def load_all(self) -> None:
        for mod in self._lazy_modules:
            if mod in self._loaded_modules:
                continue
            importlib.import_module(mod)
            self._loaded_modules.add(mod)

    def groups(self) -> dict[str, list[ToolSpec]]:
        out: dict[str, list[ToolSpec]] = {}
        for spec in self.tools.values():
            out.setdefault(spec.group, []).append(spec)
        return out

    def __iter__(self) -> Iterable[ToolSpec]:
        return iter(self.tools.values())

    def __len__(self) -> int:
        return len(self.tools)


REGISTRY = Registry()


def register(
    name: str,
    handler: Callable[..., Any],
    description: str,
    *,
    group: str,
) -> Callable[..., Any]:
    """Module-level helper; forwards to the global :data:`REGISTRY`."""
    return REGISTRY.register(name, handler, description, group=group)


def _wrap_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a handler so that any exception is returned as a JSON payload.

    FastMCP lets handlers return strings, so this keeps the MCP wire format
    stable even when the tool fails — clients see ``{"ok": false, "error": ...}``
    instead of a transport-level error.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — this is the single handler for all tool errors
            log.exception("tool %s failed", getattr(func, "__name__", "?"))
            return format_exception(exc, include_traceback=_DEBUG)

    return wrapper
