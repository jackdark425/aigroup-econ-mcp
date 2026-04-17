"""Error types for MCP tool execution.

Tool handlers return a structured JSON error payload on failure:
    {"ok": false, "error": {"code": "...", "message": "...", "details": {...}}}
This lets MCP clients distinguish data/validation errors from estimation or
internal failures without parsing free-form strings.
"""

from __future__ import annotations

import json
import traceback
from typing import Any


class ToolError(Exception):
    """Base class for errors raised from a registered tool."""

    code: str = "tool_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_payload(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


class ValidationError(ToolError):
    """Input data did not meet the tool's contract (shape, types, length)."""

    code = "validation_error"


class EstimationError(ToolError):
    """A statistical model failed to fit or converged to an invalid solution."""

    code = "estimation_error"


def format_exception(exc: BaseException, *, include_traceback: bool = False) -> str:
    """Render any exception as the JSON error payload used by tool handlers."""
    if isinstance(exc, ToolError):
        payload = exc.to_payload()
    else:
        payload = {
            "ok": False,
            "error": {
                "code": type(exc).__name__,
                "message": str(exc),
                "details": {},
            },
        }
    if include_traceback:
        payload["error"]["traceback"] = traceback.format_exc()
    return json.dumps(payload, indent=2, ensure_ascii=False)
