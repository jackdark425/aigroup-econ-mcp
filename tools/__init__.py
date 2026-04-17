"""Adapter layer that bridges ``econometrics/`` algorithms to MCP tools.

Each ``*_adapter.py`` module wraps one domain of algorithms with I/O,
data-loading, and output-formatting. MCP tool registration lives in
:mod:`aigroup_econ_mcp._registrations` — adapter modules themselves have no
dependency on the server or the MCP protocol.
"""

from .data_loader import DataLoader
from .output_formatter import OutputFormatter

__all__ = ["DataLoader", "OutputFormatter"]
