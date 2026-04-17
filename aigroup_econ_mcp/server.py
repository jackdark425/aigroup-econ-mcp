"""FastMCP server for aigroup-econ-mcp.

All 66 tools are wired up in :mod:`aigroup_econ_mcp._registrations`. Here we
just trigger that module and hand every registered spec to FastMCP.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import __version__
from ._registrations import load_all
from .registry import REGISTRY

log = logging.getLogger(__name__)


def build_server() -> FastMCP:
    _configure_stdio_utf8()
    load_all()

    mcp = FastMCP("aigroup-econ-mcp")
    for spec in REGISTRY:
        mcp.tool(name=spec.name, description=spec.description)(spec.handler)

    @mcp.resource("guide://econometrics")
    def econometrics_guide() -> str:
        return _render_guide()

    @mcp.resource("guide://data-formats")
    def data_formats_guide() -> str:
        return _load_static_guide("MCP_TOOLS_DATA_FORMAT_GUIDE.md")

    groups = REGISTRY.groups()
    log.info(
        "registered %d tools across %d groups: %s",
        len(REGISTRY),
        len(groups),
        ", ".join(f"{g}={len(t)}" for g, t in sorted(groups.items())),
    )
    return mcp


def _render_guide() -> str:
    """Generate the tool catalogue from the live REGISTRY.

    Avoids shipping a stale static file — every rebuild reflects the current
    set of registered tools. Input/output format notes are static because
    they don't depend on tool membership.
    """
    lines = [
        "# aigroup-econ-mcp — Tool Catalogue",
        "",
        f"Version {__version__} — {len(REGISTRY)} tools across {len(REGISTRY.groups())} groups.",
        "",
        "## Input formats",
        "- Direct: pass data fields (`y_data`, `x_data`, etc.) in the tool call.",
        "- File: set `file_path` to a `.csv` / `.json` / `.xlsx` / `.xls` / `.txt` file.",
        "",
        "## Output formats",
        "- `json` (default, structured result model)",
        "- `markdown` (human-readable tables and coefficient stars)",
        "- `text` (compact dictionary dump — Pydantic model fallback)",
        "",
        "## Error payload",
        "All tool failures return a uniform shape:",
        "```json",
        '{"ok": false, "error": {"code": "...", "message": "...", "details": {...}}}',
        "```",
        "Set the environment variable `AIGROUP_ECON_MCP_DEBUG=1` to include a",
        "traceback in the `error` object.",
        "",
        "## Tools by group",
        "",
    ]
    for group, specs in sorted(REGISTRY.groups().items()):
        pretty = group.replace("_", " ").title()
        lines.append(f"### {pretty} ({len(specs)})")
        lines.append("")
        for spec in sorted(specs, key=lambda s: s.name):
            lines.append(f"- **`{spec.name}`** — {spec.description}")
        lines.append("")
    lines.append(
        "Detailed per-tool parameter shapes are available via "
        "the `guide://data-formats` MCP resource."
    )
    return "\n".join(lines)


def _load_static_guide(name: str) -> str:
    path = Path(__file__).resolve().parent.parent / "resources" / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return f"Guide {name!r} not available. See the project README."


def _configure_stdio_utf8() -> None:
    """Best-effort UTF-8 stdio on Windows consoles."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass


def main() -> None:
    """Entry point used by ``aigroup-econ-mcp`` and ``python -m aigroup_econ_mcp``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log.info("aigroup-econ-mcp %s starting", __version__)
    mcp = build_server()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
