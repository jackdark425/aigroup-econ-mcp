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
        return _load_guide()

    groups = REGISTRY.groups()
    log.info(
        "registered %d tools across %d groups: %s",
        len(REGISTRY),
        len(groups),
        ", ".join(f"{g}={len(t)}" for g, t in sorted(groups.items())),
    )
    return mcp


def _load_guide() -> str:
    guide_path = Path(__file__).resolve().parent.parent / "resources" / "MCP_MASTER_GUIDE.md"
    if guide_path.exists():
        return guide_path.read_text(encoding="utf-8")
    return "Guide not available. See the project README for tool documentation."


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
