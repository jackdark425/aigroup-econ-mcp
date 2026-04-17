"""Tests for the dynamic `guide://econometrics` MCP resource.

The old ``resources/MCP_MASTER_GUIDE.md`` said "21 tools" and referenced
the deleted component architecture. To prevent that class of bug from
coming back, the guide is now rendered from the live ``REGISTRY`` at
request time. These tests assert the guide content reflects reality.
"""

from __future__ import annotations

from aigroup_econ_mcp import __version__
from aigroup_econ_mcp._registrations import load_all
from aigroup_econ_mcp.registry import REGISTRY
from aigroup_econ_mcp.server import _render_guide


def _load() -> None:
    if len(REGISTRY) == 0:
        load_all()


def test_guide_mentions_current_version() -> None:
    _load()
    guide = _render_guide()
    assert __version__ in guide, "guide must render the package __version__"


def test_guide_mentions_tool_count() -> None:
    _load()
    guide = _render_guide()
    assert f"{len(REGISTRY)} tools" in guide


def test_guide_lists_every_registered_tool() -> None:
    """No tool should silently drop off the advertised catalogue."""
    _load()
    guide = _render_guide()
    missing = [spec.name for spec in REGISTRY if f"`{spec.name}`" not in guide]
    assert not missing, f"tools missing from guide: {missing}"


def test_guide_includes_error_payload_contract() -> None:
    """The uniform error shape is part of the public contract — document it."""
    _load()
    guide = _render_guide()
    assert '"ok"' in guide and '"error"' in guide
    assert "AIGROUP_ECON_MCP_DEBUG" in guide


def test_guide_has_no_stale_tool_count() -> None:
    """Regression guard against the old hard-coded '21 tools'."""
    _load()
    guide = _render_guide()
    # Any four-letter tool-count claim that isn't the current total is suspicious.
    for stale in ("21 tools", "32 tools", "44 tools", "50 tools"):
        assert stale not in guide, f"stale tool count snuck in: {stale!r}"


def test_guide_has_group_headings_for_each_group() -> None:
    _load()
    guide = _render_guide()
    for group in REGISTRY.groups():
        pretty = group.replace("_", " ").title()
        # Markdown H3 heading for each group
        assert f"### {pretty}" in guide, f"missing group heading: {pretty}"
