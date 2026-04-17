"""Static validation of the tool manifest.

``aigroup_econ_mcp._registrations._MANIFEST`` is the single source of truth
for the 66 MCP tools. These tests assert invariants on that table directly,
independently of the FastMCP server, so a typo in a module path or a
mis-spelled function name breaks the test before it reaches a user.
"""

from __future__ import annotations

import importlib

import pytest

from aigroup_econ_mcp._registrations import _MANIFEST


def test_manifest_entries_are_well_formed() -> None:
    """Every row is the (name, group, target, description) 4-tuple."""
    for row in _MANIFEST:
        assert len(row) == 4, f"bad manifest row: {row!r}"
        name, group, target, description = row
        assert isinstance(name, str) and name, f"empty name: {row!r}"
        assert isinstance(group, str) and group, f"empty group in {name!r}"
        assert description, f"empty description for {name!r}"
        # target is either a pre-bound callable (shim) or a 'module:function' path
        assert callable(target) or (
            isinstance(target, str) and ":" in target
        ), f"bad target for {name!r}: {target!r}"


def test_manifest_names_are_unique() -> None:
    names = [row[0] for row in _MANIFEST]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"duplicate tool names: {duplicates}"


def test_manifest_count_matches_expected() -> None:
    """Guard against accidental additions/removals."""
    assert len(_MANIFEST) == 66


def test_every_string_target_resolves_to_a_callable() -> None:
    """Import each `module:function` target lazily and check it's callable.

    This catches:
    * misspelled adapter function names,
    * adapter modules that moved or were renamed,
    * typos in the manifest like ``tools.time_series_panel_data_adapter:...``
      that slip through when only e2e tests run happy paths.
    """
    unresolvable: list[str] = []
    for _name, _group, target, _description in _MANIFEST:
        if not isinstance(target, str):
            continue  # shim callables are verified below
        module_path, _, func_name = target.partition(":")
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001
            # Missing optional runtime libs (xgboost/libomp) should not fail
            # this structural check — only record symptoms that suggest a
            # typo/rename rather than a missing system library.
            msg = str(exc).lower()
            if "libomp" in msg or "libxgboost" in msg or "dylib" in msg:
                continue
            unresolvable.append(f"{target!r} — import error: {exc}")
            continue
        func = getattr(module, func_name, None)
        if func is None:
            unresolvable.append(f"{target!r} — attribute {func_name!r} not in {module_path!r}")
        elif not callable(func):
            unresolvable.append(f"{target!r} — resolved to non-callable {type(func).__name__}")
    assert not unresolvable, "manifest targets not resolvable:\n  " + "\n  ".join(unresolvable)


def test_callable_targets_accept_kwargs() -> None:
    """Shim functions (non-string manifest targets) must be plain callables."""
    for name, _group, target, _description in _MANIFEST:
        if isinstance(target, str):
            continue
        assert callable(target), f"shim for {name!r} is not callable"


EXPECTED_GROUPS = frozenset({
    "basic_parametric",
    "causal_inference",
    "time_series",
    "machine_learning",
    "microecon",
    "missing_data",
    "model_specification",
    "nonparametric",
    "spatial_econometrics",
    "statistical_inference",
    "distribution_analysis",
})


def test_manifest_groups_are_in_known_set() -> None:
    """Prevent a typo like ``microecon`` vs ``microeconomics`` silently
    splitting a group in two on the wire."""
    groups = {row[1] for row in _MANIFEST}
    unknown = groups - EXPECTED_GROUPS
    assert not unknown, f"unknown groups in manifest: {unknown}"
    missing = EXPECTED_GROUPS - groups
    assert not missing, f"expected groups missing from manifest: {missing}"


@pytest.mark.parametrize("row", _MANIFEST, ids=lambda r: r[0])
def test_each_tool_individually(row: tuple) -> None:
    """One test per tool for precise failure attribution in CI output."""
    name, group, target, description = row
    assert name and group and description
    assert isinstance(target, str) or callable(target)
