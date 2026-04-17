"""Smoke tests for the MCP tool registry.

These tests guard the wire between ``_registrations`` and the adapter layer.
They don't exercise any statistical algorithm — those belong under
``econometrics/tests`` — but they do catch:

* a renamed adapter function (``ImportError`` at registration time),
* a duplicate MCP tool name,
* a registration dropped when editing ``_registrations.py``,
* a handler that doesn't pass exceptions through the error wrapper.
"""

from __future__ import annotations

import json

import pytest

from aigroup_econ_mcp._registrations import load_all
from aigroup_econ_mcp.registry import REGISTRY

EXPECTED_TOOL_COUNT = 66

EXPECTED_GROUP_COUNTS = {
    "basic_parametric": 3,
    "causal_inference": 13,
    "time_series": 11,
    "machine_learning": 8,
    "microecon": 7,
    "missing_data": 2,
    "model_specification": 7,
    "nonparametric": 4,
    "spatial_econometrics": 6,
    "statistical_inference": 2,
    "distribution_analysis": 3,
}


@pytest.fixture(scope="module", autouse=True)
def _load_registry() -> None:
    if len(REGISTRY) == 0:
        load_all()


def test_total_tool_count() -> None:
    assert len(REGISTRY) == EXPECTED_TOOL_COUNT


def test_group_counts() -> None:
    actual = {group: len(specs) for group, specs in REGISTRY.groups().items()}
    assert actual == EXPECTED_GROUP_COUNTS


def test_tool_names_unique_and_snake_case() -> None:
    names = [spec.name for spec in REGISTRY]
    assert len(set(names)) == len(names), "duplicate tool names"
    for name in names:
        assert name.replace("_", "").isalnum(), f"non-snake_case tool name: {name}"
        assert name == name.lower(), f"tool name must be lowercase: {name}"


def test_every_tool_has_description() -> None:
    for spec in REGISTRY:
        assert spec.description.strip(), f"{spec.name} has empty description"


def test_handlers_are_callable() -> None:
    for spec in REGISTRY:
        assert callable(spec.handler), f"{spec.name} handler is not callable"


def test_error_wrapper_returns_json_payload() -> None:
    """Any exception inside a handler should come back as a JSON error payload.

    Uses an obviously invalid argument so adapters fail fast without running
    any statistical code.
    """
    spec = REGISTRY.tools["basic_parametric_estimation_ols"]
    # Pass mismatched y/x lengths — this will raise inside the adapter.
    out = spec.handler(y_data=[1.0, 2.0], x_data=[[1.0]], output_format="json")
    assert isinstance(out, str)
    # The adapter's own code returns a JSON blob on failure too, so we accept
    # either a top-level "error" key or our structured {"ok": false, ...}.
    parsed = json.loads(out)
    assert "error" in parsed or parsed.get("ok") is False
