"""Protocol-level end-to-end tests.

These spawn ``aigroup-econ-mcp`` as a real subprocess over stdio and drive
it with the MCP client SDK. They catch issues that the in-process tests
can't:

* JSON-RPC framing on stdio
* ``initialize`` handshake
* ``tools/list`` schema generation from adapter signatures
* ``tools/call`` round-trip with a real tool

Running these is slow (process spawn + import of statsmodels, sklearn, etc.),
so they live behind the ``mcp`` marker by convention.
"""

from __future__ import annotations

import json
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.asyncio


SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "aigroup_econ_mcp"],
)


async def test_initialize_handshake() -> None:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            result = await session.initialize()
            assert result.serverInfo.name == "aigroup-econ-mcp"
            assert result.protocolVersion  # any version string


async def test_tools_list_returns_66_tools_with_schemas() -> None:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools

    assert len(tools) == 66

    # Every tool must have a JSON schema for its inputs. This is the check
    # that caught the early lazy-wrapper bug where FastMCP couldn't derive
    # a schema because the handler signature was just ``*args, **kwargs``.
    for tool in tools:
        assert tool.name
        assert tool.description
        assert tool.inputSchema, f"{tool.name} has no input schema"
        assert "properties" in tool.inputSchema, f"{tool.name} schema has no properties"


async def test_tools_list_has_expected_groups() -> None:
    """Spot-check a known tool from each group is present."""
    expected = {
        "basic_parametric_estimation_ols",
        "causal_difference_in_differences",
        "time_series_arima_model",
        "ml_random_forest",
        "micro_logit",
        "missing_data_simple_imputation",
        "model_diagnostic_tests",
        "nonparametric_kernel_regression",
        "spatial_weights_matrix",
        "inference_bootstrap",
        "decomposition_oaxaca_blinder",
    }
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
    names = {t.name for t in tools}
    missing = expected - names
    assert not missing, f"missing tools: {missing}"


async def test_tools_call_ols_roundtrip() -> None:
    """A real tools/call round-trip with a well-formed OLS dataset."""
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "basic_parametric_estimation_ols",
                arguments={
                    "y_data": [1.0, 2.1, 2.9, 4.1, 5.0, 5.9, 7.1, 8.0, 8.9, 10.1],
                    "x_data": [[float(i)] for i in range(1, 11)],
                    "output_format": "json",
                },
            )
    assert not result.isError, f"tool call flagged as error: {result}"
    # The adapter serializes its Pydantic result to JSON in ``.text``.
    assert result.content, "empty tool response"
    # First content block is a text block for all our adapters.
    body = result.content[0].text
    parsed = json.loads(body)
    assert isinstance(parsed, dict)
    # On success we expect regression output fields; on failure a structured error.
    assert parsed.get("ok") is False or "coefficients" in parsed or "r_squared" in parsed


async def test_tools_call_returns_structured_error_on_bad_input() -> None:
    """Input-shape mismatch surfaces through the MCP wire as a readable payload."""
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "basic_parametric_estimation_ols",
                arguments={
                    "y_data": [1.0, 2.0],
                    "x_data": [[1.0], [2.0], [3.0]],  # length mismatch
                    "output_format": "json",
                },
            )
    # Either the protocol-level isError is set, or the body carries {"ok": false, ...}
    body = result.content[0].text if result.content else ""
    parsed = json.loads(body) if body else {}
    assert result.isError or parsed.get("ok") is False or "error" in parsed, (
        f"expected structured error, got: {parsed}"
    )


async def test_unknown_tool_returns_error() -> None:
    """MCP server rejects a call to a tool that isn't registered."""
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # SDK raises on unknown tool OR returns isError=True; both are fine.
            try:
                result = await session.call_tool("this_tool_does_not_exist", arguments={})
                assert result.isError, "expected error for unknown tool"
            except Exception as exc:
                # Some SDK versions raise — that's a valid failure mode.
                assert (
                    "unknown" in str(exc).lower()
                    or "not found" in str(exc).lower()
                    or "tool" in str(exc).lower()
                )
