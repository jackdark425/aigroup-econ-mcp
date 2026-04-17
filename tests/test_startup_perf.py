"""Startup-time regression guard.

Server startup dominates perceived latency for MCP clients because every
``initialize`` handshake pays the cost once. Two numbers matter:

1. Python import of ``aigroup_econ_mcp.server`` + ``build_server()`` — this is
   what an in-process embedder sees.
2. Subprocess spawn + stdio handshake — what a real MCP client sees.

We assert only that these stay under *generous* ceilings so a stray
"import statsmodels at module scope" regression gets caught, but we don't
over-specify exact numbers (they vary by machine). On the developer laptop
where this was written, cold build_server() is ~1.5s and subprocess
handshake is ~3-4s.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import time

import pytest

BUILD_SERVER_CEILING_SECONDS = 5.0  # cold Python import + registry wiring
SUBPROCESS_HANDSHAKE_CEILING_SECONDS = 15.0  # process spawn + stdio + init


def test_build_server_under_ceiling() -> None:
    """``build_server()`` must stay fast enough for in-process embedders.

    If statsmodels / sklearn / xgboost ever start getting imported eagerly
    (instead of lazily per adapter call), this test will regress before it
    reaches users.
    """
    # Fresh import to avoid warm-cache lying. This clears only the package
    # surface; transitive deps already loaded by pytest stay cached, so the
    # measurement undersells real cold starts — that's fine, we're asserting
    # an upper bound.
    for mod in list(sys.modules):
        if mod.startswith(("aigroup_econ_mcp", "tools.")):
            del sys.modules[mod]

    start = time.perf_counter()
    server_mod = importlib.import_module("aigroup_econ_mcp.server")
    server_mod.build_server()
    elapsed = time.perf_counter() - start

    assert elapsed < BUILD_SERVER_CEILING_SECONDS, (
        f"build_server() took {elapsed:.2f}s (ceiling {BUILD_SERVER_CEILING_SECONDS}s). "
        f"Something likely got imported eagerly at module load — check new imports in "
        f"tools/*_adapter.py and aigroup_econ_mcp/_registrations.py."
    )


@pytest.mark.slow
def test_subprocess_startup_to_initialize_under_ceiling() -> None:
    """Full subprocess spawn + MCP initialize handshake latency."""
    # Spawn the server exactly as an MCP client would.
    start = time.perf_counter()

    proc = subprocess.Popen(
        [sys.executable, "-m", "aigroup_econ_mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Minimal MCP handshake: send initialize, wait for response.
        # We write raw JSON-RPC (no SDK) so the test measures plain stdio.
        init_request = (
            '{"jsonrpc":"2.0","id":1,"method":"initialize",'
            '"params":{"protocolVersion":"2025-11-25",'
            '"capabilities":{},"clientInfo":{"name":"perf-test","version":"0"}}}\n'
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(init_request)
        proc.stdin.flush()

        # Read response line (blocks until server writes back)
        response_line = proc.stdout.readline()
        elapsed = time.perf_counter() - start

        assert '"result"' in response_line, f"unexpected init response: {response_line!r}"
        assert elapsed < SUBPROCESS_HANDSHAKE_CEILING_SECONDS, (
            f"subprocess initialize took {elapsed:.2f}s "
            f"(ceiling {SUBPROCESS_HANDSHAKE_CEILING_SECONDS}s)"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
