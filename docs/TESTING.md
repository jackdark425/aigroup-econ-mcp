# Testing

The repo has 121 tests across six files, split between fast in-process checks
and slow subprocess-based MCP-protocol checks.

## Run the fast suite (dev iteration)

```bash
uv run pytest -m "not slow"
```

~1.3s, 114 tests. Skips the subprocess spawn tests in
`tests/test_mcp_protocol.py`.

## Run everything (CI)

```bash
uv run pytest
```

~12s, 121 tests. CI runs this on every push and PR.

## Test layout

| File | Count | What it covers |
|------|------:|----------------|
| `tests/test_registry.py` | 6 | Registration shape invariants |
| `tests/test_e2e_tools.py` | 36 | In-process tool invocations across all 11 groups |
| `tests/test_manifest.py` | 72 | Static manifest validation (6 global + 66 per-tool) |
| `tests/test_guide.py` | 6 | Dynamic `guide://econometrics` resource content |
| `tests/test_startup_perf.py` | 2 | Cold `build_server()` and subprocess handshake latency |
| `tests/test_mcp_protocol.py` | 6 *(slow)* | Real JSON-RPC over stdio via the MCP client SDK |

## Manual smoke checks

### Via `uvx` (user-style)

```bash
uvx aigroup-econ-mcp --version
uvx aigroup-econ-mcp         # starts stdio MCP server
```

### Via local wheel (release candidate)

```bash
uv build
uv run --isolated --no-project --with dist/aigroup_econ_mcp-*.whl \
    --with mcp python -c "
import asyncio, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command=sys.executable, args=['-m', 'aigroup_econ_mcp'])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools
            print(f'{len(tools)} tools registered')

asyncio.run(main())
"
```

Expected output: `66 tools registered`.

## Debugging tool failures

Set `AIGROUP_ECON_MCP_DEBUG=1` to include Python tracebacks in the
structured error payload:

```bash
AIGROUP_ECON_MCP_DEBUG=1 aigroup-econ-mcp
```

Any tool failure will then emit:

```json
{"ok": false, "error": {"code": "...", "message": "...", "details": {}, "traceback": "..."}}
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs:
- `ruff check` + `ruff format --check`
- `pytest` on Python 3.10 / 3.11 / 3.12
- `uv build` + `twine check`

See the workflow file for the full pipeline.
