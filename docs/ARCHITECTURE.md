# Architecture

`aigroup-econ-mcp` exposes 66 econometrics tools over the Model Context
Protocol. The codebase is three layers — keep dependencies flowing *downward*
only.

```
┌──────────────────────────────────────────────────────┐
│  aigroup_econ_mcp/        (MCP server & CLI)         │
│    ├── cli.py             argparse entry             │
│    ├── server.py          FastMCP wire-up            │
│    ├── registry.py        ToolSpec registry          │
│    ├── _registrations.py  the ONE place tools are    │
│    │                      mapped to MCP names        │
│    └── errors.py          JSON error payloads        │
└──────────────────────────────────────────────────────┘
                      │ imports
                      ▼
┌──────────────────────────────────────────────────────┐
│  tools/                   (adapter layer)            │
│    ├── *_adapter.py       I/O + formatting, one      │
│    │                      module per domain          │
│    ├── data_loader.py     CSV/JSON/Excel/TXT input   │
│    └── output_formatter.py                           │
└──────────────────────────────────────────────────────┘
                      │ imports
                      ▼
┌──────────────────────────────────────────────────────┐
│  econometrics/            (algorithms)               │
│    ├── basic_parametric_estimation/                  │
│    ├── causal_inference/                             │
│    ├── specific_data_modeling/                       │
│    ├── model_specification_.../                      │
│    └── ...                                           │
└──────────────────────────────────────────────────────┘
```

## Adding a new tool

1. Implement the algorithm under `econometrics/<domain>/<name>/`. Return a
   Pydantic result model.
2. Add a thin adapter in the matching `tools/<domain>_adapter.py` that
   accepts MCP-style arguments (direct lists **or** `file_path`), loads data
   via `DataLoader`, calls the algorithm, and returns a JSON string.
3. Register the adapter in
   `aigroup_econ_mcp/_registrations.py` with a unique `name`, a short
   `description`, and the matching `group` key.

No other wiring is required. `server.build_server()` picks up the new tool on
the next start.

## Error handling contract

Adapters may raise freely. The registry wraps every handler with
`aigroup_econ_mcp.errors.format_exception`, which returns:

```json
{
  "ok": false,
  "error": {
    "code": "ValidationError",
    "message": "...",
    "details": {},
    "traceback": "..."   // only when AIGROUP_ECON_MCP_DEBUG=1
  }
}
```

`ValidationError` and `EstimationError` in `aigroup_econ_mcp/errors.py` are
available for adapters that want to differentiate error classes; any other
exception is relayed with its type name as the `code`.

## Invariants

- `tools/` MUST NOT import `aigroup_econ_mcp.*` — the server depends on the
  adapters, not the reverse.
- `econometrics/` MUST NOT import `tools.*` — algorithms are pure.
  (Historically seven files did; that reverse dependency was removed in 2.0.9.)
- Tool names are part of the public MCP contract. Don't rename them without
  a major version bump.
