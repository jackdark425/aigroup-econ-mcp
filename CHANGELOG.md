# Changelog

All notable changes to `aigroup-econ-mcp` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project uses [Semantic Versioning](https://semver.org/).

## [2.0.9] — 2026-04-17

Major internal refactor. The 66 MCP tool names and their parameters are
unchanged, so existing MCP clients continue to work.

### Changed
- Moved CLI and server into a real `aigroup_econ_mcp/` package; dropped the
  top-level `__init__.py`/`cli.py`/`server.py` scripts.
- Consolidated tool registration in `aigroup_econ_mcp/_registrations.py`.
  Removed the parallel `tools/mcp_tool_groups/` wrapper layer (12 files,
  ~3,400 lines) whose only purpose was to re-wrap adapter functions with
  ctx-logging boilerplate.
- Tightened dependency ranges in `pyproject.toml` with explicit upper bounds
  so upstream breaking changes don't silently land in the wheel.
- Wheel now ships only the three real packages (`aigroup_econ_mcp`,
  `econometrics`, `tools`) instead of the whole repo tree.

### Added
- `aigroup_econ_mcp.errors` with `ToolError`/`ValidationError`/
  `EstimationError` and a JSON error payload format. All tool failures now
  surface as `{"ok": false, "error": {"code", "message", "details"}}` instead
  of mixed string/JSON ad-hoc responses.
- `AIGROUP_ECON_MCP_DEBUG=1` environment variable to include tracebacks in
  tool error payloads.
- `docs/ARCHITECTURE.md`, `docs/PUBLISHING.md`, `docs/TESTING.md`, and this
  `CHANGELOG.md`.
- Smoke test verifying all 66 tools register cleanly.
- `ruff`, `mypy`, and `pytest` configuration under `[tool.*]` in
  `pyproject.toml`.

### Removed
- `tools/mcp_tool_groups/` — entire wrapper layer.
- `tools/mcp_tools_registry.py`, `tools/decorators.py`,
  `tools/time_series_panel_data_tools.py` — dead/duplicate registration code.
- `tools.decorators` imports and no-op `@econometric_tool`/`@validate_input`
  decorators from seven files under
  `econometrics/model_specification_diagnostics_robust_inference/`.
  This restores the one-way dependency: `econometrics/` has no knowledge of
  the tool layer.
- Root-level dev/debug scripts: `quick_test.py`, `check_pypi_version.py`,
  `clear_uvx_cache.{sh,bat,py}`.
- Root-level ad-hoc Chinese planning docs: `66个计量经济学MCP工具功能列表.md`,
  `VERSION_2.0.6_FIX_SUMMARY.md`,
  `econometrics/未开发大类优先级分析.md`.
- Committed `test_results/` and `test_data/` directories. Both paths are now
  in `.gitignore`.

### Fixed
- `cli.py` used `from __init__ import ...`, which only worked when run as a
  loose script. The new package entry point uses proper relative imports.
- `two_stage_least_squares` referenced an undefined name `equations` in its
  fallback path (caught by ruff F821). Replaced with the correct `x_data` /
  `constant` derivation.
- `multi_quantile_regression` used a list literal as a default argument
  (shared mutable state hazard). Replaced with `None` sentinel.
- 24 bare `except:` clauses tightened to `except Exception:` so process-
  level signals (`KeyboardInterrupt`, `SystemExit`) are not swallowed.
- 15 `zip()` calls now pass `strict=False` explicitly (Python 3.10+ linter
  compliance, no behaviour change).

### Tooling & quality
- `ruff check` now runs clean across all shipped code (`aigroup_econ_mcp/`,
  `tools/`, `tests/`, and non-test `econometrics/`). The remaining ~27
  warnings are in `econometrics/tests/` (not part of the wheel) and are
  covered by per-file-ignores in `pyproject.toml`.
- End-to-end smoke tests expanded to **35 tests** under `tests/test_e2e_tools.py`
  covering every one of the 11 tool groups, plus shim behaviour for 1D→2D
  coercion, input-shape rejection, and the structured error payload contract.
- Redundant dispatcher layer in `tools/output_formatter.py` replaced with a
  tiny factory pattern; `TextFormatter` deleted (45 lines of low-value
  Chinese one-liner summaries — now falls back to Pydantic dict str).
- `DataLoader` consolidated: `load_from_file()` for tabular inputs,
  `load_flat()` for univariate. `MLEDataLoader` kept as a 3-line shim for
  backwards import compatibility.
- New `merge_file_data(file_path, **defaults)` helper collapses the
  canonical "load from file if given, else keep caller args" pattern used
  across adapters. Applied to 13 sites in `causal_inference_adapter.py`.
- Removed dead `prompts/` directory (orphan FastMCP prompt templates never
  wired up anywhere).
- 32 sites now use proper `raise X from err`/`from None` exception chaining
  (was plain re-raise — tracebacks are now intact for debuggers).
- Dynamic panel fit-failure error payload normalized to the registry's
  `{"ok": false, "error": {...}}` schema; the rich diagnostic hints are
  preserved under `details.suggestions`.
