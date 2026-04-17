# Verification Methodology

How we test `aigroup-econ-mcp`. This document is the project's testing
paradigm — what each tier catches, what ground truth we demand, and the
record of real bugs this approach has caught.

Start here if you're adding a new tool, reviewing a PR, or wondering
whether a returned number is trustworthy.

## Testing pyramid

Four tiers, progressively more expensive and more exacting:

| Tier | File(s) | Count | Runtime | What it proves |
|------|---------|------:|--------:|----------------|
| 1. Registration shape | `tests/test_registry.py`, `tests/test_manifest.py` | 78 | 0.3 s | Every tool registers, has a unique snake_case name, a description, a callable handler, and a resolvable `module:function` target |
| 2. Smoke coverage | `tests/test_all_tools_smoke.py` | 67 | 1.3 s | Every tool accepts a minimal input and returns parseable JSON — either a success payload, `{"ok": false, "error": {...}}`, or a legacy `{"error": ...}` blob. The test fails only on uncaught exceptions, non-string returns, or malformed JSON |
| 3. Mathematical correctness | `tests/test_correctness.py` | 70 | 1.5 s | Every tool, driven by a known-truth synthetic DGP, returns numeric output within a declared tolerance of the truth |
| 4. Real MCP protocol e2e | `tests/test_mcp_protocol.py` *(slow)* | 6 | 8–11 s | Spawning `python -m aigroup_econ_mcp` as a subprocess, the full stdio JSON-RPC handshake + `tools/list` + `tools/call` round-trip works exactly as an MCP client would see |

Auxiliary:

| File | Count | What it catches |
|------|------:|-----------------|
| `tests/test_guide.py` | 6 | The dynamic `guide://econometrics` resource stays in sync with REGISTRY — no stale "21 tools" literals |
| `tests/test_fit_warnings.py` | 9 | Silent-fallback result models surface a `fit_warnings` field rather than silently substituting sentinel zeros |
| `tests/test_startup_perf.py` | 2 | `build_server()` < 5 s and subprocess-level `initialize` < 15 s — regression guard against eager imports of heavy libs |

**Total: 271 tests (264 fast + 7 slow) + 1 pygam skip**. `pytest -m "not slow"` runs the fast suite in ~1.5 s for rapid-iteration loops; CI runs the full suite in ~12 s.

## Tier 3: mathematical correctness in detail

This is the tier most of the value lives in. A tool that passes smoke
but returns wrong math is the real failure mode, and smoke won't catch
it.

### Design rule

For every tool, construct a synthetic DGP where the true parameter is
known analytically, drive the tool with a fixed RNG seed, and assert
the recovered statistic lies within tolerance of the truth.

### Tolerances — how tight is tight enough

Tests use generous tolerances that capture "the adapter returns the
wrong thing", not "the estimator could be 0.1% more accurate". Typical
bounds:

| Family | Tolerance used | Reasoning |
|--------|----------------|-----------|
| OLS / GLS / WLS / robust coefficients | ±0.1 absolute | σ=0.3, n=500 → SE ≈ 0.01–0.05; ×2–3 for safety |
| GMM / IV / 2SLS coefficients | ±0.15–0.25 | noisier IV variance inflates CI |
| Causal treatment effects (DID, PSM, DDD, RDD) | ±0.3–0.5 | finite-sample + matching bias |
| Mediation / moderation coefficients | ±0.15–0.2 | multi-step estimation |
| ARIMA / AR coefficients | range check [0.5, 0.9] for true 0.7 | statsmodels ML + small-sample bias |
| Classification accuracy (ML) | > 0.85–0.9 | on linearly-separable 2-D blobs |
| Clustering silhouette | > 0.5 | well-separated blobs |
| Spatial Moran's I | > 0.3 clustered, |·| < 0.5 random | strong/weak autocorr |
| Bootstrap CI | contains truth | unit coverage |

Over-tight bounds cause flaky CI and hide real regressions in noise;
over-loose bounds hide real bugs. The sweet spot is "3–5× the asymptotic
SE" for coefficient-recovery tests.

### Reproducibility

- Every correctness test uses `np.random.default_rng(seed)` with a
  distinct seed per test (101, 103, 107, ... — spread across primes to
  make accidental seed collision visible).
- No test relies on wall-clock time, system locale, or network.
- `tests/conftest.py` injects `libomp.dylib` via macOS-specific rpath
  patching so xgboost-based tests aren't quietly skipped on user-scope
  Homebrew installs. Linux CI bypasses this no-op.

### Stochastic-tool handling

Bootstrap, permutation, RF/GB accuracy, CF ATE — these are stochastic
by construction. We assert **distributional properties**, not point
values:

- Bootstrap: CI contains the true mean
- Permutation: p < 0.05 on shifted samples; p > 0.05 on iid samples
- RF/GB: train accuracy > 0.9 on separable classes
- Causal forest: ATE lands in `[1, 3]` when true τ = 2 on randomised treatment

### Ground-truth DGPs

Key DGPs in `tests/test_correctness.py`, named helpers:

| Helper | Shape | Used by |
|--------|-------|---------|
| `_linear_dgp(n, beta, sigma, seed)` | `y = const + β·x + ε` | OLS, GLS, WLS, robust, ridge, diagnostic |
| `_binary_dgp(n, beta, seed)` | `P(y=1) = σ(X·β)` | logit, probit, multinomial, neg-bin |
| `_separable_dgp(n, seed)` | Two well-spaced 2-D blobs | RF, GB, SVM, NN, k-means, hier. clust. |
| `_chain(n)` | Linear 0—1—…—(n-1) graph | Moran's I, Geary's C, LISA |
| `_panel(n_entities, n_periods)` | Balanced panel with entity FE | FE, RE, first-difference |

## Coverage audit

Every one of the **66 registered tools** is covered by at least one
correctness test. The audit that guards this:

```python
from aigroup_econ_mcp._registrations import load_all
from aigroup_econ_mcp.registry import REGISTRY

load_all()
src = open("tests/test_correctness.py").read()
uncovered = {t for t in REGISTRY.tools if f'"{t}"' not in src}
assert not uncovered, f"uncovered: {sorted(uncovered)}"
```

Run this after adding a new tool — if it fails, you haven't written its
correctness test yet.

## Bug-finding track record

The verification approach caught **9 real bugs** that smoke testing
alone would have missed. Each one was a silent failure: the tool
returned well-formed JSON, but the numbers were wrong.

| # | Commit | Bug | Detection layer |
|---|--------|-----|-----------------|
| 1 | early | `simultaneous_equations_model.py:185` referenced undefined name `equations` (was `x_data`) in fallback | Ruff F821 |
| 2 | 65ec070 | `multi_quantile_regression` had a mutable list literal as default — shared-state hazard | Ruff B006 |
| 3 | b47c5ba | **ML file_path silent KeyError** — all 8 ML tools read `data.get("X", data.get("features"))` but DataLoader returns `x_data`. Every file_path input silently produced `X_data=None` | Correctness + smoke regression test |
| 4 | a8a5f7c | Dynamic-panel IV→OLS fallback kept `model_type="Difference GMM (Arellano-Bond)"` — the label lied about which estimator actually ran | Code review |
| 5 | a8a5f7c | `simultaneous_equations` outer fallback shadowed the `equation_names` parameter with `equation_names = []`, making user-supplied names unreachable | Code inspection during fit_warnings work |
| 6 | a8a5f7c | **All 7 time-series file_path silently broken** — adapters called `DataLoader.load_from_file` (structured loader) then read `data_dict["data"]` (flat-loader key). Pure KeyError on any real file input | Correctness tests + shape audit |
| 7 | df697d2 | `arima_model` with `forecast_steps=0` crashed — statsmodels raised "Prediction must have `end` after `start`" | Correctness test |
| 8 | df697d2 | `regularized_regression` (ridge/lasso/elastic-net) **completely broken** — `float(scaler_y.mean_)` was `float()` of a 1-element numpy array, always raised `TypeError: only 0-dimensional arrays can be converted to Python scalars` | Correctness test |
| 9 | 7ec0efd | **Structural break tests were stubs** — `chow_test`, `quandt_andrews_test`, `bai_perron_test` all returned hardcoded `test_statistic=4.2 / 5.1 / 6.8` regardless of input. The tool claimed to detect structural breaks while completely ignoring the data | Correctness test (Chow F should be ≫ 10 on a 1→5 mean shift; returned 4.2) |

Five of the nine are tier-3 (correctness) finds; four are code-review
or static-check finds. Smoke alone caught zero — smoke's role is
regression-prevention after correctness has vetted the happy path.

## Adding a new tool

When you register a new MCP tool, you owe three tests, in order:

1. **Fixture in `tests/test_all_tools_smoke.py`** — add an entry to
   `_fixtures()` with minimal valid kwargs. The parametrized
   `test_every_tool_invoked_with_minimal_input[<your-tool>]` test is
   auto-generated from the REGISTRY; the coverage-guard test asserts
   every registered tool has a fixture.

2. **Correctness test in `tests/test_correctness.py`** — construct a
   DGP with a known true parameter, call the tool, assert recovered
   value within tolerance. Use an existing `_linear_dgp` /
   `_separable_dgp` / `_chain` helper if it fits; otherwise add a new
   helper at the top of its section.

3. **If the tool has non-trivial fallbacks** — add a
   `tests/test_fit_warnings.py` entry asserting the fallback surfaces
   through the `fit_warnings` field. See the `VARResult` /
   `CoxRegressionResult` / `ModelSelectionResult` / `DynamicPanelResult`
   / `SimultaneousEquationsResult` examples.

Local quick-check:

```bash
uv run pytest tests/ -m "not slow" -q    # ~1.5 s
uv run ruff check .                      # must be clean
```

Before pushing:

```bash
uv run pytest tests/ -q                  # full suite, ~12 s
uv build                                 # wheel must build
uv run twine check dist/*                # wheel must be PyPI-clean
```

## Key references

- `tests/conftest.py` — macOS libomp rpath shim
- `tests/test_correctness.py` — the ground-truth DGP tests (source of truth
  for coverage)
- `tests/test_all_tools_smoke.py::_fixtures` — the canonical minimal
  input for every tool (the same fixtures are consumed by the coverage
  guard)
- `aigroup_econ_mcp/errors.py` — the uniform `{"ok": false, "error":
  {"code", "message", "details"}}` error payload every correctness test
  relies on to distinguish "the tool worked and returned an error
  payload" from "the process crashed"
- `docs/ARCHITECTURE.md` — layer boundaries this testing strategy
  protects
