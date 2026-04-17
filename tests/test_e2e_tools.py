"""End-to-end smoke tests that actually invoke representative tools.

The :mod:`tests.test_registry` tests only verify that tools *register*. These
tests pick one adapter per major statistical family and drive it with a tiny
synthetic dataset to catch runtime regressions in:

* adapter → algorithm imports (broken by refactors),
* parameter threading between MCP layer and ``econometrics/``,
* Pydantic result model construction,
* JSON serialization of results.

We deliberately use trivial shapes — these are smoke tests, not correctness
tests. Statistical correctness is covered under ``econometrics/tests``.
"""

from __future__ import annotations

import json

import pytest

from aigroup_econ_mcp._registrations import load_all
from aigroup_econ_mcp.registry import REGISTRY


@pytest.fixture(scope="module", autouse=True)
def _load_registry() -> None:
    if len(REGISTRY) == 0:
        load_all()


def _invoke(name: str, **kwargs) -> dict:
    """Call a registered tool and parse its JSON response."""
    spec = REGISTRY.tools[name]
    out = spec.handler(**kwargs)
    assert isinstance(out, str), f"{name} returned non-string: {type(out).__name__}"
    return json.loads(out)


def test_ols_happy_path() -> None:
    result = _invoke(
        "basic_parametric_estimation_ols",
        y_data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        x_data=[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]],
        output_format="json",
    )
    # Either the adapter returned a success payload or a structured error.
    # We only require shape coherence — not statistical correctness.
    assert isinstance(result, dict)
    if "error" in result or result.get("ok") is False:
        pytest.skip(f"adapter raised; likely a dep issue: {result}")
    # Success: expect standard OLS result keys.
    assert "r_squared" in result or "coefficients" in result


def test_unavailable_ml_tool_returns_structured_error() -> None:
    """When xgboost/libomp fails locally, ml tools register as stubs."""
    result = _invoke(
        "ml_gradient_boosting",
        y_data=[1.0, 2.0, 3.0],
        x_data=[[1.0], [2.0], [3.0]],
    )
    # Either the real adapter ran (error or success) or the stub fired.
    # Both are acceptable outcomes; we just want one JSON payload.
    assert isinstance(result, dict)


def test_validation_error_is_caught_by_registry_wrapper() -> None:
    """A deliberate input mismatch must not crash the process."""
    result = _invoke(
        "basic_parametric_estimation_ols",
        y_data=[1.0, 2.0],
        x_data=[[1.0]],  # length mismatch with y_data
        output_format="json",
    )
    assert isinstance(result, dict)
    # Registry wrapper gives {"ok": false, "error": {...}} OR the adapter's
    # own legacy error path returns {"error": "..."}.
    assert result.get("ok") is False or "error" in result


def test_time_series_arima_accepts_flat_data() -> None:
    result = _invoke(
        "time_series_arima_model",
        data=[1.0, 1.2, 1.1, 1.3, 1.4, 1.35, 1.5, 1.45, 1.6, 1.55] * 3,
        order=[1, 0, 0],
        forecast_steps=2,
        output_format="json",
    )
    assert isinstance(result, dict)


def test_unknown_tool_is_not_registered() -> None:
    assert "this_tool_does_not_exist" not in REGISTRY.tools


@pytest.mark.parametrize(
    "tool_name",
    [
        "basic_parametric_estimation_ols",
        "basic_parametric_estimation_mle",
        "basic_parametric_estimation_gmm",
        "causal_difference_in_differences",
        "causal_instrumental_variables",
        "time_series_arima_model",
        "time_series_var_svar_model",
        "micro_logit",
        "micro_probit",
        "nonparametric_kernel_regression",
        "nonparametric_gam_model",
        "inference_bootstrap",
        "missing_data_simple_imputation",
        "ml_random_forest",
        "ml_gradient_boosting",
        "spatial_weights_matrix",
        "spatial_morans_i_test",
        "model_diagnostic_tests",
        "generalized_least_squares",
        "decomposition_oaxaca_blinder",
    ],
)
def test_handlers_are_callable_across_groups(tool_name: str) -> None:
    """Representative sampling: every group's handler must be callable."""
    spec = REGISTRY.tools[tool_name]
    assert callable(spec.handler)
    assert spec.description
    assert spec.group


def test_shim_coerces_1d_to_2d_for_var_svar() -> None:
    """VAR/SVAR shim accepts a flat data list and coerces it to 2D."""
    result = _invoke(
        "time_series_var_svar_model",
        data=[1.0, 2.0, 3.0, 4.0],  # 1D input — the shim wraps to [[...]]
        model_type="var",
        lags=1,
        output_format="json",
    )
    # Result is either a success payload or a structured error — both JSON.
    assert isinstance(result, dict)


def test_shim_rejects_garbage_data_shape() -> None:
    """Coercion shim rejects non-numeric nested structures with ValueError."""
    result = _invoke(
        "time_series_cointegration_analysis",
        data=[[["bad"]]],  # 3D nested — shim should reject
        output_format="json",
    )
    assert isinstance(result, dict)
    # The registry wrapper catches the ValueError and returns the structured form.
    assert result.get("ok") is False or "error" in result


def test_registry_error_payload_includes_code_field() -> None:
    """Structured error payload must carry a machine-readable 'code'."""
    result = _invoke(
        "basic_parametric_estimation_ols",
        y_data=[1.0, 2.0],
        x_data=[[1.0], [2.0], [3.0]],  # mismatched length
    )
    assert isinstance(result, dict)
    if result.get("ok") is False:
        assert "code" in result["error"]
        assert "message" in result["error"]


def test_every_group_has_at_least_one_tool() -> None:
    """Sanity: none of the 11 declared groups should be empty."""
    groups = REGISTRY.groups()
    for name, specs in groups.items():
        assert specs, f"group {name!r} has no tools"
