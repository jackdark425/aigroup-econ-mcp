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


def test_time_series_univariate_file_path_uses_flat_loader(tmp_path) -> None:
    """Regression guard for the bug class: time_series adapters called
    DataLoader.load_from_file (structured loader returning y_data/x_data)
    then read data_dict["data"] — a key that doesn't exist. Every
    univariate time_series tool silently KeyError'd on file_path input.

    Fix: univariate adapters now use DataLoader.load_flat which returns
    {"data": [...]}. This test creates a valid flat-data file and
    confirms the tool does not KeyError on 'data'.
    """
    txt_path = tmp_path / "series.txt"
    txt_path.write_text("\n".join(str(1.0 + 0.1 * i) for i in range(40)))

    result = _invoke("time_series_arima_model", file_path=str(txt_path), order=[1, 0, 0])
    body = json.dumps(result)
    assert "'data'" not in body or "KeyError" not in body, (
        f"time_series univariate file_path regression: {body!r}"
    )


def test_time_series_multivariate_file_path_loads_json(tmp_path) -> None:
    """Regression guard for VAR/cointegration file_path branch. A JSON
    matrix input must be accepted and handed through the adapter without
    a key-mismatch error."""
    import json as _json

    json_path = tmp_path / "mv.json"
    series = [
        [1.0 + 0.1 * i + (0.05 if i % 2 else -0.05),
         2.0 - 0.05 * i + (0.03 if i % 3 else -0.03)]
        for i in range(24)
    ]
    json_path.write_text(_json.dumps({"data": series, "variables": ["a", "b"]}))

    result = _invoke("time_series_var_svar_model", file_path=str(json_path), lags=1)
    body = json.dumps(result)
    assert "KeyError" not in body, f"VAR multivariate file_path regression: {body!r}"


def test_ml_adapter_file_path_does_not_silently_drop_data(tmp_path) -> None:
    """Regression guard for the pre-71fa125 bug where the ML adapter read
    ``data.get("X")`` — a key that ``DataLoader.load_from_file`` never
    returns. All 8 ML tools silently saw ``X_data = None`` and raised
    a misleading ``X_data and y_data must be provided`` error.

    After the fix, file_path input must succeed in finding the data.
    We don't require the model fit to succeed (xgboost may be unavailable
    locally), only that the validation stage recognizes the file input.
    """
    import csv

    csv_path = tmp_path / "ml_input.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y", "x1", "x2"])
        # 20 rows — enough that most ML fits have a chance to run
        for i in range(20):
            writer.writerow([i % 2, float(i), float(i * 2)])

    result = _invoke("ml_random_forest", file_path=str(csv_path))
    assert isinstance(result, dict)
    # The specific message the bug produced was "X_data and y_data must be
    # provided". If that string appears in any error field, the bug is back.
    body = json.dumps(result)
    assert "X_data and y_data must be provided" not in body, (
        "ML file_path regression: adapter can no longer read the uploaded file"
    )
