"""Every-tool smoke coverage.

Calls each of the 66 registered tools with minimal tailored input. The
bar is deliberately low: a tool passes if it either (a) returns a
parseable JSON response, or (b) returns a structured ``{"ok": false,
"error": ...}`` payload. The test FAILS only when the tool raises an
uncaught Python exception, returns a non-string, or produces a
malformed body that can't be parsed.

This is the safety net the ML ``file_path`` bug (fixed in b47c5ba)
needed — it lived undetected because no test ever exercised those
8 tools with real data.

Tools with external dep failures (e.g. ``xgboost`` missing ``libomp``)
are caught by the registry's ``tool_unavailable`` stub and succeed
this test with a structured error. That's the correct outcome.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from aigroup_econ_mcp._registrations import load_all
from aigroup_econ_mcp.registry import REGISTRY

# --- minimal data helpers ----------------------------------------------------


def _y(n: int = 12) -> list[float]:
    """Monotonic noisy-ish response — safe for regression-style fits."""
    return [i + (0.1 if i % 2 else -0.1) for i in range(1, n + 1)]


def _x_1d(n: int = 12) -> list[list[float]]:
    return [[float(i)] for i in range(1, n + 1)]


def _x_2d(n: int = 12) -> list[list[float]]:
    return [[float(i), float(i * 2)] for i in range(1, n + 1)]


def _series(n: int = 30) -> list[float]:
    """Trending series with mild noise — enough for ARIMA/GARCH fits."""
    return [1.0 + 0.1 * i + (0.05 if i % 3 else -0.05) for i in range(n)]


def _binary_y(n: int = 20) -> list[int]:
    return [i % 2 for i in range(n)]


def _multi_series(n_periods: int = 16, n_vars: int = 2) -> list[list[float]]:
    return [[1.0 + 0.1 * t + 0.01 * v for v in range(n_vars)] for t in range(n_periods)]


def _panel(n_entities: int = 3, n_periods: int = 6) -> dict[str, Any]:
    y, x, e, t = [], [], [], []
    for i in range(n_entities):
        for p in range(n_periods):
            y.append(1.0 + 0.1 * p + 0.3 * i)
            x.append([float(p + 1), float(i + 1)])
            e.append(i + 1)
            t.append(p + 1)
    return {"y_data": y, "x_data": x, "entity_ids": e, "time_periods": t}


def _coords(n: int = 8) -> list[list[float]]:
    return [[float(i % 3), float(i // 3)] for i in range(n)]


# --- per-tool fixture -------------------------------------------------------


def _fixtures() -> dict[str, dict[str, Any]]:
    """Return kwargs for every tool.

    Kept explicit (rather than inferred from signatures) so that a tool
    whose fixture is wrong can be spot-debugged by reading the dict entry.
    """
    panel = _panel()

    return {
        # --- basic parametric ------------------------------------------------
        "basic_parametric_estimation_ols": {"y_data": _y(), "x_data": _x_1d()},
        "basic_parametric_estimation_mle": {"data": _series(40)},
        "basic_parametric_estimation_gmm": {
            "y_data": _y(),
            "x_data": _x_1d(),
            "instruments": [[i * 1.1] for i in range(1, 13)],
        },
        # --- causal inference ------------------------------------------------
        "causal_difference_in_differences": {
            "treatment": [0, 0, 1, 1, 0, 0, 1, 1] * 2,
            "time_period": [0, 1, 0, 1] * 4,
            "outcome": _y(16),
        },
        "causal_instrumental_variables": {
            "y_data": _y(),
            "x_data": _x_1d(),
            "instruments": [[i * 1.1] for i in range(1, 13)],
        },
        "causal_propensity_score_matching": {
            "treatment": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "outcome": _y(),
            "covariates": _x_2d(),
        },
        "causal_fixed_effects": {**panel},
        "causal_random_effects": {**panel},
        "causal_regression_discontinuity": {
            "running_variable": [float(i) - 6 for i in range(12)],
            "outcome": _y(),
            "cutoff": 0.0,
        },
        "causal_synthetic_control": {
            "outcome": _y(16),
            "treatment_period": 8,
            "treated_unit": "u1",
            "donor_units": ["u2", "u3"],
            "time_periods": list(range(1, 17)),
        },
        "causal_event_study": {
            "outcome": _y(16),
            "treatment": [0] * 8 + [1] * 8,
            "entity_ids": ["a"] * 8 + ["b"] * 8,
            "time_periods": list(range(1, 9)) * 2,
        },
        "causal_triple_difference": {
            "outcome": _y(16),
            "treatment_group": [0, 1] * 8,
            "time_period": [0, 0, 1, 1] * 4,
            "cohort_group": [0, 0, 0, 0, 1, 1, 1, 1] * 2,
        },
        "causal_mediation_analysis": {
            "outcome": _y(20),
            "treatment": [float(i % 2) for i in range(20)],
            "mediator": [float(i) * 0.5 for i in range(20)],
        },
        "causal_moderation_analysis": {
            "outcome": _y(20),
            "predictor": [float(i) for i in range(20)],
            "moderator": [float(i) * 0.3 for i in range(20)],
        },
        "causal_control_function": {
            "y_data": _y(),
            "x_data": [float(i) for i in range(1, 13)],
            "z_data": [[i * 1.1] for i in range(1, 13)],
        },
        "causal_first_difference": {
            "y_data": panel["y_data"],
            "x_data": [row[0] for row in panel["x_data"]],
            "entity_ids": [str(e) for e in panel["entity_ids"]],
        },
        # --- time series -----------------------------------------------------
        "time_series_arima_model": {"data": _series(40), "order": [1, 0, 0], "forecast_steps": 2},
        "time_series_exponential_smoothing": {
            "data": _series(40),
            "trend": True,
            "seasonal": False,
            "forecast_steps": 2,
        },
        "time_series_garch_model": {"data": _series(60), "order": [1, 1]},
        "time_series_unit_root_tests": {"data": _series(40), "test_type": "adf"},
        "time_series_var_svar_model": {
            "data": _multi_series(20, 2),
            "model_type": "var",
            "lags": 1,
        },
        "time_series_cointegration_analysis": {
            "data": _multi_series(20, 2),
            "analysis_type": "engle_granger",
        },
        "structural_break_tests": {"data": _series(40), "test_type": "chow", "break_point": 20},
        "time_varying_parameter_models": {
            "y_data": _y(20),
            "x_data": [[float(i)] for i in range(1, 21)],
            "model_type": "tar",
            "threshold_variable": [float(i) for i in range(20)],
            "n_regimes": 2,
        },
        # --- panel data ------------------------------------------------------
        "panel_data_dynamic_model": {**panel, "model_type": "diff_gmm", "lags": 1},
        "panel_data_diagnostics": {
            "test_type": "hausman",
            "fe_coefficients": [1.0, 2.0],
            "re_coefficients": [1.1, 2.1],
            "fe_covariance": [[0.1, 0.0], [0.0, 0.1]],
            "re_covariance": [[0.12, 0.0], [0.0, 0.12]],
        },
        "panel_var_model": {
            "data": _multi_series(18, 2),
            "entity_ids": [1] * 9 + [2] * 9,
            "time_periods": list(range(1, 10)) * 2,
            "lags": 1,
        },
        # --- machine learning (stub when xgboost/libomp missing) ------------
        "ml_random_forest": {"X_data": _x_2d(30), "y_data": _binary_y(30)},
        "ml_gradient_boosting": {"X_data": _x_2d(30), "y_data": _binary_y(30)},
        "ml_support_vector_machine": {"X_data": _x_2d(30), "y_data": _binary_y(30)},
        "ml_neural_network": {"X_data": _x_2d(30), "y_data": _binary_y(30)},
        "ml_kmeans_clustering": {"X_data": _x_2d(30), "n_clusters": 2},
        "ml_hierarchical_clustering": {"X_data": _x_2d(30), "n_clusters": 2},
        "ml_double_machine_learning": {
            "X_data": _x_2d(30),
            "y_data": _y(30),
            "treatment_data": [float(i % 2) for i in range(30)],
        },
        "ml_causal_forest": {
            "X_data": _x_2d(30),
            "y_data": _y(30),
            "treatment_data": [float(i % 2) for i in range(30)],
        },
        # --- microeconometrics ----------------------------------------------
        "micro_logit": {"X_data": _x_2d(20), "y_data": _binary_y(20)},
        "micro_probit": {"X_data": _x_2d(20), "y_data": _binary_y(20)},
        "micro_multinomial_logit": {"X_data": _x_2d(30), "y_data": [i % 3 for i in range(30)]},
        "micro_poisson": {"X_data": _x_2d(20), "y_data": list(range(1, 21))},
        "micro_negative_binomial": {"X_data": _x_2d(20), "y_data": list(range(1, 21))},
        "micro_tobit": {"X_data": _x_2d(20), "y_data": _y(20), "lower_bound": 0.0},
        "micro_heckman": {
            "X_select_data": _x_2d(30),
            "Z_data": _x_1d(30),
            "y_data": _y(30),
            "s_data": _binary_y(30),
        },
        # --- missing data ---------------------------------------------------
        "missing_data_simple_imputation": {
            "data": [[1.0, 2.0], [None, 4.0], [5.0, None]],
            "strategy": "mean",
        },
        "missing_data_multiple_imputation": {
            "data": [[1.0, 2.0], [None, 4.0], [5.0, None], [7.0, 8.0]],
            "n_imputations": 2,
            "max_iter": 3,
        },
        # --- model specification --------------------------------------------
        "model_diagnostic_tests": {"y_data": _y(), "x_data": _x_1d()},
        "generalized_least_squares": {"y_data": _y(), "x_data": _x_1d()},
        "weighted_least_squares": {"y_data": _y(), "x_data": _x_1d(), "weights": [1.0] * 12},
        "robust_errors_regression": {"y_data": _y(), "x_data": _x_1d()},
        "model_selection_criteria": {"y_data": _y(), "x_data": _x_1d(), "cv_folds": 3},
        "regularized_regression": {
            "y_data": _y(),
            "x_data": _x_1d(),
            "method": "ridge",
            "alpha": 0.1,
        },
        "simultaneous_equations_model": {
            "y_data": [_y(), _y()],
            "x_data": _x_2d(),
            "instruments": [[i * 1.1, i * 0.9] for i in range(1, 13)],
        },
        # --- nonparametric --------------------------------------------------
        "nonparametric_kernel_regression": {
            "y_data": _y(20),
            "x_data": [[float(i)] for i in range(1, 21)],
        },
        "nonparametric_quantile_regression": {
            "y_data": _y(20),
            "x_data": [[float(i)] for i in range(1, 21)],
            "quantile": 0.5,
        },
        "nonparametric_spline_regression": {
            "y_data": _y(20),
            "x_data": [[float(i)] for i in range(1, 21)],
            "n_knots": 4,
        },
        "nonparametric_gam_model": {
            "y_data": _y(40),
            "x_data": [[float(i), float(i * 2)] for i in range(1, 41)],
        },
        # --- spatial --------------------------------------------------------
        "spatial_weights_matrix": {"coordinates": _coords(8), "weight_type": "knn", "k": 2},
        "spatial_morans_i_test": {
            "values": [float(i) for i in range(8)],
            "neighbors": [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6]],
            "weights": [
                [1.0],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [1.0],
            ],
        },
        "spatial_gearys_c_test": {
            "values": [float(i) for i in range(8)],
            "neighbors": [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6]],
            "weights": [
                [1.0],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [1.0],
            ],
        },
        "spatial_local_moran_lisa": {
            "values": [float(i) for i in range(8)],
            "neighbors": [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6]],
            "weights": [
                [1.0],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [1.0],
            ],
        },
        "spatial_regression_model": {
            "y_data": [float(i) for i in range(8)],
            "x_data": [[float(i)] for i in range(8)],
            "neighbors": [[1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6]],
            "weights": [
                [1.0],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [1.0],
            ],
            "model_type": "sar",
        },
        "spatial_gwr_model": {
            "y_data": [float(i) for i in range(8)],
            "x_data": [[float(i)] for i in range(8)],
            "coordinates": _coords(8),
        },
        # --- statistical inference ------------------------------------------
        "inference_bootstrap": {"data": _series(40), "statistic_func": "mean", "n_bootstrap": 50},
        "inference_permutation_test": {
            "sample_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "sample_b": [2.0, 3.0, 4.0, 5.0, 6.0],
            "test_type": "mean_diff",
            "n_permutations": 100,
        },
        # --- distribution / decomposition -----------------------------------
        "decomposition_oaxaca_blinder": {
            "y_a": _y(12),
            "x_a": _x_1d(12),
            "y_b": [v + 1 for v in _y(12)],
            "x_b": _x_1d(12),
        },
        "decomposition_variance_anova": {
            "values": [1.0, 2.0, 1.5, 3.0, 4.0, 3.5, 5.0, 6.0, 5.5],
            "groups": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        },
        "decomposition_time_series": {"data": _series(48), "period": 12},
    }


# --- test harness -----------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _load_registry() -> None:
    if len(REGISTRY) == 0:
        load_all()


def _coverage_ids() -> list[str]:
    if len(REGISTRY) == 0:
        load_all()
    return sorted(REGISTRY.tools)


@pytest.mark.parametrize("tool_name", _coverage_ids())
def test_every_tool_invoked_with_minimal_input(tool_name: str) -> None:
    fixtures = _fixtures()
    if tool_name not in fixtures:
        pytest.fail(
            f"no fixture for {tool_name!r} — add one to _fixtures() so this tool "
            f"doesn't escape smoke coverage"
        )

    spec = REGISTRY.tools[tool_name]
    out = spec.handler(**fixtures[tool_name])

    assert isinstance(out, str), f"{tool_name}: handler must return str, got {type(out).__name__}"
    assert out, f"{tool_name}: empty response"
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError as exc:
        pytest.fail(f"{tool_name}: non-JSON response (first 200 chars): {out[:200]!r} ({exc})")

    # A handler may return:
    #   - a success payload (adapter-specific shape)
    #   - {"ok": false, "error": {...}} (structured failure via registry wrapper)
    #   - {"error": "..."} (legacy adapter-side error format — still valid)
    # Anything else is a contract violation.
    assert isinstance(parsed, (dict, list)), (
        f"{tool_name}: top-level JSON must be object/array, got {type(parsed).__name__}"
    )


def test_fixture_covers_every_registered_tool() -> None:
    """Guard: a new tool added without a fixture entry would quietly skip
    the parametrized test above unless we also assert the coverage is total."""
    fixtures = _fixtures()
    registered = set(REGISTRY.tools)
    missing = registered - set(fixtures)
    extra = set(fixtures) - registered
    assert not missing, f"tools missing smoke fixtures: {sorted(missing)}"
    assert not extra, f"fixtures for non-existent tools: {sorted(extra)}"
