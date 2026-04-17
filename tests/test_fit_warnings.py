"""Regression tests for the ``fit_warnings`` contract.

Several algorithm modules contain fallback paths that substitute sentinel
values when the real statistic can't be computed (Hessian inversion fails,
IV numerically unstable, etc.). Before this was surfaced, users saw
``p_value = 1.0`` with no way to distinguish "really insignificant" from
"silent computation failure".

Each affected result model now carries a ``fit_warnings: list[str]`` field.
These tests assert the field exists on all affected models and that a
known-bad input triggers at least one warning.
"""

from __future__ import annotations

from aigroup_econ_mcp._registrations import load_all
from aigroup_econ_mcp.registry import REGISTRY


def _load() -> None:
    if len(REGISTRY) == 0:
        load_all()


def test_quantile_regression_result_carries_fit_warnings_field() -> None:
    from econometrics.nonparametric.quantile_regression import QuantileRegressionResult

    assert "fit_warnings" in QuantileRegressionResult.model_fields


def test_var_result_carries_fit_warnings_field() -> None:
    from econometrics.specific_data_modeling.time_series_panel_data.var_svar_model import (
        VARResult,
    )

    assert "fit_warnings" in VARResult.model_fields


def test_cox_regression_result_carries_fit_warnings_field() -> None:
    from econometrics.survival_analysis.survival_models import CoxRegressionResult

    assert "fit_warnings" in CoxRegressionResult.model_fields


def test_dynamic_panel_result_carries_fit_warnings_field() -> None:
    from econometrics.specific_data_modeling.time_series_panel_data.dynamic_panel_models import (
        DynamicPanelResult,
    )

    assert "fit_warnings" in DynamicPanelResult.model_fields


def test_var_model_always_emits_std_errors_placeholder_warning() -> None:
    """The VAR adapter currently returns placeholder (1, 0, 1) for
    std_errors / t_values / p_values regardless of whether statsmodels
    succeeded. Until that's properly wired, the warning must fire every
    time so users don't mistake placeholders for real statistics."""
    _load()
    from econometrics.specific_data_modeling.time_series_panel_data.var_svar_model import (
        var_model,
    )

    # var_model expects data[var_idx][time_idx] — one series per outer list.
    # Each series is slightly noisy to avoid statsmodels' constant-column rejection.
    series_a = [1.0 + 0.1 * i + (0.2 if i % 3 else -0.15) for i in range(24)]
    series_b = [2.0 - 0.05 * i + (0.1 if i % 2 else -0.08) for i in range(24)]
    result = var_model(data=[series_a, series_b], lags=1)
    placeholder_warning = any("placeholder" in w.lower() for w in result.fit_warnings)
    assert placeholder_warning, (
        f"expected a placeholder warning in fit_warnings, got {result.fit_warnings!r}"
    )
