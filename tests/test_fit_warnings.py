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

import pathlib
import re

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


def test_model_selection_result_carries_fit_warnings_field() -> None:
    from econometrics.model_specification_diagnostics_robust_inference.model_selection.model_selection_model import (  # noqa: E501
        ModelSelectionResult,
    )

    assert "fit_warnings" in ModelSelectionResult.model_fields


def test_simultaneous_equations_result_carries_fit_warnings_field() -> None:
    from econometrics.model_specification_diagnostics_robust_inference.simultaneous_equations.simultaneous_equations_model import (  # noqa: E501
        SimultaneousEquationsResult,
    )

    assert "fit_warnings" in SimultaneousEquationsResult.model_fields


def test_simultaneous_equations_outer_failure_emits_warning() -> None:
    """When linearmodels fails entirely (e.g. degenerate dataset), the
    fallback must emit a warning rather than silently returning all zeros."""
    from econometrics.model_specification_diagnostics_robust_inference.simultaneous_equations.simultaneous_equations_model import (  # noqa: E501
        two_stage_least_squares,
    )

    # 2 equations, but x_data has only 1 observation → guaranteed to fail
    result = two_stage_least_squares(
        y_data=[[1.0], [2.0]],
        x_data=[[1.0, 2.0]],
        instruments=[[1.1, 0.9]],
    )
    assert result.fit_warnings, (
        f"expected a warning from the degenerate 3SLS fallback, got none; coefficients={result.coefficients!r}"
    )


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_no_silent_except_pass_in_production_code() -> None:
    """Regression guard: silently-swallowed exceptions hide bugs.

    Scans ``tools/`` and ``econometrics/`` (excluding test files) for the
    pattern ``except X: <nothing-but-pass>``. Narrow intentional swallows
    must at least catch a specific exception class — not ``Exception`` or
    a bare ``except``. This caught an ``except Exception: pass`` that was
    silently dropping any feature-importance failure in gradient_boosting.
    """
    offenders: list[str] = []
    for root in ("tools", "econometrics"):
        for path in (_REPO_ROOT / root).rglob("*.py"):
            # Skip test files — they can legitimately swallow for assertion flow
            if "/tests/" in str(path) or path.name.startswith("test_"):
                continue
            lines = path.read_text().splitlines()
            for i, line in enumerate(lines):
                # Only flag broad swallows: ``except:`` or ``except Exception:``
                # Specific-class catches like ``except (IndexError, AttributeError):`` are fine.
                m = re.match(r"(\s+)except(?:\s+Exception)?\s*:\s*$", line)
                if not m:
                    continue
                indent = m.group(1)
                # Look at the next meaningful line at a greater indent
                for j in range(i + 1, min(i + 6, len(lines))):
                    nxt = lines[j]
                    if not nxt.strip() or nxt.strip().startswith("#"):
                        continue
                    if not nxt.startswith(indent + "    "):
                        break
                    if nxt.strip() == "pass":
                        rel = path.relative_to(_REPO_ROOT)
                        offenders.append(f"{rel}:{i + 1}: {line.strip()}")
                    break
    assert not offenders, "silent broad except-pass sites:\n  " + "\n  ".join(offenders)


def test_cv_bailout_records_warning_instead_of_silent_none() -> None:
    """Known bail-out paths must record a warning so the caller can't mistake
    ``cv_score is None`` for "CV clean with no score"."""
    import numpy as np

    from econometrics.model_specification_diagnostics_robust_inference.model_selection.model_selection_model import (  # noqa: E501
        _cross_validation,
    )

    # Underdetermined: 5 rows, 6 features → must bail with a warning
    y = np.arange(5, dtype=np.float64)
    X = np.random.default_rng(0).normal(size=(5, 6))
    score, warnings = _cross_validation(y, X, folds=3)
    assert score is None
    assert warnings, "expected a warning from the underdetermined bail-out"
    assert any("underdetermined" in w for w in warnings)


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
