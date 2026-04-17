"""Central registration of all 66 econometrics tools.

We resolve handlers at registration time (FastMCP introspects the real
signature to generate JSON schemas, which rules out a transparent lazy
wrapper). But one broken dependency — e.g. xgboost missing libomp — should
not take down the rest of the 66 tools, so import failures are caught
per-module and those tools register a stub that returns a clear
``unavailable`` error on call. Everything else starts normally.
"""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Callable
from functools import cache
from typing import Any

from .registry import register

log = logging.getLogger(__name__)


# --- Eager-but-tolerant handler resolution ----------------------------------


@cache
def _import_module(module_name: str) -> Any:
    """Import a module, caching both success and failure.

    On failure, returns the exception so every tool in that module can report
    the same root cause without re-attempting the import.
    """
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 — we want every import-time failure
        log.warning("failed to import %s: %s", module_name, exc)
        return exc


def _resolve(target: str) -> Callable[..., Any]:
    """Return the callable at ``module:func``, or a stub if the module broke."""
    module_name, _, func_name = target.partition(":")
    module = _import_module(module_name)
    if isinstance(module, Exception):
        return _unavailable_stub(target, module)
    return getattr(module, func_name)


def _unavailable_stub(target: str, cause: Exception) -> Callable[..., str]:
    """Create a handler that explains why this tool is not loadable."""
    message = f"tool {target!r} unavailable: {type(cause).__name__}: {cause}"

    def stub(*args: Any, **kwargs: Any) -> str:
        return json.dumps(
            {"ok": False, "error": {"code": "tool_unavailable", "message": message}},
            ensure_ascii=False,
            indent=2,
        )

    stub.__name__ = f"unavailable_{target.rpartition(':')[2] or 'tool'}"
    stub.__doc__ = message
    return stub


# --- Shims for the 2 tools that need pre-adapter data coercion --------------


def _coerce_to_2d(data: Any, *, field: str) -> Any:
    if data is None or not data:
        return data
    first = data[0]
    if isinstance(first, (int, float)):
        return [list(data)]
    if isinstance(first, list) and (not first or isinstance(first[0], (int, float))):
        return data
    raise ValueError(f"{field!r} must be a 1D or 2D numeric list")


def _make_var_svar_shim() -> Callable[..., str]:
    adapter = _resolve("tools.time_series_panel_data_adapter:var_svar_adapter")

    def var_svar_tool(
        data: list[list[float]] | None = None,
        file_path: str | None = None,
        model_type: str = "var",
        lags: int = 1,
        variables: list[str] | None = None,
        a_matrix: list[list[float]] | None = None,
        b_matrix: list[list[float]] | None = None,
        output_format: str = "json",
        save_path: str | None = None,
    ) -> str:
        data = _coerce_to_2d(data, field="data")
        return adapter(
            data,
            file_path,
            model_type,
            lags,
            variables,
            a_matrix,
            b_matrix,
            output_format,
            save_path,
        )

    return var_svar_tool


def _make_cointegration_shim() -> Callable[..., str]:
    adapter = _resolve("tools.time_series_panel_data_adapter:cointegration_adapter")

    def cointegration_tool(
        data: list[list[float]] | None = None,
        file_path: str | None = None,
        analysis_type: str = "johansen",
        variables: list[str] | None = None,
        coint_rank: int = 1,
        output_format: str = "json",
        save_path: str | None = None,
    ) -> str:
        data = _coerce_to_2d(data, field="data")
        return adapter(
            data,
            file_path,
            analysis_type,
            variables,
            coint_rank,
            output_format,
            save_path,
        )

    return cointegration_tool


# --- The 66-tool manifest ---------------------------------------------------
#
# Each row: (name, group, target, description). ``target`` is either a
# ``module:func`` string (resolved lazily) or a callable (for shims).

_MANIFEST: tuple[tuple[str, str, Any, str], ...] = (
    # basic parametric
    (
        "basic_parametric_estimation_ols",
        "basic_parametric",
        "tools.econometrics_adapter:ols_adapter",
        "OLS Regression Analysis",
    ),
    (
        "basic_parametric_estimation_mle",
        "basic_parametric",
        "tools.econometrics_adapter:mle_adapter",
        "Maximum Likelihood Estimation",
    ),
    (
        "basic_parametric_estimation_gmm",
        "basic_parametric",
        "tools.econometrics_adapter:gmm_adapter",
        "Generalized Method of Moments",
    ),
    # causal inference
    (
        "causal_difference_in_differences",
        "causal_inference",
        "tools.causal_inference_adapter:did_adapter",
        "Difference-in-Differences (DID) Analysis",
    ),
    (
        "causal_instrumental_variables",
        "causal_inference",
        "tools.causal_inference_adapter:iv_adapter",
        "Instrumental Variables (IV/2SLS) Analysis",
    ),
    (
        "causal_propensity_score_matching",
        "causal_inference",
        "tools.causal_inference_adapter:psm_adapter",
        "Propensity Score Matching (PSM) Analysis",
    ),
    (
        "causal_fixed_effects",
        "causal_inference",
        "tools.causal_inference_adapter:fixed_effects_adapter",
        "Fixed Effects Model",
    ),
    (
        "causal_random_effects",
        "causal_inference",
        "tools.causal_inference_adapter:random_effects_adapter",
        "Random Effects Model",
    ),
    (
        "causal_regression_discontinuity",
        "causal_inference",
        "tools.causal_inference_adapter:rdd_adapter",
        "Regression Discontinuity Design (RDD)",
    ),
    (
        "causal_synthetic_control",
        "causal_inference",
        "tools.causal_inference_adapter:synthetic_control_adapter",
        "Synthetic Control Method",
    ),
    (
        "causal_event_study",
        "causal_inference",
        "tools.causal_inference_adapter:event_study_adapter",
        "Event Study Analysis",
    ),
    (
        "causal_triple_difference",
        "causal_inference",
        "tools.causal_inference_adapter:triple_difference_adapter",
        "Triple Difference (DDD) Analysis",
    ),
    (
        "causal_mediation_analysis",
        "causal_inference",
        "tools.causal_inference_adapter:mediation_adapter",
        "Mediation Effect Analysis",
    ),
    (
        "causal_moderation_analysis",
        "causal_inference",
        "tools.causal_inference_adapter:moderation_adapter",
        "Moderation Effect Analysis",
    ),
    (
        "causal_control_function",
        "causal_inference",
        "tools.causal_inference_adapter:control_function_adapter",
        "Control Function Approach",
    ),
    (
        "causal_first_difference",
        "causal_inference",
        "tools.causal_inference_adapter:first_difference_adapter",
        "First Difference Model",
    ),
    # time series & panel data
    (
        "time_series_arima_model",
        "time_series",
        "tools.time_series_panel_data_adapter:arima_adapter",
        "ARIMA Time Series Model",
    ),
    (
        "time_series_exponential_smoothing",
        "time_series",
        "tools.time_series_panel_data_adapter:exp_smoothing_adapter",
        "Exponential Smoothing Model",
    ),
    (
        "time_series_garch_model",
        "time_series",
        "tools.time_series_panel_data_adapter:garch_adapter",
        "GARCH Volatility Model",
    ),
    (
        "time_series_unit_root_tests",
        "time_series",
        "tools.time_series_panel_data_adapter:unit_root_adapter",
        "Unit Root Tests (ADF/PP/KPSS)",
    ),
    ("time_series_var_svar_model", "time_series", _make_var_svar_shim(), "VAR/SVAR Model"),
    (
        "time_series_cointegration_analysis",
        "time_series",
        _make_cointegration_shim(),
        "Cointegration Analysis",
    ),
    (
        "panel_data_dynamic_model",
        "time_series",
        "tools.time_series_panel_data_adapter:dynamic_panel_adapter",
        "Dynamic Panel Data Model",
    ),
    (
        "panel_data_diagnostics",
        "time_series",
        "tools.time_series_panel_data_adapter:panel_diagnostics_adapter",
        "Panel Data Diagnostic Tests",
    ),
    (
        "panel_var_model",
        "time_series",
        "tools.time_series_panel_data_adapter:panel_var_adapter",
        "Panel VAR Model",
    ),
    (
        "structural_break_tests",
        "time_series",
        "tools.time_series_panel_data_adapter:structural_break_adapter",
        "Structural Break Tests",
    ),
    (
        "time_varying_parameter_models",
        "time_series",
        "tools.time_series_panel_data_adapter:time_varying_parameter_adapter",
        "Time-Varying Parameter Models",
    ),
    # machine learning
    (
        "ml_random_forest",
        "machine_learning",
        "tools.machine_learning_adapter:random_forest_adapter",
        "Random Forest Analysis (Regression/Classification)",
    ),
    (
        "ml_gradient_boosting",
        "machine_learning",
        "tools.machine_learning_adapter:gradient_boosting_adapter",
        "Gradient Boosting Machine Analysis",
    ),
    (
        "ml_support_vector_machine",
        "machine_learning",
        "tools.machine_learning_adapter:svm_adapter",
        "Support Vector Machine Analysis",
    ),
    (
        "ml_neural_network",
        "machine_learning",
        "tools.machine_learning_adapter:neural_network_adapter",
        "Neural Network (MLP) Analysis",
    ),
    (
        "ml_kmeans_clustering",
        "machine_learning",
        "tools.machine_learning_adapter:kmeans_clustering_adapter",
        "K-Means Clustering Analysis",
    ),
    (
        "ml_hierarchical_clustering",
        "machine_learning",
        "tools.machine_learning_adapter:hierarchical_clustering_adapter",
        "Hierarchical Clustering Analysis",
    ),
    (
        "ml_double_machine_learning",
        "machine_learning",
        "tools.machine_learning_adapter:double_ml_adapter",
        "Double/Debiased Machine Learning for Causal Inference",
    ),
    (
        "ml_causal_forest",
        "machine_learning",
        "tools.machine_learning_adapter:causal_forest_adapter",
        "Causal Forest for Heterogeneous Treatment Effects",
    ),
    # microeconometrics
    (
        "micro_logit",
        "microecon",
        "tools.microecon_adapter:logit_adapter",
        "Logistic Regression Model",
    ),
    (
        "micro_probit",
        "microecon",
        "tools.microecon_adapter:probit_adapter",
        "Probit Regression Model",
    ),
    (
        "micro_multinomial_logit",
        "microecon",
        "tools.microecon_adapter:multinomial_logit_adapter",
        "Multinomial Logit Model",
    ),
    (
        "micro_poisson",
        "microecon",
        "tools.microecon_adapter:poisson_adapter",
        "Poisson Regression Model",
    ),
    (
        "micro_negative_binomial",
        "microecon",
        "tools.microecon_adapter:negative_binomial_adapter",
        "Negative Binomial Regression Model",
    ),
    (
        "micro_tobit",
        "microecon",
        "tools.microecon_adapter:tobit_adapter",
        "Tobit Model (Censored Regression)",
    ),
    (
        "micro_heckman",
        "microecon",
        "tools.microecon_adapter:heckman_adapter",
        "Heckman Selection Model",
    ),
    # missing data
    (
        "missing_data_simple_imputation",
        "missing_data",
        "tools.missing_data_adapter:simple_imputation_adapter",
        "Simple Imputation (Mean/Median/Mode/Constant)",
    ),
    (
        "missing_data_multiple_imputation",
        "missing_data",
        "tools.missing_data_adapter:multiple_imputation_adapter",
        "Multiple Imputation (MICE)",
    ),
    # model specification / robust inference
    (
        "model_diagnostic_tests",
        "model_specification",
        "tools.model_specification_adapter:diagnostic_tests_adapter",
        "Model Diagnostic Tests (Heteroskedasticity, Autocorrelation, Normality, VIF)",
    ),
    (
        "generalized_least_squares",
        "model_specification",
        "tools.model_specification_adapter:gls_adapter",
        "Generalized Least Squares (GLS) Regression",
    ),
    (
        "weighted_least_squares",
        "model_specification",
        "tools.model_specification_adapter:wls_adapter",
        "Weighted Least Squares (WLS) Regression",
    ),
    (
        "robust_errors_regression",
        "model_specification",
        "tools.model_specification_adapter:robust_errors_adapter",
        "Robust Standard Errors Regression (Heteroskedasticity-Robust)",
    ),
    (
        "model_selection_criteria",
        "model_specification",
        "tools.model_specification_adapter:model_selection_adapter",
        "Model Selection Criteria (AIC, BIC, HQIC, Cross-Validation)",
    ),
    (
        "regularized_regression",
        "model_specification",
        "tools.model_specification_adapter:regularization_adapter",
        "Regularized Regression (Ridge, LASSO, Elastic Net)",
    ),
    (
        "simultaneous_equations_model",
        "model_specification",
        "tools.model_specification_adapter:simultaneous_equations_adapter",
        "Simultaneous Equations Model (2SLS)",
    ),
    # nonparametric
    (
        "nonparametric_kernel_regression",
        "nonparametric",
        "tools.nonparametric_adapter:kernel_regression_adapter",
        "Kernel Regression (Nonparametric)",
    ),
    (
        "nonparametric_quantile_regression",
        "nonparametric",
        "tools.nonparametric_adapter:quantile_regression_adapter",
        "Quantile Regression",
    ),
    (
        "nonparametric_spline_regression",
        "nonparametric",
        "tools.nonparametric_adapter:spline_regression_adapter",
        "Spline Regression",
    ),
    (
        "nonparametric_gam_model",
        "nonparametric",
        "tools.nonparametric_adapter:gam_adapter",
        "Generalized Additive Model (GAM)",
    ),
    # spatial econometrics
    (
        "spatial_weights_matrix",
        "spatial_econometrics",
        "tools.spatial_econometrics_adapter:spatial_weights_adapter",
        "Spatial Weights Matrix Construction",
    ),
    (
        "spatial_morans_i_test",
        "spatial_econometrics",
        "tools.spatial_econometrics_adapter:morans_i_adapter",
        "Moran's I Spatial Autocorrelation Test",
    ),
    (
        "spatial_gearys_c_test",
        "spatial_econometrics",
        "tools.spatial_econometrics_adapter:gearys_c_adapter",
        "Geary's C Spatial Autocorrelation Test",
    ),
    (
        "spatial_local_moran_lisa",
        "spatial_econometrics",
        "tools.spatial_econometrics_adapter:local_moran_adapter",
        "Local Moran's I (LISA) Analysis",
    ),
    (
        "spatial_regression_model",
        "spatial_econometrics",
        "tools.spatial_econometrics_adapter:spatial_regression_adapter",
        "Spatial Regression Models (SAR/SEM/SDM)",
    ),
    (
        "spatial_gwr_model",
        "spatial_econometrics",
        "tools.spatial_econometrics_adapter:gwr_adapter",
        "Geographically Weighted Regression (GWR)",
    ),
    # statistical inference
    (
        "inference_bootstrap",
        "statistical_inference",
        "tools.statistical_inference_adapter:bootstrap_adapter",
        "Bootstrap Resampling Inference",
    ),
    (
        "inference_permutation_test",
        "statistical_inference",
        "tools.statistical_inference_adapter:permutation_test_adapter",
        "Permutation Test (Nonparametric)",
    ),
    # distribution / decomposition
    (
        "decomposition_oaxaca_blinder",
        "distribution_analysis",
        "tools.distribution_analysis_adapter:oaxaca_blinder_adapter",
        "Oaxaca-Blinder Decomposition",
    ),
    (
        "decomposition_variance_anova",
        "distribution_analysis",
        "tools.distribution_analysis_adapter:variance_decomposition_adapter",
        "Variance Decomposition (ANOVA)",
    ),
    (
        "decomposition_time_series",
        "distribution_analysis",
        "tools.distribution_analysis_adapter:time_series_decomposition_adapter",
        "Time Series Decomposition (Trend-Seasonal-Random)",
    ),
)


def load_all() -> None:
    """Register every tool in :data:`_MANIFEST`. Idempotent."""
    from .registry import REGISTRY

    existing = set(REGISTRY.tools)
    for name, group, target, description in _MANIFEST:
        if name in existing:
            continue
        handler = target if callable(target) else _resolve(target)
        register(name, handler, description, group=group)
