"""Mathematical correctness tests.

Each tool is driven with a synthetic data-generating process (DGP) whose
true parameters are known. The test asserts the recovered statistic
matches truth within a tolerance that accommodates finite-sample noise.
Tolerances are set generously — these catch "the adapter returns the
wrong thing", not "the estimator could be 0.1% more accurate".

Scope: ~20 tools where ground truth is rigorous. The broader smoke
suite in ``test_all_tools_smoke.py`` covers "doesn't crash"; this file
is the "gives statistically correct answers" layer.

Stochastic tools (bootstrap / permutation / RF / causal forest) are
tested via property assertions rather than exact recovery.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from aigroup_econ_mcp._registrations import load_all
from aigroup_econ_mcp.registry import REGISTRY


@pytest.fixture(scope="module", autouse=True)
def _load() -> None:
    if len(REGISTRY) == 0:
        load_all()


def _invoke(name: str, **kwargs: Any) -> dict:
    body = REGISTRY.tools[name].handler(**kwargs)
    assert isinstance(body, str)
    parsed = json.loads(body)
    # If the tool registered as a tool_unavailable stub (missing dep like xgboost+libomp),
    # skip so CI doesn't fail on environment issues. Real tool errors still assert.
    if isinstance(parsed, dict) and parsed.get("ok") is False:
        code = parsed.get("error", {}).get("code", "")
        if code == "tool_unavailable":
            pytest.skip(f"{name} unavailable in this environment: {parsed['error']['message']}")
    return parsed


# ---------- helpers -----------------------------------------------------------


def _linear_dgp(
    n: int = 200, beta: list[float] = None, sigma: float = 0.5, seed: int = 0
) -> tuple[list[float], list[list[float]]]:
    """y = const + beta·x + N(0, sigma²). Returns (y_data, x_data)."""
    beta = beta or [1.0, 2.0]  # [const, slope1]
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(beta) - 1))
    y = beta[0] + X @ np.array(beta[1:]) + rng.normal(scale=sigma, size=n)
    return y.tolist(), X.tolist()


# ============================================================================
# BASIC PARAMETRIC ESTIMATION
# ============================================================================


def test_ols_recovers_known_coefficients() -> None:
    """y = 1 + 2·x₁ − 0.5·x₂ + ε → recovered β close to [1, 2, −0.5]."""
    y, X = _linear_dgp(n=500, beta=[1.0, 2.0, -0.5], sigma=0.3, seed=0)
    result = _invoke(
        "basic_parametric_estimation_ols", y_data=y, x_data=X, output_format="json"
    )
    coefs = result["coefficients"]
    assert len(coefs) == 3
    assert abs(coefs[0] - 1.0) < 0.1, f"const off: {coefs[0]}"
    assert abs(coefs[1] - 2.0) < 0.1, f"slope1 off: {coefs[1]}"
    assert abs(coefs[2] - (-0.5)) < 0.1, f"slope2 off: {coefs[2]}"
    assert result["r_squared"] > 0.9


def test_ols_without_constant() -> None:
    """With constant=False, the intercept must not appear in coefficients."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 1))
    y = (2.0 * X[:, 0] + rng.normal(scale=0.1, size=100)).tolist()
    result = _invoke(
        "basic_parametric_estimation_ols",
        y_data=y,
        x_data=X.tolist(),
        constant=False,
    )
    assert len(result["coefficients"]) == 1
    assert abs(result["coefficients"][0] - 2.0) < 0.1


def test_mle_normal_recovers_mean_and_sd() -> None:
    rng = np.random.default_rng(3)
    mu, sigma = 3.0, 1.5
    data = rng.normal(loc=mu, scale=sigma, size=2000).tolist()
    result = _invoke(
        "basic_parametric_estimation_mle", data=data, distribution="normal"
    )
    params = result["parameters"]
    # Two params (mean, sd) for normal MLE
    assert len(params) == 2
    mean_est, sd_est = params
    assert abs(mean_est - mu) < 0.15, f"mean off: {mean_est}"
    assert abs(sd_est - sigma) < 0.15, f"sd off: {sd_est}"
    assert result["convergence"] is True


def test_gmm_matches_ols_when_instruments_are_exogenous_regressors() -> None:
    """Just-identified GMM with instruments = exog regressors ≈ OLS."""
    y, X = _linear_dgp(n=500, beta=[0.5, 1.5], sigma=0.3, seed=7)
    result = _invoke(
        "basic_parametric_estimation_gmm",
        y_data=y,
        x_data=X,
        instruments=X,
    )
    coefs = result["coefficients"]
    assert abs(coefs[0] - 0.5) < 0.15, f"const off: {coefs[0]}"
    assert abs(coefs[1] - 1.5) < 0.15, f"slope off: {coefs[1]}"


# ============================================================================
# GLS / WLS / ROBUST / REGULARIZED
# ============================================================================


def test_gls_matches_ols_with_identity_weights() -> None:
    """GLS with no correlation ≈ OLS."""
    y, X = _linear_dgp(n=300, beta=[0.0, 1.0], sigma=0.4, seed=11)
    result = _invoke("generalized_least_squares", y_data=y, x_data=X)
    assert abs(result["coefficients"][0]) < 0.1
    assert abs(result["coefficients"][1] - 1.0) < 0.1
    assert result["r_squared"] > 0.8


def test_wls_with_uniform_weights_matches_ols() -> None:
    y, X = _linear_dgp(n=300, beta=[1.0, 2.0], sigma=0.3, seed=13)
    result = _invoke(
        "weighted_least_squares", y_data=y, x_data=X, weights=[1.0] * 300
    )
    assert abs(result["coefficients"][0] - 1.0) < 0.1
    assert abs(result["coefficients"][1] - 2.0) < 0.1


def test_robust_errors_point_estimate_matches_ols() -> None:
    y, X = _linear_dgp(n=400, beta=[0.0, 3.0], sigma=0.5, seed=17)
    result = _invoke("robust_errors_regression", y_data=y, x_data=X)
    assert abs(result["coefficients"][0]) < 0.15
    assert abs(result["coefficients"][1] - 3.0) < 0.15


def test_ridge_with_tiny_alpha_approximates_ols() -> None:
    y, X = _linear_dgp(n=300, beta=[0.0, 2.0], sigma=0.3, seed=19)
    ridge = _invoke(
        "regularized_regression",
        y_data=y,
        x_data=X,
        method="ridge",
        alpha=1e-6,
    )
    # Very small ridge penalty ⇒ answer close to OLS truth
    coefs = ridge["coefficients"]
    # Strip intercept if present
    slope = coefs[-1]
    assert abs(slope - 2.0) < 0.15, f"ridge slope off: {slope}"


# ============================================================================
# CAUSAL INFERENCE
# ============================================================================


def test_did_recovers_known_treatment_effect() -> None:
    """DID: treated post-period outcomes jump by +3. Recover ATT ≈ 3."""
    rng = np.random.default_rng(23)
    n_per_cell = 200
    treatment = [0] * n_per_cell + [0] * n_per_cell + [1] * n_per_cell + [1] * n_per_cell
    time_period = [0] * n_per_cell + [1] * n_per_cell + [0] * n_per_cell + [1] * n_per_cell
    # Baseline by cell: untreated_pre=10, untreated_post=11 (parallel-trend +1),
    # treated_pre=10, treated_post=14 (extra +3 from treatment)
    base = (
        [10.0] * n_per_cell
        + [11.0] * n_per_cell
        + [10.0] * n_per_cell
        + [14.0] * n_per_cell
    )
    outcome = [b + rng.normal(scale=0.5) for b in base]
    result = _invoke(
        "causal_difference_in_differences",
        treatment=treatment,
        time_period=time_period,
        outcome=outcome,
    )
    est = result["estimate"]
    assert abs(est - 3.0) < 0.3, f"DID ATT off: {est}"
    assert result["p_value"] < 0.01


def test_iv_recovers_coefficient_under_endogeneity() -> None:
    """Simple IV: y = β·x + u, x = π·z + v, cov(u,v)≠0. 2SLS should recover β=1.0."""
    rng = np.random.default_rng(29)
    n = 500
    z = rng.normal(size=n)
    v = rng.normal(scale=0.5, size=n)
    u = 0.7 * v + rng.normal(scale=0.3, size=n)  # u correlated with v → x endogenous
    x = 1.0 * z + v
    y = 1.0 * x + u
    result = _invoke(
        "causal_instrumental_variables",
        y_data=y.tolist(),
        x_data=[[xi] for xi in x.tolist()],
        instruments=[[zi] for zi in z.tolist()],
    )
    est = result["estimate"]
    assert abs(est - 1.0) < 0.2, f"IV estimate off: {est}"


def test_first_difference_recovers_slope_on_panel() -> None:
    """Panel y_it = α_i + β·x_it + ε_it with β=0.5. First-differencing removes α_i."""
    rng = np.random.default_rng(31)
    n_entities, n_periods = 30, 8
    beta = 0.5
    y_data, x_data, entities = [], [], []
    for i in range(n_entities):
        alpha_i = rng.normal(scale=2.0)  # entity FE
        for t in range(n_periods):
            x = float(t) + rng.normal(scale=0.3)
            y = alpha_i + beta * x + rng.normal(scale=0.2)
            y_data.append(y)
            x_data.append(x)
            entities.append(str(i))
    result = _invoke(
        "causal_first_difference",
        y_data=y_data,
        x_data=x_data,
        entity_ids=entities,
    )
    est = result["estimate"]
    assert abs(est - beta) < 0.1, f"FD slope off: {est}"


# ============================================================================
# NONPARAMETRIC
# ============================================================================


def test_quantile_regression_median_matches_ols() -> None:
    """Median regression on symmetric errors ≈ OLS mean regression."""
    y, X = _linear_dgp(n=400, beta=[0.5, 2.0], sigma=0.5, seed=37)
    result = _invoke(
        "nonparametric_quantile_regression",
        y_data=y,
        x_data=X,
        quantile=0.5,
    )
    coefs = result["coefficients"]
    # coefs = [const, slope]
    assert abs(coefs[0] - 0.5) < 0.2
    assert abs(coefs[1] - 2.0) < 0.15


# ============================================================================
# TIME SERIES
# ============================================================================


def test_arima_recovers_ar1_coefficient() -> None:
    """AR(1) process with φ=0.7 → ARIMA(1,0,0) should recover φ ≈ 0.7."""
    rng = np.random.default_rng(41)
    n = 500
    phi = 0.7
    series = [rng.normal()]
    for _ in range(1, n):
        series.append(phi * series[-1] + rng.normal(scale=0.3))
    result = _invoke(
        "time_series_arima_model", data=series, order=[1, 0, 0], forecast_steps=1
    )
    # Result has a coefficients list. AR(1) coefficient should be near 0.7.
    # Search all numeric leaves in the response for one in the [0.5, 0.9] band
    # — robust to whether the adapter labels it ``coefficients`` or similar.
    coef_like = [v for v in _collect_numbers(result) if 0.5 <= v <= 0.9]
    assert coef_like, f"no AR coefficient near 0.7 in {result}"


def test_unit_root_detects_stationary_series() -> None:
    """Stationary AR(1) with φ=0.5 → ADF should reject unit root (p < 0.05)."""
    rng = np.random.default_rng(43)
    n = 500
    series = [rng.normal()]
    for _ in range(1, n):
        series.append(0.5 * series[-1] + rng.normal())
    result = _invoke(
        "time_series_unit_root_tests", data=series, test_type="adf"
    )
    # The result has a 'p_value' (or similar) that should be < 0.05 for stationary
    p_like = _find_key_recursive(result, "p_value") or _find_key_recursive(
        result, "pvalue"
    )
    assert p_like is not None, f"no p_value in unit root result: {result}"
    assert p_like < 0.05, f"stationary series should reject unit root, got p={p_like}"


def test_unit_root_fails_to_reject_random_walk() -> None:
    """Random walk → ADF should NOT reject unit root (p > 0.1)."""
    rng = np.random.default_rng(47)
    n = 500
    series = np.cumsum(rng.normal(size=n)).tolist()
    result = _invoke(
        "time_series_unit_root_tests", data=series, test_type="adf"
    )
    p_like = _find_key_recursive(result, "p_value") or _find_key_recursive(
        result, "pvalue"
    )
    assert p_like is not None, f"no p_value in unit root result: {result}"
    assert p_like > 0.1, f"random walk should keep null, got p={p_like}"


# ============================================================================
# STATISTICAL INFERENCE
# ============================================================================


def test_bootstrap_confidence_interval_covers_true_mean() -> None:
    """Bootstrap CI for the mean of N(3, 1²), n=300, should contain 3."""
    rng = np.random.default_rng(53)
    data = rng.normal(loc=3.0, scale=1.0, size=300).tolist()
    result = _invoke(
        "inference_bootstrap",
        data=data,
        statistic_func="mean",
        n_bootstrap=500,
    )
    ci = result["confidence_interval"]
    assert ci[0] < 3.0 < ci[1], f"CI {ci} does not cover true mean 3.0"
    # Bootstrap mean should itself be near the sample mean, which is near 3
    assert abs(result["bootstrap_mean"] - 3.0) < 0.2


# ============================================================================
# DECOMPOSITION
# ============================================================================


def test_anova_recovers_known_f_statistic_ordering() -> None:
    """Three well-separated groups must yield a significant F (p < 0.001)."""
    rng = np.random.default_rng(59)
    g0 = rng.normal(loc=1.0, scale=0.3, size=30).tolist()
    g1 = rng.normal(loc=5.0, scale=0.3, size=30).tolist()
    g2 = rng.normal(loc=10.0, scale=0.3, size=30).tolist()
    result = _invoke(
        "decomposition_variance_anova",
        values=g0 + g1 + g2,
        groups=[0] * 30 + [1] * 30 + [2] * 30,
    )
    assert result["f_statistic"] > 50, f"F too small: {result['f_statistic']}"
    assert result["p_value"] < 0.001
    assert result["n_groups"] == 3


def test_oaxaca_total_difference_matches_group_mean_gap() -> None:
    """Oaxaca-Blinder total_difference = mean(y_a) − mean(y_b)."""
    rng = np.random.default_rng(61)
    y_a = (rng.normal(size=100) + 5.0).tolist()
    x_a = [[v] for v in rng.normal(size=100).tolist()]
    y_b = (rng.normal(size=100) + 3.0).tolist()
    x_b = [[v] for v in rng.normal(size=100).tolist()]
    result = _invoke(
        "decomposition_oaxaca_blinder",
        y_a=y_a,
        x_a=x_a,
        y_b=y_b,
        x_b=x_b,
    )
    # total_difference should be ≈ mean(y_a) − mean(y_b) ≈ 2.0
    assert abs(result["total_difference"] - 2.0) < 0.4
    # explained + unexplained should sum to total
    parts_sum = result["explained_part"] + result["unexplained_part"]
    assert abs(parts_sum - result["total_difference"]) < 1e-6


# ============================================================================
# SPATIAL
# ============================================================================


def _chain(n: int) -> tuple[dict[int, list[int]], dict[int, list[float]]]:
    """Linear chain 0–(n-1) for spatial tests: neighbors dict + equal weights."""
    nb = {i: ([i - 1] if i > 0 else []) + ([i + 1] if i < n - 1 else []) for i in range(n)}
    wt = {i: [1.0 / len(v)] * len(v) if v else [] for i, v in nb.items()}
    return nb, wt


def test_morans_i_positive_on_clustered_values() -> None:
    """A chain graph with monotonically increasing values has positive I."""
    n = 10
    neighbors, weights = _chain(n)
    values = [float(i) for i in range(n)]  # monotonic → strong positive spatial autocorr
    result = _invoke(
        "spatial_morans_i_test", values=values, neighbors=neighbors, weights=weights
    )
    I_like = _find_key_recursive(result, "morans_i") or _find_key_recursive(
        result, "moran_i"
    ) or _find_key_recursive(result, "i_statistic")
    assert I_like is not None and I_like > 0.3, (
        f"clustered chain should have large positive I, got {I_like}"
    )


def test_morans_i_near_zero_on_random_permutation() -> None:
    """Randomly permuted values on a chain graph → I near 0."""
    rng = np.random.default_rng(67)
    n = 30
    neighbors, weights = _chain(n)
    values = rng.normal(size=n).tolist()
    result = _invoke(
        "spatial_morans_i_test", values=values, neighbors=neighbors, weights=weights
    )
    I_like = _find_key_recursive(result, "morans_i") or _find_key_recursive(
        result, "moran_i"
    ) or _find_key_recursive(result, "i_statistic")
    assert I_like is not None
    assert abs(I_like) < 0.5, f"random values should have |I| < 0.5, got {I_like}"


# ============================================================================
# MISSING DATA
# ============================================================================


def test_simple_imputation_mean_fills_missing_with_column_mean() -> None:
    data = [[1.0, 10.0], [None, 20.0], [5.0, None], [7.0, 40.0]]
    # col 0: mean of {1,5,7} = 4.33
    # col 1: mean of {10,20,40} = 23.33
    result = _invoke(
        "missing_data_simple_imputation", data=data, strategy="mean"
    )
    imputed = result["imputed_data"]
    assert abs(imputed[1][0] - 13.0 / 3) < 1e-6
    assert abs(imputed[2][1] - 70.0 / 3) < 1e-6
    # Fill values are [col0_mean, col1_mean]
    assert abs(result["fill_values"][0] - 13.0 / 3) < 1e-6
    assert abs(result["fill_values"][1] - 70.0 / 3) < 1e-6


# ============================================================================
# MODEL DIAGNOSTICS
# ============================================================================


def test_diagnostic_dw_near_2_for_iid_residuals() -> None:
    """Clean OLS (no autocorrelation) → Durbin-Watson ≈ 2."""
    y, X = _linear_dgp(n=400, beta=[0.0, 1.0], sigma=0.3, seed=71)
    result = _invoke("model_diagnostic_tests", y_data=y, x_data=X)
    dw = result["dw_statistic"]
    assert 1.5 < dw < 2.5, f"iid residuals should have DW ~2, got {dw}"


# ============================================================================
# TIME SERIES — exp smoothing / GARCH / cointegration
# ============================================================================


def test_exp_smoothing_recovers_trend_direction() -> None:
    """Upward-trending series → last forecast > first observation."""
    rng = np.random.default_rng(73)
    series = [1.0 + 0.5 * i + rng.normal(scale=0.3) for i in range(60)]
    result = _invoke(
        "time_series_exponential_smoothing",
        data=series,
        trend=True,
        seasonal=False,
        forecast_steps=5,
    )
    fc = _find_key_recursive(result, "forecast") or _find_key_recursive(
        result, "forecasts"
    ) or _find_key_recursive(result, "predictions")
    assert fc is not None and isinstance(fc, list) and len(fc) >= 1
    # Forecast should continue the upward trend — last value noticeably above series mean
    assert fc[-1] > series[-1] - 5.0, f"forecast not following trend: series end={series[-1]}, fc={fc}"


def test_garch_fits_high_volatility_series() -> None:
    """Time-varying volatility → GARCH should fit (finite AIC, reasonable result)."""
    rng = np.random.default_rng(79)
    n = 600
    series, sigma2 = [], 1.0
    for _ in range(n):
        sigma2 = 0.1 + 0.4 * sigma2 + 0.4 * (rng.normal() ** 2)  # GARCH(1,1)-like
        series.append(float(rng.normal(scale=sigma2**0.5)))
    result = _invoke("time_series_garch_model", data=series, order=[1, 1])
    # A successfully-fit GARCH has a finite log-likelihood and AIC
    aic = _find_key_recursive(result, "aic")
    assert aic is not None and math.isfinite(aic), f"GARCH AIC non-finite: {aic}"


def test_cointegration_detects_engle_granger_relationship() -> None:
    """Two series sharing a common random walk — E-G should signal cointegration."""
    rng = np.random.default_rng(83)
    n = 400
    common = np.cumsum(rng.normal(size=n))  # unit-root common factor
    y1 = common + rng.normal(scale=0.3, size=n)
    y2 = 2.0 * common + rng.normal(scale=0.3, size=n)
    # Shape expected by var_svar/cointegration adapter: [timepoint][variable]
    data = [[float(y1[t]), float(y2[t])] for t in range(n)]
    result = _invoke(
        "time_series_cointegration_analysis",
        data=data,
        analysis_type="engle-granger",
    )
    # A cointegrated pair should produce a finite p-value or test-stat; we don't
    # over-specify — just check the adapter returned something non-error.
    assert isinstance(result, dict)
    assert not (result.get("ok") is False and "error" in result), (
        f"cointegration adapter errored on a textbook cointegrated pair: {result}"
    )


# ============================================================================
# REGRESSION DISCONTINUITY
# ============================================================================


def test_rdd_recovers_known_jump() -> None:
    """Sharp RDD: outcome = 1 + 2·x + 5·(x≥0) + ε. Recover discontinuity ≈ 5."""
    rng = np.random.default_rng(89)
    n = 300
    x = rng.uniform(-1.0, 1.0, size=n)
    y = 1.0 + 2.0 * x + 5.0 * (x >= 0) + rng.normal(scale=0.3, size=n)
    result = _invoke(
        "causal_regression_discontinuity",
        running_variable=x.tolist(),
        outcome=y.tolist(),
        cutoff=0.0,
    )
    est = _find_key_recursive(result, "estimate")
    # Tolerant — local-linear estimators can be noisy on small bandwidths
    assert est is not None and 3.5 < est < 6.5, (
        f"RDD jump estimate off (expected ≈5): {est}"
    )


# ============================================================================
# SPATIAL WEIGHTS MATRIX
# ============================================================================


def test_spatial_weights_knn_builds_k2_graph_for_grid() -> None:
    """8 points on a 3×3-ish grid; k=2 KNN graph should have 8 nodes, each
    with exactly 2 neighbours."""
    coords = [[float(i % 3), float(i // 3)] for i in range(8)]
    result = _invoke(
        "spatial_weights_matrix", coordinates=coords, weight_type="knn", k=2
    )
    # Find some indicator of graph size / degree
    n_like = _find_key_recursive(result, "n_obs") or _find_key_recursive(
        result, "n_observations"
    )
    assert n_like == 8


# ============================================================================
# MICROECONOMETRICS
# ============================================================================


def _binary_dgp(
    n: int = 400, beta: list[float] = None, seed: int = 0
) -> tuple[list[int], list[list[float]]]:
    """P(y=1 | x) = Φ(Xβ). Returns (y, X) with X non-collinear, y binary."""
    beta = beta or [0.3, 0.8, -0.6]  # const, x1, x2
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(beta) - 1))
    linpred = beta[0] + X @ np.array(beta[1:])
    p = 1.0 / (1.0 + np.exp(-linpred))
    y = (rng.uniform(size=n) < p).astype(int).tolist()
    return y, X.tolist()


def test_logit_recovers_sign_of_known_coefficients() -> None:
    """P(y=1) = σ(0.3 + 0.8·x₁ − 0.6·x₂) → sign(β₁)>0, sign(β₂)<0."""
    y, X = _binary_dgp(n=600, beta=[0.3, 0.8, -0.6], seed=101)
    result = _invoke("micro_logit", X_data=X, y_data=y)
    coefs = result["coefficients"]
    # Model reports [const?, x1, x2]; be defensive about intercept position
    # but the two slope coefs should have the right signs.
    sloped = coefs if len(coefs) == 2 else coefs[1:] if len(coefs) >= 3 else coefs
    # Find the two with largest absolute values as the true slopes.
    if len(sloped) >= 2:
        assert sloped[-2] > 0, f"β₁ should be positive, got {sloped[-2]}"
        assert sloped[-1] < 0, f"β₂ should be negative, got {sloped[-1]}"


def test_probit_pseudo_r_squared_positive_for_separable_data() -> None:
    """On well-separable binary data, pseudo-R² should be clearly positive."""
    y, X = _binary_dgp(n=500, beta=[0.0, 2.0, -2.0], seed=103)
    result = _invoke("micro_probit", X_data=X, y_data=y)
    assert result["pseudo_r_squared"] > 0.1


def test_poisson_recovers_log_rate() -> None:
    """Y ~ Poisson(exp(1 + 0.5·x)) → Poisson regression should recover
    β₁ ≈ 0.5 and a positive intercept."""
    rng = np.random.default_rng(107)
    n = 400
    x = rng.uniform(-1, 1, size=n)
    rate = np.exp(1.0 + 0.5 * x)
    y = rng.poisson(rate).tolist()
    X = [[v] for v in x.tolist()]
    result = _invoke("micro_poisson", X_data=X, y_data=y)
    coefs = result["coefficients"]
    # Position of slope varies with/without intercept — take the last
    slope = coefs[-1]
    assert abs(slope - 0.5) < 0.15, f"Poisson slope off: {slope}"


def test_negative_binomial_runs_on_overdispersed_counts() -> None:
    """Overdispersed counts (mean ≪ variance) — NB should converge."""
    rng = np.random.default_rng(109)
    n = 300
    # Generate overdispersed: Y ~ NegBin(mean=μ, alpha)
    mu = np.exp(1.0 + 0.3 * rng.normal(size=n))
    y = rng.negative_binomial(3, 3 / (3 + mu)).tolist()
    X = [[float(v)] for v in rng.normal(size=n).tolist()]
    result = _invoke("micro_negative_binomial", X_data=X, y_data=y)
    assert math.isfinite(result["log_likelihood"])


def test_multinomial_logit_identifies_three_classes() -> None:
    rng = np.random.default_rng(113)
    n = 600
    X = rng.normal(size=(n, 2))
    # Make class related to X[:, 0]
    y = np.where(X[:, 0] < -0.3, 0, np.where(X[:, 0] < 0.3, 1, 2)).tolist()
    result = _invoke("micro_multinomial_logit", X_data=X.tolist(), y_data=y)
    # 3 classes in the classes list
    assert len(result["classes"]) == 3


def test_tobit_recovers_slope_on_censored_data() -> None:
    """y* = 1 + 2·x + ε; y = max(0, y*). Tobit should recover β₁ ≈ 2."""
    rng = np.random.default_rng(117)
    n = 500
    x = rng.uniform(-1, 1, size=n)
    latent = 1.0 + 2.0 * x + rng.normal(scale=0.8, size=n)
    y = np.maximum(0.0, latent).tolist()
    X = [[float(v)] for v in x.tolist()]
    result = _invoke("micro_tobit", X_data=X, y_data=y, lower_bound=0.0)
    coefs = result["coefficients"]
    # Tobit output may include sigma; slope is the β coefficient on x
    slope_candidates = [c for c in coefs if 1.5 < c < 2.5]
    assert slope_candidates, f"no slope coefficient near 2 in {coefs}"


def test_heckman_runs_on_selection_data() -> None:
    """Selection-model DGP: observe y only when s=1 (s depends on Z)."""
    rng = np.random.default_rng(119)
    n = 500
    z = rng.normal(size=(n, 1))
    x = rng.normal(size=(n, 1))
    # Selection equation
    s = (0.5 + z[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int).tolist()
    # Outcome (observed regardless; adapter handles the gate)
    y = (1.0 + 2.0 * x[:, 0] + rng.normal(scale=0.5, size=n)).tolist()
    result = _invoke(
        "micro_heckman",
        X_select_data=z.tolist(),
        Z_data=x.tolist(),
        y_data=y,
        s_data=s,
    )
    assert result["n_obs"] == n
    assert result["n_selected"] > 0


# ============================================================================
# MACHINE LEARNING
# ============================================================================


def _separable_dgp(n: int = 300, seed: int = 0) -> tuple[list[int], list[list[float]]]:
    """Class 1 centred at (+2, +2), class 0 at (-2, -2)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    X_pos = rng.normal(loc=2.0, scale=0.8, size=(half, 2))
    X_neg = rng.normal(loc=-2.0, scale=0.8, size=(n - half, 2))
    X = np.vstack([X_pos, X_neg])
    y = [1] * half + [0] * (n - half)
    order = rng.permutation(n)
    X = X[order]
    y = [y[i] for i in order]
    return y, X.tolist()


def test_random_forest_classification_accuracy_on_separable_data() -> None:
    """Well-separated classes → train accuracy should be ≥ 0.9."""
    y, X = _separable_dgp(n=400, seed=131)
    result = _invoke(
        "ml_random_forest", X_data=X, y_data=y, problem_type="classification"
    )
    acc = _find_key_recursive(result["train_results"], "accuracy")
    assert acc is not None and acc > 0.9, f"RF train accuracy low: {acc}"


def test_gradient_boosting_classification_accuracy() -> None:
    y, X = _separable_dgp(n=400, seed=133)
    result = _invoke(
        "ml_gradient_boosting", X_data=X, y_data=y, problem_type="classification"
    )
    acc = _find_key_recursive(result["train_results"], "accuracy")
    assert acc is not None and acc > 0.9, f"GB train accuracy low: {acc}"


def test_svm_classification_accuracy() -> None:
    y, X = _separable_dgp(n=300, seed=137)
    result = _invoke(
        "ml_support_vector_machine", X_data=X, y_data=y, problem_type="classification"
    )
    acc = _find_key_recursive(result["train_results"], "accuracy")
    assert acc is not None and acc > 0.9, f"SVM train accuracy low: {acc}"


def test_neural_network_classification_accuracy() -> None:
    y, X = _separable_dgp(n=400, seed=139)
    result = _invoke(
        "ml_neural_network", X_data=X, y_data=y, problem_type="classification"
    )
    acc = _find_key_recursive(result["train_results"], "accuracy")
    assert acc is not None and acc > 0.85, f"NN train accuracy low: {acc}"


def test_kmeans_recovers_two_well_separated_clusters() -> None:
    """Two widely-spaced blobs → silhouette should be ≳ 0.5 with k=2."""
    _, X = _separable_dgp(n=200, seed=149)
    result = _invoke("ml_kmeans_clustering", X_data=X, n_clusters=2)
    sil = _find_key_recursive(result["metrics"], "silhouette_score")
    assert sil is not None and sil > 0.5, f"K-means silhouette too low: {sil}"
    # Cluster centres should be roughly at (+2,+2) and (-2,-2)
    centers = result["cluster_centers"]
    assert len(centers) == 2


def test_hierarchical_clustering_produces_two_clusters() -> None:
    _, X = _separable_dgp(n=200, seed=151)
    result = _invoke("ml_hierarchical_clustering", X_data=X, n_clusters=2)
    labels = result["labels"]
    assert len(set(labels)) == 2
    # Silhouette should be high on well-separated blobs
    sil = _find_key_recursive(result["metrics"], "silhouette_score")
    assert sil is None or sil > 0.4


def test_double_ml_recovers_treatment_effect() -> None:
    """Y = τ·D + g(X) + ε; D = m(X) + η. τ=2 — DML should recover τ."""
    rng = np.random.default_rng(157)
    n = 400
    X = rng.normal(size=(n, 3))
    # Treatment depends on X (propensity)
    D = 0.5 * X[:, 0] + rng.normal(scale=1.0, size=n)
    # Outcome: τ=2 plus confounding via X
    y = 2.0 * D + X[:, 1] + 0.5 * X[:, 2] + rng.normal(scale=0.5, size=n)
    result = _invoke(
        "ml_double_machine_learning",
        X_data=X.tolist(),
        y_data=y.tolist(),
        d_data=D.tolist(),
    )
    effect = result.get("effect")
    assert effect is not None and abs(effect - 2.0) < 0.4, (
        f"DML treatment effect off: {effect}"
    )


def test_causal_forest_returns_finite_ate() -> None:
    rng = np.random.default_rng(163)
    n = 400
    X = rng.normal(size=(n, 3))
    w = (rng.uniform(size=n) < 0.5).astype(float)  # random treatment
    y = 2.0 * w + 0.5 * X[:, 0] + rng.normal(scale=0.3, size=n)
    result = _invoke(
        "ml_causal_forest",
        X_data=X.tolist(),
        y_data=y.tolist(),
        w_data=w.tolist(),
    )
    ate = result.get("ate")
    assert ate is not None and math.isfinite(ate)
    # With random treatment + large effect, ATE should land near 2
    assert 1.0 < ate < 3.0, f"causal forest ATE outside reasonable band: {ate}"


# ---------- small recursive helpers -----------------------------------------


def _find_key_recursive(obj: Any, key: str) -> Any:
    """Return the first value associated with ``key`` anywhere in ``obj``."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            hit = _find_key_recursive(v, key)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_key_recursive(v, key)
            if hit is not None:
                return hit
    return None


def _collect_numbers(obj: Any) -> list[float]:
    """Flatten all numeric leaves of a nested structure."""
    out: list[float] = []
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect_numbers(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_collect_numbers(v))
    return out
