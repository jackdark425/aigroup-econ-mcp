"""
动态面板模型实现（差分GMM、系统GMM）
"""

from pydantic import BaseModel, Field


class DynamicPanelResult(BaseModel):
    """动态面板模型结果"""

    model_type: str = Field(..., description="模型类型")
    coefficients: list[float] = Field(..., description="回归系数")
    std_errors: list[float] | None = Field(None, description="系数标准误")
    t_values: list[float] | None = Field(None, description="t统计量")
    p_values: list[float] | None = Field(None, description="p值")
    conf_int_lower: list[float] | None = Field(None, description="置信区间下界")
    conf_int_upper: list[float] | None = Field(None, description="置信区间上界")
    instruments: int | None = Field(None, description="工具变量数量")
    j_statistic: float | None = Field(None, description="过度识别约束检验统计量")
    j_p_value: float | None = Field(None, description="过度识别约束检验p值")
    n_obs: int = Field(..., description="观测数量")
    n_individuals: int = Field(..., description="个体数量")
    n_time_periods: int = Field(..., description="时间期数")
    fit_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal fit issues; e.g. the GMM estimator may have fallen back to plain OLS, in which case model_type is also annotated",
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _validate_and_build_panel(
    y_data: list[float],
    x_data: list[list[float]],
    entity_ids: list[int],
    time_periods: list[int],
):
    """Validate inputs and build panel DataFrames; return (y_df, x_df).

    Raises ValueError with Chinese messages on any validation failure,
    exactly matching the original per-function validation logic.
    """
    import pandas as pd

    if not y_data:
        raise ValueError("因变量数据不能为空")

    if not x_data:
        raise ValueError("自变量数据不能为空")

    if not all(isinstance(series, (list, tuple)) for series in x_data):
        raise ValueError("自变量数据必须是二维列表格式，每个子列表代表一个自变量的完整时间序列")

    if not entity_ids:
        raise ValueError("个体标识符不能为空")

    if not time_periods:
        raise ValueError("时间标识符不能为空")

    # 检查数据长度一致性
    lengths = [len(y_data), len(entity_ids), len(time_periods)]
    for x_series in x_data:
        lengths.append(len(x_series))

    if len(set(lengths)) > 1:
        error_msg = "所有数据序列的长度必须一致，当前长度分别为:\n"
        error_msg += f"- 因变量: {len(y_data)} 个观测\n"
        error_msg += f"- 个体标识符: {len(entity_ids)} 个观测\n"
        error_msg += f"- 时间标识符: {len(time_periods)} 个观测\n"
        for i, x_series in enumerate(x_data):
            error_msg += f"- 自变量{i + 1}: {len(x_series)} 个观测\n"
        error_msg += "\n请确保所有数据的观测数量相同"
        raise ValueError(error_msg)

    # 创建面板数据结构
    index = pd.MultiIndex.from_arrays([entity_ids, time_periods], names=["entity", "time"])

    if index.has_duplicates:
        raise ValueError("存在重复的个体-时间索引")

    y_df = pd.DataFrame({"y": y_data}, index=index)

    x_dict = {f"x{i}": x for i, x in enumerate(x_data)}
    x_df = pd.DataFrame(x_dict, index=index)

    if y_df.empty or x_df.empty:
        raise ValueError("构建的面板数据为空")

    return y_df, x_df


def _x_data_to_array(x_data: list[list[float]]):
    """Convert x_data (list of series) to a 2-D numpy array (obs × vars)."""
    import numpy as np

    if isinstance(x_data[0], (list, tuple)):
        x_array = np.array(x_data)
        if x_array.shape[0] == 1 and x_array.shape[1] > 1:
            x_array = x_array.T
        elif x_array.ndim == 1:
            x_array = x_array.reshape(-1, 1)
    else:
        x_array = np.array(x_data).reshape(-1, 1)

    if x_array.ndim == 1:
        x_array = x_array.reshape(-1, 1)

    return x_array


def _extract_linearmodels_result(fitted_model):
    """Pull coefficients, SE, t, p, CI, instruments, and J-stat from a
    linearmodels fitted result object (DifferenceGMM or SystemGMM result).
    Returns a dict with keys matching DynamicPanelResult fields.
    """
    params = fitted_model.params.tolist()

    std_errors = (
        fitted_model.std_errors.tolist() if fitted_model.std_errors is not None else None
    )
    t_values = fitted_model.tstats.tolist() if fitted_model.tstats is not None else None
    p_values = fitted_model.pvalues.tolist() if fitted_model.pvalues is not None else None

    conf_int_lower = None
    conf_int_upper = None
    if fitted_model.conf_int() is not None:
        conf_int = fitted_model.conf_int()
        conf_int_lower = conf_int.iloc[:, 0].tolist()
        conf_int_upper = conf_int.iloc[:, 1].tolist()

    instruments = None
    try:
        if hasattr(fitted_model, "summary") and len(fitted_model.summary.tables) > 0:
            instruments = int(fitted_model.summary.tables[0].data[6][1])
    except (IndexError, ValueError, TypeError):
        instruments = None

    j_statistic = (
        float(fitted_model.j_stat.stat)
        if hasattr(fitted_model, "j_stat") and hasattr(fitted_model.j_stat, "stat")
        else None
    )
    j_p_value = (
        float(fitted_model.j_stat.pval)
        if hasattr(fitted_model, "j_stat") and hasattr(fitted_model.j_stat, "pval")
        else None
    )

    return {
        "params": params,
        "std_errors": std_errors,
        "t_values": t_values,
        "p_values": p_values,
        "conf_int_lower": conf_int_lower,
        "conf_int_upper": conf_int_upper,
        "instruments": instruments,
        "j_statistic": j_statistic,
        "j_p_value": j_p_value,
    }


def _iv_estimate(X, y, Z):
    """2SLS / IV estimation. Returns a dict with regression results.

    Raises np.linalg.LinAlgError or ValueError on numerical failure.
    """
    import numpy as np
    from scipy.stats import chi2
    from scipy.stats import t as t_dist

    Z_proj = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
    X_hat = Z_proj @ X

    params_iv = np.linalg.lstsq(X_hat, y, rcond=None)[0]
    residuals = y - X @ params_iv

    n_params = len(params_iv)
    sigma2 = np.var(residuals)
    XtX_inv = np.linalg.inv(X_hat.T @ X_hat)
    cov_matrix = sigma2 * XtX_inv
    std_errors = np.sqrt(np.diag(cov_matrix)).tolist()

    params = params_iv.tolist()
    t_values = (params_iv / std_errors).tolist()
    p_values = [
        2 * (1 - t_dist.cdf(abs(tv), len(y) - n_params)) for tv in t_values
    ]
    t_critical = t_dist.ppf(0.975, len(y) - n_params)
    conf_int_lower = [p - t_critical * se for p, se in zip(params, std_errors, strict=False)]
    conf_int_upper = [p + t_critical * se for p, se in zip(params, std_errors, strict=False)]

    instruments = Z.shape[1] if Z.ndim > 1 else 1
    if instruments > n_params:
        j_statistic = float(np.sum(residuals**2) / sigma2)
        j_p_value = float(1 - chi2.cdf(j_statistic, instruments - n_params))
    else:
        j_statistic = 0.0
        j_p_value = 1.0

    return {
        "params": params,
        "std_errors": std_errors,
        "t_values": t_values,
        "p_values": p_values,
        "conf_int_lower": conf_int_lower,
        "conf_int_upper": conf_int_upper,
        "instruments": instruments,
        "j_statistic": j_statistic,
        "j_p_value": j_p_value,
    }


def _ols_fallback(X, y, n_vars: int):
    """OLS on the supplied design matrix X and response y.

    Returns a dict with the same keys as _iv_estimate.
    instruments is set to n_vars + 1 (constant + regressors), matching
    the original fallback behaviour in both functions.
    """
    import numpy as np
    from scipy.stats import t as t_dist

    params_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ params_ols

    n_params = len(params_ols)
    sigma2 = np.var(residuals)
    XtX_inv = np.linalg.inv(X.T @ X)
    std_errors = np.sqrt(np.diag(sigma2 * XtX_inv)).tolist()

    params = params_ols.tolist()
    t_values = (params_ols / std_errors).tolist()
    p_values = [
        2 * (1 - t_dist.cdf(abs(tv), len(y) - n_params)) for tv in t_values
    ]
    t_critical = t_dist.ppf(0.975, len(y) - n_params)
    conf_int_lower = [p - t_critical * se for p, se in zip(params, std_errors, strict=False)]
    conf_int_upper = [p + t_critical * se for p, se in zip(params, std_errors, strict=False)]

    return {
        "params": params,
        "std_errors": std_errors,
        "t_values": t_values,
        "p_values": p_values,
        "conf_int_lower": conf_int_lower,
        "conf_int_upper": conf_int_upper,
        "instruments": n_vars + 1,
        "j_statistic": 0.0,
        "j_p_value": 1.0,
    }


def _pad_and_stack(z_list):
    """Pad a list of 1-D numpy arrays to equal length and return a 2-D array."""
    import numpy as np

    max_len = max(len(z) for z in z_list)
    padded = [np.pad(z, (0, max_len - len(z)), "constant") if len(z) < max_len else z
              for z in z_list]
    return np.array(padded)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diff_gmm_model(
    y_data: list[float],
    x_data: list[list[float]],
    entity_ids: list[int],
    time_periods: list[int],
    lags: int = 1,
) -> DynamicPanelResult:
    """
    差分GMM模型实现（Arellano-Bond估计）

    Args:
        y_data: 因变量数据
        x_data: 自变量数据 (格式: 每个子列表代表一个自变量的时间序列)
        entity_ids: 个体标识符
        time_periods: 时间标识符
        lags: 滞后期数

    Returns:
        DynamicPanelResult: 差分GMM模型结果
    """
    try:
        import numpy as np

        # 尝试不同的导入路径
        try:
            from linearmodels.panel import DifferenceGMM

            use_linearmodels = True
        except ImportError:
            try:
                from linearmodels import DifferenceGMM

                use_linearmodels = True
            except ImportError:
                use_linearmodels = False

        y_df, x_df = _validate_and_build_panel(y_data, x_data, entity_ids, time_periods)

        fallback_err: str | None = None

        if use_linearmodels:
            model = DifferenceGMM(y_df, x_df, lags=lags)
            res = _extract_linearmodels_result(model.fit())
        else:
            y_array = np.array(y_data)
            x_array = _x_data_to_array(x_data)
            n_obs = len(y_data)
            n_vars = x_array.shape[1]

            # 构建差分数据
            dy = np.diff(y_array)
            dx = np.diff(x_array, axis=0)

            # 构建工具变量矩阵（使用滞后水平作为工具变量）
            Z_list = []
            for t in range(2, n_obs):
                lag_y = y_array[: t - 1].flatten()
                lag_x = x_array[: t - 1, :].flatten()
                if len(lag_y) + len(lag_x) > 0:
                    Z_list.append(np.concatenate([lag_y, lag_x]))

            if Z_list:
                Z = _pad_and_stack(Z_list)
            else:
                Z = np.column_stack([y_array[:-1], x_array[:-1, :]])

            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)

            X_diff = np.column_stack([np.ones(len(dy)), dx])

            try:
                res = _iv_estimate(X_diff, dy, Z)
            except (np.linalg.LinAlgError, ValueError) as iv_err:
                fallback_err = str(iv_err)
                res = _ols_fallback(X_diff, dy, n_vars)

        _fallback_note: list[str] = []
        _model_type = "Difference GMM (Arellano-Bond)"
        if fallback_err is not None:
            _model_type = "Difference GMM (Arellano-Bond) — OLS fallback"
            _fallback_note.append(
                f"IV estimation numerically failed ({fallback_err}); results are from "
                "an OLS fallback on first-differenced data, NOT Arellano-Bond GMM"
            )

        return DynamicPanelResult(
            model_type=_model_type,
            coefficients=res["params"],
            std_errors=res["std_errors"],
            t_values=res["t_values"],
            p_values=res["p_values"],
            conf_int_lower=res["conf_int_lower"],
            conf_int_upper=res["conf_int_upper"],
            instruments=res["instruments"],
            j_statistic=res["j_statistic"],
            j_p_value=res["j_p_value"],
            n_obs=len(y_data),
            n_individuals=len(set(entity_ids)),
            n_time_periods=len(set(time_periods)),
            fit_warnings=_fallback_note,
        )
    except Exception as e:
        raise ValueError(f"差分GMM模型拟合失败: {str(e)}") from e


def sys_gmm_model(
    y_data: list[float],
    x_data: list[list[float]],
    entity_ids: list[int],
    time_periods: list[int],
    lags: int = 1,
) -> DynamicPanelResult:
    """
    系统GMM模型实现（Blundell-Bond估计）

    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        entity_ids: 个体标识符
        time_periods: 时间标识符
        lags: 滞后期数

    Returns:
        DynamicPanelResult: 系统GMM模型结果
    """
    try:
        import numpy as np

        # 尝试不同的导入路径
        try:
            from linearmodels.panel import SystemGMM

            use_linearmodels = True
        except ImportError:
            try:
                from linearmodels import SystemGMM

                use_linearmodels = True
            except ImportError:
                use_linearmodels = False

        y_df, x_df = _validate_and_build_panel(y_data, x_data, entity_ids, time_periods)

        fallback_err: str | None = None

        if use_linearmodels:
            model = SystemGMM(y_df, x_df, lags=lags)
            res = _extract_linearmodels_result(model.fit())
        else:
            y_array = np.array(y_data)
            x_array = _x_data_to_array(x_data)
            n_obs = len(y_data)
            n_vars = x_array.shape[1]

            # 构建差分数据（用于差分方程）
            dy = np.diff(y_array)
            dx = np.diff(x_array, axis=0)

            # 构建水平数据（用于水平方程）
            y_level = y_array[1:]
            x_level = x_array[1:, :]

            # 构建工具变量矩阵（系统GMM使用滞后差分作为水平方程的工具变量）
            Z_diff_list = []
            Z_level_list = []

            for t in range(2, n_obs):
                # 差分方程的工具变量：滞后水平
                lag_y_diff = y_array[: t - 1].flatten()
                lag_x_diff = x_array[: t - 1, :].flatten()
                if len(lag_y_diff) + len(lag_x_diff) > 0:
                    Z_diff_list.append(np.concatenate([lag_y_diff, lag_x_diff]))

                # 水平方程的工具变量：滞后差分
                if t > 2:
                    lag_dy = np.diff(y_array[:t]).flatten()
                    lag_dx = np.diff(x_array[:t, :], axis=0).flatten()
                    if len(lag_dy) + len(lag_dx) > 0:
                        Z_level_list.append(np.concatenate([lag_dy, lag_dx]))

            # 合并工具变量
            if Z_diff_list and Z_level_list:
                max_len_diff = max(len(z) for z in Z_diff_list)
                max_len_level = max(len(z) for z in Z_level_list)
                max_len = max(max_len_diff, max_len_level)

                Z_diff_padded = [
                    np.pad(z, (0, max_len - len(z)), "constant") if len(z) < max_len else z
                    for z in Z_diff_list
                ]
                Z_level_padded = [
                    np.pad(z, (0, max_len - len(z)), "constant") if len(z) < max_len else z
                    for z in Z_level_list
                ]

                min_len = min(len(Z_diff_padded), len(Z_level_padded))
                Z = np.column_stack([Z_diff_padded[:min_len], Z_level_padded[:min_len]])
            else:
                Z = np.column_stack([y_array[:-1], x_array[:-1, :]])

            # 构建系统方程的设计矩阵
            X_diff = np.column_stack([np.ones(len(dy)), dx])
            X_level = np.column_stack([np.ones(len(y_level)), x_level])
            X_sys = np.vstack([X_diff, X_level])
            y_sys = np.concatenate([dy, y_level])

            try:
                res = _iv_estimate(X_sys, y_sys, Z)
            except (np.linalg.LinAlgError, ValueError) as iv_err:
                fallback_err = str(iv_err)
                res = _ols_fallback(X_sys, y_sys, n_vars)

        _fallback_note: list[str] = []
        _model_type = "System GMM (Blundell-Bond)"
        if fallback_err is not None:
            _model_type = "System GMM (Blundell-Bond) — OLS fallback"
            _fallback_note.append(
                f"IV estimation numerically failed ({fallback_err}); results are from "
                "an OLS fallback on the stacked level+difference system, NOT Blundell-Bond GMM"
            )

        return DynamicPanelResult(
            model_type=_model_type,
            coefficients=res["params"],
            std_errors=res["std_errors"],
            t_values=res["t_values"],
            p_values=res["p_values"],
            conf_int_lower=res["conf_int_lower"],
            conf_int_upper=res["conf_int_upper"],
            instruments=res["instruments"],
            j_statistic=res["j_statistic"],
            j_p_value=res["j_p_value"],
            n_obs=len(y_data),
            n_individuals=len(set(entity_ids)),
            n_time_periods=len(set(time_periods)),
            fit_warnings=_fallback_note,
        )
    except Exception as e:
        raise ValueError(f"系统GMM模型拟合失败: {str(e)}") from e
