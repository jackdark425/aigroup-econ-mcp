"""
模型选择 (Model Selection) 模块实现

包括：
- 信息准则（AIC/BIC/HQIC）
- 交叉验证（K折、留一法）
- 格兰杰因果检验
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, Field
from statsmodels.tsa.stattools import grangercausalitytests


class GrangerCausalityResult(BaseModel):
    """格兰杰因果检验结果"""

    f_statistic: float = Field(..., description="F统计量")
    p_value: float = Field(..., description="p值")
    lag_order: int = Field(..., description="滞后阶数")
    n_obs: int = Field(..., description="观测数量")
    dependent_variable: str = Field(..., description="因变量")
    independent_variable: str = Field(..., description="格兰杰原因变量")


class ModelSelectionResult(BaseModel):
    """模型选择结果"""

    aic: float = Field(..., description="赤池信息准则 (AIC)")
    bic: float = Field(..., description="贝叶斯信息准则 (BIC)")
    hqic: float = Field(..., description="汉南-奎因信息准则 (HQIC)")
    r_squared: float = Field(..., description="R方")
    adj_r_squared: float = Field(..., description="调整R方")
    log_likelihood: float = Field(..., description="对数似然值")
    n_obs: int = Field(..., description="观测数量")
    n_params: int = Field(..., description="参数数量")
    cv_score: float | None = Field(None, description="交叉验证得分")
    fit_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues; if non-empty, cv_score may be None or degraded because one or more CV folds failed numerically",
    )


def granger_causality_test(
    x_data: list[float], y_data: list[float], max_lag: int = 1, add_constant: bool = True
) -> GrangerCausalityResult:
    """
    格兰杰因果检验

    Args:
        x_data: 可能的格兰杰原因变量
        y_data: 因变量
        max_lag: 最大滞后阶数
        add_constant: 是否添加常数项

    Returns:
        GrangerCausalityResult: 格兰杰因果检验结果
    """
    # 转换为numpy数组
    x = np.asarray(x_data, dtype=np.float64)
    y = np.asarray(y_data, dtype=np.float64)

    # 检查数据长度
    if len(x) != len(y):
        raise ValueError("x_data和y_data的长度必须相同")

    if len(x) <= max_lag:
        raise ValueError("数据长度必须大于滞后阶数")

    # 构建数据框用于statsmodels
    data = pd.DataFrame({"y": y, "x": x})

    # 执行格兰杰因果检验
    try:
        # grangercausalitytests返回一个字典，键为滞后阶数
        test_result = grangercausalitytests(data, max_lag, addconst=add_constant, verbose=False)

        # 获取指定滞后阶数的结果（使用最大滞后阶数）
        lag_order = max_lag
        test_stats = test_result[lag_order][0]

        # 提取F统计量和p值（使用ssr F-test）
        f_statistic = test_stats["F test"]
        f_stat = f_statistic[0]  # F统计量
        p_value = f_statistic[1]  # p值

    except Exception:
        # 如果检验失败，返回默认值
        f_stat = 0.0
        p_value = 1.0
        lag_order = max_lag

    return GrangerCausalityResult(
        f_statistic=float(f_stat),
        p_value=float(p_value),
        lag_order=lag_order,
        n_obs=len(y) - lag_order,  # 考虑滞后后的实际观测数
        dependent_variable="y",
        independent_variable="x",
    )


def model_selection_criteria(
    y_data: list[float],
    x_data: list[list[float]],
    feature_names: list[str] | None = None,
    constant: bool = True,
    cv_folds: int | None = None,
) -> ModelSelectionResult:
    """
    计算模型选择信息准则

    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        feature_names: 特征名称
        constant: 是否包含常数项
        cv_folds: 交叉验证折数 (None表示不进行交叉验证，-1表示留一法)

    Returns:
        ModelSelectionResult: 模型选择结果
    """
    # 转换为numpy数组
    y = np.array(y_data)
    X = np.array(x_data)

    # 添加常数项
    if constant:
        X = sm.add_constant(X)
        if feature_names:
            feature_names = ["const"] + feature_names
        else:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    else:
        if not feature_names:
            feature_names = [f"x{i}" for i in range(X.shape[1])]

    # 执行OLS回归
    try:
        model = sm.OLS(y, X)
        results = model.fit()
    except Exception as e:
        raise ValueError(f"无法拟合模型: {str(e)}") from e

    # 提取统计量
    n = int(results.nobs)
    k = len(results.params)
    r_squared = float(results.rsquared)
    adj_r_squared = float(results.rsquared_adj)
    log_likelihood = float(results.llf)
    aic = float(results.aic)
    bic = float(results.bic)

    # 计算HQIC (statsmodels中没有直接提供HQIC)
    if n > 1 and np.log(n) != 0:
        hqic = -2 * log_likelihood + 2 * k * np.log(np.log(n))
    else:
        hqic = np.inf

    # 交叉验证
    cv_score: float | None = None
    fit_warnings: list[str] = []
    if cv_folds is not None:
        cv_score, cv_warnings = _cross_validation(y, X, cv_folds)
        fit_warnings.extend(cv_warnings)

    return ModelSelectionResult(
        aic=aic,
        bic=bic,
        hqic=float(hqic) if np.isfinite(hqic) else np.inf,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        log_likelihood=log_likelihood,
        n_obs=n,
        n_params=k,
        cv_score=float(cv_score) if cv_score is not None else None,
        fit_warnings=fit_warnings,
    )


def _cross_validation(
    y: np.ndarray, X: np.ndarray, folds: int | None
) -> tuple[float | None, list[str]]:
    """
    执行交叉验证

    Args:
        y: 因变量
        X: 自变量矩阵
        folds: 折数 (-1表示留一法，其他正数表示K折交叉验证)

    Returns:
        (cv_score, warnings) — cv_score is mean MSE when computable, else None;
        warnings names each bail-out so the caller can surface them rather than
        silently substituting zero / None.
    """
    warnings: list[str] = []
    n = len(y)

    if folds is None or folds == 0:
        warnings.append(f"cv_folds={folds!r}; cross-validation skipped")
        return None, warnings

    if folds == -1 or folds >= n:
        folds = n  # leave-one-out

    if folds <= 1:
        warnings.append(f"cv_folds resolved to {folds}; need >=2 folds, skipping")
        return None, warnings
    if X.shape[0] != n:
        warnings.append(f"y length ({n}) != X rows ({X.shape[0]}); CV aborted")
        return None, warnings

    if X.shape[0] < X.shape[1]:
        warnings.append(
            f"underdetermined system (n_obs={X.shape[0]} < n_features={X.shape[1]}); "
            "CV aborted to avoid singular training folds"
        )
        return None, warnings

    rng = np.random.default_rng(42)
    indices = rng.permutation(n)

    fold_sizes = np.full(folds, n // folds)
    fold_sizes[: n % folds] += 1

    current = 0
    mse_scores: list[float] = []

    for fold_index, fold_size in enumerate(fold_sizes):
        start, stop = current, current + fold_size
        current = stop
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            if X_train.shape[0] < X_train.shape[1] or X_train.shape[0] == 0 or X_test.shape[0] == 0:
                warnings.append(
                    f"fold {fold_index}: degenerate train/test split "
                    f"(train={X_train.shape}, test={X_test.shape}), skipped"
                )
                continue

            try:
                beta_train = sm.OLS(y_train, X_train).fit().params
            except Exception:
                XtX = X_train.T @ X_train
                if XtX.shape[0] == 0:
                    warnings.append(
                        f"fold {fold_index}: empty XtX matrix during regularized fit, skipped"
                    )
                    continue
                # Tiny ridge to ward off singularity; fall back to pinv if it still trips.
                reg_param = (
                    1e-10 * np.trace(XtX) / XtX.shape[0]
                    if np.trace(XtX) > 0
                    else 1e-10
                )
                XtX_reg = XtX + reg_param * np.eye(XtX.shape[0])
                try:
                    beta_train = np.linalg.solve(XtX_reg, X_train.T @ y_train)
                except np.linalg.LinAlgError:
                    beta_train = np.linalg.pinv(XtX_reg) @ X_train.T @ y_train

            try:
                y_pred = X_test @ beta_train
            except Exception as exc:
                warnings.append(f"fold {fold_index}: prediction failed ({exc}), skipped")
                continue

            if not np.all(np.isfinite(y_pred)):
                warnings.append(f"fold {fold_index}: non-finite predictions, skipped")
                continue

            mse = np.mean((y_test - y_pred) ** 2)
            if np.isfinite(mse):
                mse_scores.append(float(mse))
            else:
                warnings.append(f"fold {fold_index}: non-finite MSE, skipped")
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError) as exc:
            warnings.append(f"fold {fold_index}: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 — last-resort guard, still recorded
            warnings.append(f"fold {fold_index}: unexpected {type(exc).__name__}: {exc}")

    if not mse_scores:
        warnings.append("no CV fold produced a finite MSE; cv_score is None")
        return None, warnings
    return float(np.mean(mse_scores)), warnings
