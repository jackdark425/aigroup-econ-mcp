"""
时间序列分析工具
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


class StationarityTest(BaseModel):
    """平稳性检验结果"""
    adf_statistic: float
    adf_pvalue: float
    adf_critical_values: Dict[str, float]
    kpss_statistic: float
    kpss_pvalue: float
    is_stationary: bool


class ACFPACFResult(BaseModel):
    """自相关分析结果"""
    acf_values: List[float]
    pacf_values: List[float]
    acf_confidence: List[Tuple[float, float]]
    pacf_confidence: List[Tuple[float, float]]


class ARIMAResult(BaseModel):
    """ARIMA模型结果"""
    order: Tuple[int, int, int]
    aic: float
    bic: float
    coefficients: Dict[str, float]
    fitted_values: List[float]
    residuals: List[float]
    forecast: Optional[List[float]] = None


def check_stationarity(data: List[float], max_lags: int = None) -> StationarityTest:
    """平稳性检验（ADF和KPSS）"""
    series = pd.Series(data)

    # ADF检验
    adf_result = adfuller(series, maxlag=max_lags, autolag='AIC')
    adf_stat, adf_pvalue = adf_result[0], adf_result[1]
    adf_critical = adf_result[4]

    # KPSS检验
    kpss_result = kpss(series, regression='c', nlags='auto')
    kpss_stat, kpss_pvalue = kpss_result[0], kpss_result[1]

    # 综合判断平稳性
    is_stationary = (adf_pvalue < 0.05) and (kpss_pvalue > 0.05)

    return StationarityTest(
        adf_statistic=adf_stat,
        adf_pvalue=adf_pvalue,
        adf_critical_values=adf_critical,
        kpss_statistic=kpss_stat,
        kpss_pvalue=kpss_pvalue,
        is_stationary=is_stationary
    )


def calculate_acf_pacf(
    data: List[float],
    nlags: int = 20,
    alpha: float = 0.05
) -> ACFPACFResult:
    """计算自相关和偏自相关函数"""
    series = pd.Series(data)

    # 计算ACF和PACF
    acf_values = acf(series, nlags=nlags, alpha=alpha)
    pacf_values = pacf(series, nlags=nlags, alpha=alpha)

    # 构建置信区间
    acf_conf = []
    pacf_conf = []

    for i in range(len(acf_values[1])):
        acf_conf.append((acf_values[1][i][0], acf_values[1][i][1]))
        pacf_conf.append((pacf_values[1][i][0], pacf_values[1][i][1]))

    return ACFPACFResult(
        acf_values=acf_values[0].tolist(),
        pacf_values=pacf_values[0].tolist(),
        acf_confidence=acf_conf,
        pacf_confidence=pacf_conf
    )


def fit_arima_model(
    data: List[float],
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> ARIMAResult:
    """拟合ARIMA模型"""
    series = pd.Series(data)

    try:
        if seasonal_order != (0, 0, 0, 0):
            # 季节性ARIMA
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
        else:
            # 普通ARIMA
            model = ARIMA(series, order=order)

        fitted_model = model.fit()

        return ARIMAResult(
            order=order,
            aic=fitted_model.aic,
            bic=fitted_model.bic,
            coefficients=fitted_model.params.to_dict(),
            fitted_values=fitted_model.fittedvalues.tolist(),
            residuals=fitted_model.resid.tolist()
        )

    except Exception as e:
        raise ValueError(f"ARIMA模型拟合失败: {str(e)}")


def find_best_arima_order(
    data: List[float],
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    seasonal: bool = False,
    max_P: int = 1,
    max_D: int = 1,
    max_Q: int = 1,
    m: int = 12
) -> Dict[str, Any]:
    """自动寻找最佳ARIMA模型阶数"""
    series = pd.Series(data)
    best_aic = float('inf')
    best_order = (0, 0, 0)
    best_seasonal_order = (0, 0, 0, 0)
    best_model = None

    # 非季节性ARIMA
    if not seasonal:
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            best_model = fitted_model
                    except:
                        continue

    # 季节性ARIMA
    else:
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    for P in range(max_P + 1):
                        for D in range(max_D + 1):
                            for Q in range(max_Q + 1):
                                try:
                                    seasonal_order = (P, D, Q, m)
                                    model = SARIMAX(series, order=(p, d, q), seasonal_order=seasonal_order)
                                    fitted_model = model.fit()
                                    if fitted_model.aic < best_aic:
                                        best_aic = fitted_model.aic
                                        best_order = (p, d, q)
                                        best_seasonal_order = seasonal_order
                                        best_model = fitted_model
                                except:
                                    continue

    if best_model is None:
        raise ValueError("无法找到合适的ARIMA模型")

    return {
        "best_order": best_order,
        "best_seasonal_order": best_seasonal_order if seasonal else None,
        "best_aic": best_aic,
        "best_bic": best_model.bic,
        "coefficients": best_model.params.to_dict(),
        "model_summary": str(best_model.summary())
    }


def decompose_time_series(
    data: List[float],
    model: str = "additive",
    period: Optional[int] = None
) -> Dict[str, List[float]]:
    """时间序列分解"""
    series = pd.Series(data)

    if period is None:
        # 自动检测周期（简单方法）
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(series, model=model, extrapolate_trend='freq')

        return {
            "trend": decomposition.trend.fillna(0).tolist(),
            "seasonal": decomposition.seasonal.fillna(0).tolist(),
            "residual": decomposition.resid.fillna(0).tolist(),
            "observed": decomposition.observed.tolist()
        }
    else:
        # 指定周期的分解
        decomposition = seasonal_decompose(series, model=model, period=period)

        return {
            "trend": decomposition.trend.fillna(0).tolist(),
            "seasonal": decomposition.seasonal.fillna(0).tolist(),
            "residual": decomposition.resid.fillna(0).tolist(),
            "observed": decomposition.observed.tolist()
        }


def forecast_arima(
    data: List[float],
    order: Tuple[int, int, int] = (1, 1, 1),
    steps: int = 10,
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> Dict[str, Any]:
    """ARIMA模型预测"""
    series = pd.Series(data)

    try:
        if seasonal_order != (0, 0, 0, 0):
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
        else:
            model = ARIMA(series, order=order)

        fitted_model = model.fit()

        # 生成预测
        forecast_result = fitted_model.forecast(steps=steps)
        forecast_values = forecast_result.tolist()

        # 预测置信区间
        pred_conf = fitted_model.get_forecast(steps=steps)
        conf_int = pred_conf.conf_int()

        return {
            "forecast": forecast_values,
            "conf_int_lower": conf_int.iloc[:, 0].tolist(),
            "conf_int_upper": conf_int.iloc[:, 1].tolist(),
            "model_aic": fitted_model.aic,
            "model_bic": fitted_model.bic
        }

    except Exception as e:
        raise ValueError(f"ARIMA预测失败: {str(e)}")