"""
工具处理器模块
集中管理所有工具的核心业务逻辑
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa import stattools
from scipy import stats
from typing import Dict, List, Any, Optional
from mcp.types import CallToolResult, TextContent

from .statistics import calculate_descriptive_stats, calculate_correlation_matrix, perform_hypothesis_test
from .regression import perform_ols_regression
from .panel_data import fixed_effects_model, random_effects_model, hausman_test, panel_unit_root_test
from .time_series import var_model, vecm_model, garch_model, state_space_model, variance_decomposition
from .machine_learning import (
    random_forest_regression, gradient_boosting_regression,
    lasso_regression, ridge_regression, cross_validation, feature_importance_analysis
)


async def handle_descriptive_statistics(ctx, data: Dict[str, List[float]], **kwargs) -> CallToolResult:
    """处理描述性统计"""
    if not data:
        raise ValueError("数据不能为空")
    
    df = pd.DataFrame(data)
    
    # 计算统计量
    result_data = {
        "count": len(df),
        "mean": float(df.mean().mean()),
        "std": float(df.std().mean()),
        "min": float(df.min().min()),
        "max": float(df.max().max()),
        "median": float(df.median().mean()),
        "skewness": float(df.skew().mean()),
        "kurtosis": float(df.kurtosis().mean())
    }
    
    correlation_matrix = df.corr().round(4)
    
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"描述性统计结果：\n"
                     f"均值: {result_data['mean']:.4f}\n"
                     f"标准差: {result_data['std']:.4f}\n"
                     f"最小值: {result_data['min']:.4f}\n"
                     f"最大值: {result_data['max']:.4f}\n"
                     f"中位数: {result_data['median']:.4f}\n"
                     f"偏度: {result_data['skewness']:.4f}\n"
                     f"峰度: {result_data['kurtosis']:.4f}\n\n"
                     f"相关系数矩阵：\n{correlation_matrix.to_string()}"
            )
        ],
        structuredContent=result_data
    )


async def handle_ols_regression(ctx, y_data: List[float], x_data: List[List[float]], 
                                feature_names: Optional[List[str]] = None, **kwargs) -> CallToolResult:
    """处理OLS回归"""
    if not y_data or not x_data:
        raise ValueError("因变量和自变量数据不能为空")
    
    X = np.array(x_data)
    y = np.array(y_data)
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    
    if feature_names is None:
        feature_names = [f"x{i+1}" for i in range(X.shape[1])]
    
    conf_int = model.conf_int()
    coefficients = {}
    
    for i, coef in enumerate(model.params):
        var_name = "const" if i == 0 else feature_names[i-1]
        coefficients[var_name] = {
            "coef": float(coef),
            "std_err": float(model.bse[i]),
            "t_value": float(model.tvalues[i]),
            "p_value": float(model.pvalues[i]),
            "ci_lower": float(conf_int[i][0]),
            "ci_upper": float(conf_int[i][1])
        }
    
    result_data = {
        "rsquared": float(model.rsquared),
        "rsquared_adj": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "coefficients": coefficients
    }
    
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"OLS回归分析结果：\n"
                     f"R² = {result_data['rsquared']:.4f}\n"
                     f"调整R² = {result_data['rsquared_adj']:.4f}\n"
                     f"F统计量 = {result_data['f_statistic']:.4f} (p = {result_data['f_pvalue']:.4f})\n"
                     f"AIC = {result_data['aic']:.2f}, BIC = {result_data['bic']:.2f}\n\n"
                     f"回归系数：\n{model.summary().tables[1]}"
            )
        ],
        structuredContent=result_data
    )


async def handle_hypothesis_testing(ctx, data1: List[float], data2: Optional[List[float]] = None,
                                    test_type: str = "t_test", **kwargs) -> CallToolResult:
    """处理假设检验"""
    if test_type == "t_test":
        if data2 is None:
            result = stats.ttest_1samp(data1, 0)
            ci = stats.t.interval(0.95, len(data1)-1, loc=np.mean(data1), scale=stats.sem(data1))
        else:
            result = stats.ttest_ind(data1, data2)
            ci = None
        
        test_result = {
            "test_type": test_type,
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "significant": bool(result.pvalue < 0.05),
            "confidence_interval": list(ci) if ci else None
        }
    elif test_type == "adf":
        result = stattools.adfuller(data1)
        test_result = {
            "test_type": "adf",
            "statistic": float(result[0]),
            "p_value": float(result[1]),
            "significant": bool(result[1] < 0.05),
            "confidence_interval": None
        }
    else:
        raise ValueError(f"不支持的检验类型: {test_type}")
    
    ci_text = ""
    if test_result['confidence_interval']:
        ci_lower = test_result['confidence_interval'][0]
        ci_upper = test_result['confidence_interval'][1]
        ci_text = f"95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]"
    
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"{test_type.upper()}检验结果：\n"
                     f"检验统计量 = {test_result['statistic']:.4f}\n"
                     f"p值 = {test_result['p_value']:.4f}\n"
                     f"{'显著' if test_result['significant'] else '不显著'} (5%水平)\n"
                     f"{ci_text}"
            )
        ],
        structuredContent=test_result
    )


async def handle_time_series_analysis(ctx, data: List[float], **kwargs) -> CallToolResult:
    """处理时间序列分析"""
    if not data or len(data) < 5:
        raise ValueError("时间序列数据至少需要5个观测点")
    
    adf_result = stattools.adfuller(data)
    max_nlags = min(20, len(data) - 1, len(data) // 2)
    if max_nlags < 1:
        max_nlags = 1
    
    try:
        acf_values = stattools.acf(data, nlags=max_nlags)
        pacf_values = stattools.pacf(data, nlags=max_nlags)
    except:
        acf_values = np.zeros(max_nlags + 1)
        pacf_values = np.zeros(max_nlags + 1)
        acf_values[0] = pacf_values[0] = 1.0
    
    result_data = {
        "adf_statistic": float(adf_result[0]),
        "adf_pvalue": float(adf_result[1]),
        "stationary": bool(adf_result[1] < 0.05),
        "acf": [float(x) for x in acf_values.tolist()],
        "pacf": [float(x) for x in pacf_values.tolist()]
    }
    
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"时间序列分析结果：\n"
                     f"ADF检验统计量 = {result_data['adf_statistic']:.4f}\n"
                     f"ADF检验p值 = {result_data['adf_pvalue']:.4f}\n"
                     f"{'平稳' if result_data['stationary'] else '非平稳'}序列\n"
                     f"ACF前5阶: {result_data['acf'][:5]}\n"
                     f"PACF前5阶: {result_data['pacf'][:5]}"
            )
        ],
        structuredContent=result_data
    )


async def handle_correlation_analysis(ctx, data: Dict[str, List[float]], 
                                     method: str = "pearson", **kwargs) -> CallToolResult:
    """处理相关性分析"""
    if not data or len(data) < 2:
        raise ValueError("至少需要2个变量进行相关性分析")
    
    df = pd.DataFrame(data)
    correlation_matrix = df.corr(method=method)
    
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"{method.title()}相关系数矩阵：\n{correlation_matrix.round(4).to_string()}"
            )
        ]
    )


# 面板数据处理器
async def handle_panel_fixed_effects(ctx, y_data, x_data, entity_ids, time_periods, 
                                    feature_names=None, entity_effects=True, time_effects=False, **kwargs):
    result = fixed_effects_model(y_data, x_data, entity_ids, time_periods, feature_names, entity_effects, time_effects)
    return CallToolResult(
        content=[TextContent(type="text", text=f"固定效应模型: R²={result.rsquared:.4f}")],
        structuredContent=result.model_dump()
    )


async def handle_panel_random_effects(ctx, y_data, x_data, entity_ids, time_periods,
                                     feature_names=None, entity_effects=True, time_effects=False, **kwargs):
    result = random_effects_model(y_data, x_data, entity_ids, time_periods, feature_names, entity_effects, time_effects)
    return CallToolResult(
        content=[TextContent(type="text", text=f"随机效应模型: R²={result.rsquared:.4f}")],
        structuredContent=result.model_dump()
    )


async def handle_panel_hausman_test(ctx, y_data, x_data, entity_ids, time_periods, feature_names=None, **kwargs):
    result = hausman_test(y_data, x_data, entity_ids, time_periods, feature_names)
    return CallToolResult(
        content=[TextContent(type="text", text=f"Hausman检验: p={result.p_value:.4f}, 建议={result.recommendation}")],
        structuredContent=result.model_dump()
    )


async def handle_panel_unit_root_test(ctx, **kwargs):
    """
    处理面板单位根检验
    
    panel_unit_root_test函数期望：data, entity_ids, time_periods
    但panel装饰器会传入：y_data, x_data, entity_ids, time_periods
    """
    # 提取参数
    data = kwargs.get('data')
    y_data = kwargs.get('y_data')
    entity_ids = kwargs.get('entity_ids')
    time_periods = kwargs.get('time_periods')
    test_type = kwargs.get('test_type', 'levinlin')
    
    # 如果没有data但有y_data，使用y_data（来自panel装饰器）
    if data is None and y_data is not None:
        data = y_data
    
    if data is None:
        raise ValueError("需要提供数据（data或y_data）")
    
    if entity_ids is None or time_periods is None:
        raise ValueError("需要提供entity_ids和time_periods")
    
    # 只传递panel_unit_root_test需要的参数
    result = panel_unit_root_test(data, entity_ids, time_periods, test_type)
    return CallToolResult(
        content=[TextContent(type="text", text=f"面板单位根检验: {'平稳' if result.stationary else '非平稳'}")],
        structuredContent=result.model_dump()
    )


# 时间序列处理器
async def handle_var_model(ctx, data, max_lags=5, ic="aic", **kwargs):
    result = var_model(data, max_lags=max_lags, ic=ic)
    return CallToolResult(
        content=[TextContent(type="text", text=f"VAR模型: 滞后阶数={result.order}, AIC={result.aic:.2f}")],
        structuredContent=result.model_dump()
    )


async def handle_vecm_model(ctx, data, coint_rank=1, deterministic="co", max_lags=5, **kwargs):
    result = vecm_model(data, coint_rank=coint_rank, deterministic=deterministic, max_lags=max_lags)
    return CallToolResult(
        content=[TextContent(type="text", text=f"VECM模型: 协整秩={result.coint_rank}, AIC={result.aic:.2f}")],
        structuredContent=result.model_dump()
    )


async def handle_garch_model(ctx, data, order=(1, 1), dist="normal", **kwargs):
    result = garch_model(data, order=order, dist=dist)
    return CallToolResult(
        content=[TextContent(type="text", text=f"GARCH模型: 持久性={result.persistence:.4f}")],
        structuredContent=result.model_dump()
    )


async def handle_state_space_model(ctx, data, state_dim=1, observation_dim=1, 
                                  trend=True, seasonal=False, period=12, **kwargs):
    result = state_space_model(data, state_dim, observation_dim, trend, seasonal, period)
    return CallToolResult(
        content=[TextContent(type="text", text=f"状态空间模型: AIC={result.aic:.2f}")],
        structuredContent=result.model_dump()
    )


async def handle_variance_decomposition(ctx, data, periods=10, max_lags=5, **kwargs):
    result = variance_decomposition(data, periods=periods, max_lags=max_lags)
    return CallToolResult(
        content=[TextContent(type="text", text=f"方差分解: {periods}期")],
        structuredContent=result
    )


# 机器学习处理器
async def handle_random_forest(ctx, y_data, x_data, feature_names=None, n_estimators=100, max_depth=None, **kwargs):
    result = random_forest_regression(y_data, x_data, feature_names, n_estimators, max_depth)
    return CallToolResult(
        content=[TextContent(type="text", text=f"随机森林: R²={result.r2_score:.4f}")],
        structuredContent=result.model_dump()
    )


async def handle_gradient_boosting(ctx, y_data, x_data, feature_names=None, 
                                  n_estimators=100, learning_rate=0.1, max_depth=3, **kwargs):
    result = gradient_boosting_regression(y_data, x_data, feature_names, n_estimators, learning_rate, max_depth)
    return CallToolResult(
        content=[TextContent(type="text", text=f"梯度提升树: R²={result.r2_score:.4f}")],
        structuredContent=result.model_dump()
    )


async def handle_lasso_regression(ctx, y_data, x_data, feature_names=None, alpha=1.0, **kwargs):
    result = lasso_regression(y_data, x_data, feature_names, alpha)
    return CallToolResult(
        content=[TextContent(type="text", text=f"Lasso回归: R²={result.r2_score:.4f}")],
        structuredContent=result.model_dump()
    )


async def handle_ridge_regression(ctx, y_data, x_data, feature_names=None, alpha=1.0, **kwargs):
    result = ridge_regression(y_data, x_data, feature_names, alpha)
    return CallToolResult(
        content=[TextContent(type="text", text=f"Ridge回归: R²={result.r2_score:.4f}")],
        structuredContent=result.model_dump()
    )


async def handle_cross_validation(ctx, y_data, x_data, model_type="random_forest", cv_folds=5, scoring="r2", **kwargs):
    result = cross_validation(y_data, x_data, model_type, cv_folds, scoring)
    return CallToolResult(
        content=[TextContent(type="text", text=f"交叉验证: 平均得分={result.mean_score:.4f}")],
        structuredContent=result.model_dump()
    )


async def handle_feature_importance(ctx, y_data, x_data, feature_names=None, method="random_forest", top_k=5, **kwargs):
    result = feature_importance_analysis(y_data, x_data, feature_names, method, top_k)
    return CallToolResult(
        content=[TextContent(type="text", text=f"特征重要性: Top特征={result.top_features}")],
        structuredContent=result.model_dump()
    )