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
    """处理时间序列分析 - 增强版"""
    if not data or len(data) < 5:
        raise ValueError("时间序列数据至少需要5个观测点")
    
    # 基本统计量
    series = pd.Series(data)
    basic_stats = {
        "count": len(series),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
        "skewness": float(series.skew()),
        "kurtosis": float(series.kurtosis()),
        "variance": float(series.var()),
        "range": float(series.max() - series.min()),
        "cv": float(series.std() / series.mean()) if series.mean() != 0 else 0  # 变异系数
    }
    
    # 平稳性检验
    adf_result = stattools.adfuller(data)
    kpss_result = stattools.kpss(data, regression='c', nlags='auto')
    
    # 自相关分析
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
    
    # 计算更多诊断统计量
    # 趋势强度
    trend_strength = abs(np.corrcoef(range(len(data)), data)[0, 1]) if len(data) > 1 else 0
    
    # 季节性检测（如果数据足够长）
    seasonal_pattern = False
    if len(data) >= 12:
        try:
            # 简单的季节性检测：检查是否存在周期性模式
            seasonal_acf = stattools.acf(data, nlags=min(12, len(data)//2))
            seasonal_pattern = any(abs(x) > 0.3 for x in seasonal_acf[1:])
        except:
            seasonal_pattern = False
    
    # 构建详细的结果文本
    result_text = f"""📊 时间序列分析结果

🔍 基本统计信息：
- 观测数量 = {basic_stats['count']}
- 均值 = {basic_stats['mean']:.4f}
- 标准差 = {basic_stats['std']:.4f}
- 方差 = {basic_stats['variance']:.4f}
- 最小值 = {basic_stats['min']:.4f}
- 最大值 = {basic_stats['max']:.4f}
- 极差 = {basic_stats['range']:.4f}
- 中位数 = {basic_stats['median']:.4f}
- 偏度 = {basic_stats['skewness']:.4f}
- 峰度 = {basic_stats['kurtosis']:.4f}
- 变异系数 = {basic_stats['cv']:.4f}

📈 平稳性检验：
- ADF检验统计量 = {adf_result[0]:.4f}
- ADF检验p值 = {adf_result[1]:.4f}
- KPSS检验统计量 = {kpss_result[0]:.4f}
- KPSS检验p值 = {kpss_result[1]:.4f}
- 平稳性判断 = {'平稳' if adf_result[1] < 0.05 and kpss_result[1] > 0.05 else '非平稳'}

🔬 自相关分析：
- ACF前5阶: {[f'{x:.4f}' for x in acf_values[:5]]}
- PACF前5阶: {[f'{x:.4f}' for x in pacf_values[:5]]}
- 最大自相关: {max(abs(acf_values[1:])) if len(acf_values) > 1 else 0:.4f}
- 最大偏自相关: {max(abs(pacf_values[1:])) if len(pacf_values) > 1 else 0:.4f}

📊 诊断统计量：
- 趋势强度: {trend_strength:.4f}
- 季节性模式: {'存在' if seasonal_pattern else '未检测到'}
- 数据波动性: {'高' if basic_stats['cv'] > 0.5 else '中等' if basic_stats['cv'] > 0.2 else '低'}
- 分布形态: {'右偏' if basic_stats['skewness'] > 0.5 else '左偏' if basic_stats['skewness'] < -0.5 else '近似对称'}
- 峰度类型: {'尖峰' if basic_stats['kurtosis'] > 3 else '低峰' if basic_stats['kurtosis'] < 3 else '正态'}"""

    # 详细的模型建议
    result_text += f"\n\n💡 详细模型建议："
    
    if adf_result[1] < 0.05:  # 平稳序列
        result_text += f"\n- 数据为平稳序列，可直接建模"
        
        # 根据ACF/PACF模式给出详细建议
        acf_decay = abs(acf_values[1]) > 0.5
        pacf_cutoff = abs(pacf_values[1]) > 0.5 and all(abs(x) < 0.3 for x in pacf_values[2:5])
        
        if acf_decay and pacf_cutoff:
            result_text += f"\n- ACF缓慢衰减，PACF在1阶截尾，建议尝试AR(1)模型"
            result_text += f"\n- 可考虑ARMA(1,1)作为备选模型"
        elif not acf_decay and pacf_cutoff:
            result_text += f"\n- ACF快速衰减，PACF截尾，建议尝试MA模型"
        elif acf_decay and not pacf_cutoff:
            result_text += f"\n- ACF缓慢衰减，PACF无截尾，建议尝试AR模型"
        else:
            result_text += f"\n- ACF和PACF均缓慢衰减，建议尝试ARMA模型"
            
        # 根据数据特征给出额外建议
        if seasonal_pattern:
            result_text += f"\n- 检测到季节性模式，可考虑SARIMA模型"
        if trend_strength > 0.7:
            result_text += f"\n- 强趋势模式，可考虑带趋势项的模型"
            
    else:  # 非平稳序列
        result_text += f"\n- 数据为非平稳序列，建议进行差分处理"
        result_text += f"\n- 可尝试ARIMA(p,d,q)模型，其中d为差分阶数"
        
        # 根据趋势强度建议差分阶数
        if trend_strength > 0.8:
            result_text += f"\n- 强趋势，建议尝试1-2阶差分"
        elif trend_strength > 0.5:
            result_text += f"\n- 中等趋势，建议尝试1阶差分"
        else:
            result_text += f"\n- 弱趋势，可尝试1阶差分"
            
        if seasonal_pattern:
            result_text += f"\n- 检测到季节性模式，可考虑SARIMA模型"
    
    # 根据数据长度给出建议
    if len(data) < 30:
        result_text += f"\n- 数据量较少({len(data)}个观测点)，建议谨慎解释结果"
    elif len(data) < 100:
        result_text += f"\n- 数据量适中({len(data)}个观测点)，适合大多数时间序列模型"
    else:
        result_text += f"\n- 数据量充足({len(data)}个观测点)，可考虑复杂模型"
    
    result_text += f"\n\n⚠️ 建模注意事项："
    result_text += f"\n- 平稳性是时间序列建模的重要前提"
    result_text += f"\n- ACF和PACF模式有助于识别合适的模型阶数"
    result_text += f"\n- 建议结合信息准则（AIC/BIC）进行模型选择"
    result_text += f"\n- 模型诊断：检查残差的自相关性和正态性"
    result_text += f"\n- 模型验证：使用样本外数据进行预测验证"
    result_text += f"\n- 参数稳定性：确保模型参数在整个样本期内稳定"
    
    result_data = {
        "basic_statistics": basic_stats,
        "adf_statistic": float(adf_result[0]),
        "adf_pvalue": float(adf_result[1]),
        "kpss_statistic": float(kpss_result[0]),
        "kpss_pvalue": float(kpss_result[1]),
        "stationary": bool(adf_result[1] < 0.05 and kpss_result[1] > 0.05),
        "acf": [float(x) for x in acf_values.tolist()],
        "pacf": [float(x) for x in pacf_values.tolist()],
        "diagnostic_stats": {
            "trend_strength": trend_strength,
            "seasonal_pattern": seasonal_pattern,
            "volatility_level": "high" if basic_stats['cv'] > 0.5 else "medium" if basic_stats['cv'] > 0.2 else "low",
            "distribution_shape": "right_skewed" if basic_stats['skewness'] > 0.5 else "left_skewed" if basic_stats['skewness'] < -0.5 else "symmetric",
            "kurtosis_type": "leptokurtic" if basic_stats['kurtosis'] > 3 else "platykurtic" if basic_stats['kurtosis'] < 3 else "mesokurtic"
        },
        "model_suggestions": {
            "is_stationary": adf_result[1] < 0.05,
            "suggested_models": ["ARMA", "ARIMA"] if adf_result[1] < 0.05 else ["ARIMA", "SARIMA"],
            "data_sufficiency": "low" if len(data) < 30 else "medium" if len(data) < 100 else "high",
            "trend_recommendation": "strong_diff" if trend_strength > 0.8 else "moderate_diff" if trend_strength > 0.5 else "weak_diff",
            "seasonal_recommendation": "consider_seasonal" if seasonal_pattern else "no_seasonal"
        }
    }
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
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
    """处理固定效应模型 - 统一输出格式"""
    result = fixed_effects_model(y_data, x_data, entity_ids, time_periods, feature_names, entity_effects, time_effects)
    
    # 构建详细的结果文本
    result_text = f"""📊 固定效应模型分析结果

🔍 模型拟合信息：
- R² = {result.rsquared:.4f}
- 调整R² = {result.rsquared_adj:.4f}
- F统计量 = {result.f_statistic:.4f} (p = {result.f_pvalue:.4f})
- AIC = {result.aic:.2f}, BIC = {result.bic:.2f}
- 观测数量 = {result.n_obs}
- 个体效应 = {'是' if result.entity_effects else '否'}
- 时间效应 = {'是' if result.time_effects else '否'}

📈 回归系数详情："""
    
    # 添加系数信息
    for var_name, coef_info in result.coefficients.items():
        significance = "***" if coef_info["p_value"] < 0.01 else "**" if coef_info["p_value"] < 0.05 else "*" if coef_info["p_value"] < 0.1 else ""
        result_text += f"\n- {var_name}: {coef_info['coef']:.4f}{significance} (se={coef_info['std_err']:.4f}, p={coef_info['p_value']:.4f})"
    
    result_text += "\n\n💡 模型说明：固定效应模型通过组内变换消除个体固定差异，适用于个体间存在不可观测固定特征的情况。"
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        structuredContent=result.model_dump()
    )


async def handle_panel_random_effects(ctx, y_data, x_data, entity_ids, time_periods,
                                     feature_names=None, entity_effects=True, time_effects=False, **kwargs):
    """处理随机效应模型 - 统一输出格式"""
    result = random_effects_model(y_data, x_data, entity_ids, time_periods, feature_names, entity_effects, time_effects)
    
    # 构建详细的结果文本
    result_text = f"""📊 随机效应模型分析结果

🔍 模型拟合信息：
- R² = {result.rsquared:.4f}
- 调整R² = {result.rsquared_adj:.4f}
- F统计量 = {result.f_statistic:.4f} (p = {result.f_pvalue:.4f})
- AIC = {result.aic:.2f}, BIC = {result.bic:.2f}
- 观测数量 = {result.n_obs}
- 个体效应 = {'是' if result.entity_effects else '否'}
- 时间效应 = {'是' if result.time_effects else '否'}

📈 回归系数详情："""
    
    # 添加系数信息
    for var_name, coef_info in result.coefficients.items():
        significance = "***" if coef_info["p_value"] < 0.01 else "**" if coef_info["p_value"] < 0.05 else "*" if coef_info["p_value"] < 0.1 else ""
        result_text += f"\n- {var_name}: {coef_info['coef']:.4f}{significance} (se={coef_info['std_err']:.4f}, p={coef_info['p_value']:.4f})"
    
    result_text += "\n\n💡 模型说明：随机效应模型假设个体差异是随机的，比固定效应模型更有效率，但需要满足个体效应与解释变量不相关的假设。"
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        structuredContent=result.model_dump()
    )


async def handle_panel_hausman_test(ctx, y_data, x_data, entity_ids, time_periods, feature_names=None, **kwargs):
    """处理Hausman检验 - 统一输出格式"""
    result = hausman_test(y_data, x_data, entity_ids, time_periods, feature_names)
    
    result_text = f"""📊 Hausman检验结果

🔍 检验信息：
- 检验统计量 = {result.statistic:.4f}
- p值 = {result.p_value:.4f}
- 显著性 = {'是' if result.significant else '否'} (5%水平)

💡 模型选择建议：
{result.recommendation}

📋 决策规则：
- p值 < 0.05: 拒绝原假设，选择固定效应模型
- p值 >= 0.05: 不能拒绝原假设，选择随机效应模型

🔬 检验原理：Hausman检验用于判断个体效应是否与解释变量相关。原假设为随机效应模型是一致的。"""
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        structuredContent=result.model_dump()
    )


async def handle_panel_unit_root_test(ctx, **kwargs):
    """
    处理面板单位根检验 - 统一输出格式
    
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
    
    # 构建详细的结果文本
    result_text = f"""📊 面板单位根检验结果

🔍 检验信息：
- 检验方法 = {test_type.upper()}
- 个体数量 = {len(set(entity_ids))}
- 时间期数 = {len(set(time_periods))}
- 检验统计量 = {result.statistic:.4f}
- p值 = {result.p_value:.4f}
- 平稳性 = {'平稳' if result.stationary else '非平稳'} (5%水平)

📈 检验详情："""
    
    # 添加检验详情信息
    if hasattr(result, 'critical_values'):
        result_text += f"\n- 临界值: {result.critical_values}"
    if hasattr(result, 'lags_used'):
        result_text += f"\n- 使用滞后阶数: {result.lags_used}"
    if hasattr(result, 'test_statistic'):
        result_text += f"\n- 检验统计量: {result.test_statistic:.4f}"
    
    result_text += f"\n\n💡 检验说明：面板单位根检验用于判断面板数据是否平稳，是面板数据分析的重要前提检验。"
    result_text += f"\n\n⚠️ 注意事项：如果数据非平稳，需要进行差分处理或使用面板协整检验。"
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        structuredContent=result.model_dump()
    )


# 时间序列处理器
async def handle_var_model(ctx, data, max_lags=5, ic="aic", **kwargs):
    """处理VAR模型分析 - 统一输出格式"""
    result = var_model(data, max_lags=max_lags, ic=ic)
    
    # 构建详细的结果文本
    result_text = f"""📊 VAR模型分析结果

🔍 模型基本信息：
- 最优滞后阶数 = {result.order}
- 变量数量 = {len(result.variables) if hasattr(result, 'variables') else '未知'}
- 信息准则 = {ic.upper()}
- AIC = {result.aic:.2f}
- BIC = {getattr(result, 'bic', 'N/A')}
- HQIC = {getattr(result, 'hqic', 'N/A')}

📈 模型诊断信息："""
    
    # 添加模型诊断信息
    if hasattr(result, 'residuals_normality'):
        result_text += f"\n- 残差正态性检验: {result.residuals_normality}"
    if hasattr(result, 'serial_correlation'):
        result_text += f"\n- 序列相关性检验: {result.serial_correlation}"
    if hasattr(result, 'stability'):
        result_text += f"\n- 模型稳定性: {result.stability}"
    
    # 添加变量信息
    if hasattr(result, 'variables'):
        result_text += f"\n\n🔬 分析变量："
        for var in result.variables:
            result_text += f"\n- {var}"
    
    result_text += f"\n\n💡 模型说明：VAR模型用于分析多个时间序列变量间的动态关系，能够捕捉变量间的相互影响和滞后效应。"
    result_text += f"\n\n⚠️ 注意事项：VAR模型假设所有变量都是内生的，适用于分析变量间的动态交互关系。"
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        structuredContent=result.model_dump()
    )


async def handle_vecm_model(ctx, data, coint_rank=1, deterministic="co", max_lags=5, **kwargs):
    """处理VECM模型分析 - 统一输出格式"""
    result = vecm_model(data, coint_rank=coint_rank, deterministic=deterministic, max_lags=max_lags)
    
    # 构建详细的结果文本
    result_text = f"""📊 VECM模型分析结果

🔍 模型基本信息：
- 协整秩 = {result.coint_rank}
- 确定性项类型 = {deterministic}
- 最大滞后阶数 = {max_lags}
- AIC = {result.aic:.2f}
- BIC = {getattr(result, 'bic', 'N/A')}
- HQIC = {getattr(result, 'hqic', 'N/A')}

📈 协整关系分析："""
    
    # 添加协整关系信息
    if hasattr(result, 'coint_relations'):
        result_text += f"\n- 协整关系数量: {len(result.coint_relations)}"
        for i, relation in enumerate(result.coint_relations[:3], 1):  # 显示前3个关系
            result_text += f"\n- 关系{i}: {relation}"
        if len(result.coint_relations) > 3:
            result_text += f"\n- ... 还有{len(result.coint_relations) - 3}个协整关系"
    
    # 添加误差修正项信息
    if hasattr(result, 'error_correction'):
        result_text += f"\n\n🔧 误差修正机制："
        result_text += f"\n- 误差修正项显著性: {result.error_correction}"
    
    result_text += f"\n\n💡 模型说明：VECM模型用于分析非平稳时间序列的长期均衡关系，包含误差修正机制来反映短期调整过程。"
    result_text += f"\n\n⚠️ 注意事项：VECM模型要求变量间存在协整关系，适用于分析经济变量的长期均衡和短期动态调整。"
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        structuredContent=result.model_dump()
    )


async def handle_garch_model(ctx, data, order=(1, 1), dist="normal", **kwargs):
    """处理GARCH模型分析 - 统一输出格式"""
    result = garch_model(data, order=order, dist=dist)
    
    # 构建详细的结果文本
    result_text = f"""📊 GARCH模型分析结果

🔍 模型基本信息：
- GARCH阶数 = ({order[0]}, {order[1]})
- 误差分布 = {dist}
- 持久性 = {result.persistence:.4f}
- AIC = {result.aic:.2f}
- BIC = {getattr(result, 'bic', 'N/A')}

📈 波动率特征："""
    
    # 添加波动率特征信息
    if hasattr(result, 'volatility_persistence'):
        result_text += f"\n- 波动率持续性: {result.volatility_persistence:.4f}"
    if hasattr(result, 'unconditional_variance'):
        result_text += f"\n- 无条件方差: {result.unconditional_variance:.4f}"
    if hasattr(result, 'leverage_effect'):
        result_text += f"\n- 杠杆效应: {result.leverage_effect}"
    
    # 添加模型诊断信息
    if hasattr(result, 'residuals_test'):
        result_text += f"\n\n🔧 模型诊断："
        result_text += f"\n- 残差检验: {result.residuals_test}"
    
    result_text += f"\n\n💡 模型说明：GARCH模型用于分析金融时间序列的波动率聚类现象，能够捕捉条件异方差性。"
    result_text += f"\n\n⚠️ 注意事项：GARCH模型适用于金融数据波动率建模，阶数选择影响模型对波动率持续性的捕捉能力。"
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        structuredContent=result.model_dump()
    )


async def handle_state_space_model(ctx, data, state_dim=1, observation_dim=1,
                                  trend=True, seasonal=False, period=12, **kwargs):
    """处理状态空间模型分析 - 统一输出格式"""
    result = state_space_model(data, state_dim, observation_dim, trend, seasonal, period)
    
    # 构建详细的结果文本
    result_text = f"""📊 状态空间模型分析结果

🔍 模型结构信息：
- 状态维度 = {state_dim}
- 观测维度 = {observation_dim}
- 趋势项 = {'包含' if trend else '不包含'}
- 季节项 = {'包含' if seasonal else '不包含'}
- 季节周期 = {period if seasonal else 'N/A'}
- AIC = {result.aic:.2f}
- BIC = {result.bic:.2f}
- 对数似然值 = {result.log_likelihood:.2f}

📈 状态分析："""

    # 添加状态信息
    if result.state_names:
        result_text += f"\n- 状态变量: {', '.join(result.state_names)}"
    if result.observation_names:
        result_text += f"\n- 观测变量: {', '.join(result.observation_names)}"
    
    # 添加状态估计信息
    if result.filtered_state:
        result_text += f"\n- 滤波状态估计: 已计算"
    if result.smoothed_state:
        result_text += f"\n- 平滑状态估计: 已计算"

    result_text += f"\n\n💡 模型说明：状态空间模型用于分析时间序列的潜在状态和观测关系，能够处理复杂的动态系统，特别适用于具有不可观测状态的时间序列建模。"
    result_text += f"\n\n⚠️ 注意事项：状态空间模型参数估计可能对初始值敏感，建议进行多次初始化尝试以获得稳定结果。"

    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        structuredContent=result.model_dump()
    )


async def handle_variance_decomposition(ctx, data, periods=10, max_lags=5, **kwargs):
    """处理方差分解分析 - 统一输出格式"""
    result = variance_decomposition(data, periods=periods, max_lags=max_lags)
    
    # 构建详细的结果文本
    result_text = f"""📊 方差分解分析结果

🔍 分析设置：
- 分解期数 = {periods}
- 最大滞后阶数 = {max_lags}
- 变量数量 = {len(data) if data else '未知'}

📈 方差分解结果："""

    # 添加方差分解结果
    if isinstance(result, dict) and "variance_decomposition" in result:
        variance_decomp = result["variance_decomposition"]
        horizon = result.get("horizon", periods)
        
        result_text += f"\n- 分析期数: {horizon}期"
        
        for var_name, decomposition in variance_decomp.items():
            result_text += f"\n\n🔬 变量 '{var_name}' 的方差来源："
            if isinstance(decomposition, dict):
                for source, percentages in decomposition.items():
                    if isinstance(percentages, list) and len(percentages) > 0:
                        # 显示最后一期的贡献度
                        final_percentage = percentages[-1] * 100 if isinstance(percentages[-1], (int, float)) else 0
                        result_text += f"\n- {source}: {final_percentage:.1f}%"
                    else:
                        result_text += f"\n- {source}: {percentages:.1f}%"
            else:
                result_text += f"\n- 总方差: {decomposition:.1f}%"
    else:
        result_text += f"\n- 结果格式异常，无法解析方差分解结果"

    result_text += f"\n\n💡 分析说明：方差分解用于分析多变量系统中各变量对预测误差方差的贡献程度，反映变量间的动态影响关系。"
    result_text += f"\n\n⚠️ 注意事项：方差分解结果依赖于VAR模型的滞后阶数选择，不同期数的分解结果反映短期和长期影响。"

    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        structuredContent=result
    )


# 机器学习处理器
async def handle_random_forest(ctx, y_data, x_data, feature_names=None, n_estimators=100, max_depth=None, **kwargs):
    """处理随机森林回归 - 统一输出格式"""
    result = random_forest_regression(y_data, x_data, feature_names, n_estimators, max_depth)
    
    # 检查R²是否为负值
    r2_warning = ""
    if result.r2_score < 0:
        r2_warning = f"\n⚠️ 警告：R²为负值({result.r2_score:.4f})，表明模型性能比简单均值预测更差。建议：1) 检查数据质量 2) 增加样本数量 3) 调整模型参数"
    
    # 构建详细的结果文本
    result_text = f"""📊 随机森林回归分析结果

🔍 模型拟合信息：
- R² = {result.r2_score:.4f}
- 均方误差(MSE) = {result.mse:.4f}
- 平均绝对误差(MAE) = {result.mae:.4f}
- 样本数量 = {result.n_obs}
- 树的数量 = {result.n_estimators}
- 最大深度 = {result.max_depth if result.max_depth else '无限制'}
- 袋外得分 = {f"{result.oob_score:.4f}" if result.oob_score else '未计算'}
{r2_warning}

📈 特征重要性（前10个）："""
    
    # 添加特征重要性信息，按重要性排序
    if result.feature_importance:
        sorted_features = sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            result_text += f"\n- {feature}: {importance:.4f}"
        if len(sorted_features) > 10:
            result_text += f"\n- ... 还有{len(sorted_features) - 10}个特征"
    else:
        result_text += "\n- 特征重要性未计算"
    
    result_text += f"\n\n💡 模型说明：随机森林通过构建多个决策树并集成结果，能够处理非线性关系和特征交互，对异常值稳健且不易过拟合。"
    result_text += f"\n\n⚠️ 注意事项：随机森林是黑盒模型，可解释性较差，但预测性能通常较好。"
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        structuredContent=result.model_dump()
    )


async def handle_gradient_boosting(ctx, y_data, x_data, feature_names=None,
                                  n_estimators=100, learning_rate=0.1, max_depth=3, **kwargs):
    """处理梯度提升树回归 - 统一输出格式"""
    result = gradient_boosting_regression(y_data, x_data, feature_names, n_estimators, learning_rate, max_depth)
    
    # 检查R²是否为负值
    r2_warning = ""
    if result.r2_score < 0:
        r2_warning = f"\n⚠️ 警告：R²为负值({result.r2_score:.4f})，表明模型性能比简单均值预测更差。建议：1) 检查数据质量 2) 增加样本数量 3) 调整模型参数"
    
    # 构建详细的结果文本
    result_text = f"""📊 梯度提升树回归分析结果

🔍 模型拟合信息：
- R² = {result.r2_score:.4f}
- 均方误差(MSE) = {result.mse:.4f}
- 平均绝对误差(MAE) = {result.mae:.4f}
- 样本数量 = {result.n_obs}
- 树的数量 = {result.n_estimators}
- 学习率 = {result.learning_rate}
- 最大深度 = {result.max_depth}
{r2_warning}

📈 特征重要性（前10个）："""
    
    # 添加特征重要性信息，按重要性排序
    if result.feature_importance:
        sorted_features = sorted(result.feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            result_text += f"\n- {feature}: {importance:.4f}"
        if len(sorted_features) > 10:
            result_text += f"\n- ... 还有{len(sorted_features) - 10}个特征"
    else:
        result_text += "\n- 特征重要性未计算"
    
    result_text += f"\n\n💡 模型说明：梯度提升树通过顺序构建决策树，每棵树修正前一棵树的错误，能够处理复杂的非线性关系，通常具有很高的预测精度。"
    result_text += f"\n\n⚠️ 注意事项：梯度提升树对参数敏感，需要仔细调优，训练时间较长但预测性能优秀。"
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        structuredContent=result.model_dump()
    )


async def handle_lasso_regression(ctx, y_data, x_data, feature_names=None, alpha=1.0, **kwargs):
    """处理Lasso回归 - 统一输出格式"""
    result = lasso_regression(y_data, x_data, feature_names, alpha)
    
    # 检查R²是否为负值
    r2_warning = ""
    if result.r2_score < 0:
        r2_warning = f"\n⚠️ 警告：R²为负值({result.r2_score:.4f})，表明模型性能比简单均值预测更差。建议：1) 检查数据质量 2) 尝试更小的alpha值 3) 增加样本数量"
    
    # 检查系数是否全为0
    coef_warning = ""
    if all(abs(coef) < 1e-10 for coef in result.coefficients.values()):
        coef_warning = f"\n⚠️ 警告：所有系数都被压缩为0，正则化参数alpha={alpha}可能过大，建议减小alpha值"
    
    # 构建详细的结果文本
    result_text = f"""📊 Lasso回归分析结果

🔍 模型拟合信息：
- R² = {result.r2_score:.4f}
- 均方误差(MSE) = {result.mse:.4f}
- 平均绝对误差(MAE) = {result.mae:.4f}
- 样本数量 = {result.n_obs}
- 正则化参数(alpha) = {result.alpha}
{r2_warning}{coef_warning}

📈 回归系数详情："""
    
    # 添加系数信息，按绝对值排序
    sorted_coefficients = sorted(result.coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    for var_name, coef in sorted_coefficients:
        if abs(coef) > 1e-10:  # 只显示非零系数
            result_text += f"\n- {var_name}: {coef:.4f}"
        else:
            result_text += f"\n- {var_name}: 0.0000 (被压缩)"
    
    result_text += f"\n\n💡 模型说明：Lasso回归使用L1正则化进行特征选择，能够自动将不重要的特征系数压缩为0，适用于高维数据和特征选择场景。"
    result_text += f"\n\n⚠️ 注意事项：由于数据标准化，系数大小需要谨慎解释。"
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        structuredContent=result.model_dump()
    )


async def handle_ridge_regression(ctx, y_data, x_data, feature_names=None, alpha=1.0, **kwargs):
    """处理Ridge回归 - 统一输出格式"""
    result = ridge_regression(y_data, x_data, feature_names, alpha)
    
    # 检查R²是否为负值
    r2_warning = ""
    if result.r2_score < 0:
        r2_warning = f"\n⚠️ 警告：R²为负值({result.r2_score:.4f})，表明模型性能比简单均值预测更差。建议：1) 检查数据质量 2) 尝试更小的alpha值 3) 增加样本数量"
    
    # 构建详细的结果文本
    result_text = f"""📊 Ridge回归分析结果

🔍 模型拟合信息：
- R² = {result.r2_score:.4f}
- 均方误差(MSE) = {result.mse:.4f}
- 平均绝对误差(MAE) = {result.mae:.4f}
- 样本数量 = {result.n_obs}
- 正则化参数(alpha) = {result.alpha}
{r2_warning}

📈 回归系数详情："""
    
    # 添加系数信息，按绝对值排序
    sorted_coefficients = sorted(result.coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    for var_name, coef in sorted_coefficients:
        result_text += f"\n- {var_name}: {coef:.4f}"
    
    result_text += f"\n\n💡 模型说明：Ridge回归使用L2正则化处理多重共线性问题，对所有系数进行收缩但不进行特征选择，适用于需要稳定估计的场景。"
    result_text += f"\n\n⚠️ 注意事项：由于数据标准化，系数大小需要谨慎解释。"
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        structuredContent=result.model_dump()
    )


async def handle_cross_validation(ctx, y_data, x_data, model_type="random_forest", cv_folds=5, scoring="r2", **kwargs):
    """处理交叉验证 - 统一输出格式"""
    result = cross_validation(y_data, x_data, model_type, cv_folds, scoring)
    
    # 构建详细的结果文本
    result_text = f"""📊 交叉验证分析结果

🔍 验证信息：
- 模型类型 = {result.model_type}
- 交叉验证折数 = {result.n_splits}
- 评分指标 = {scoring}
- 平均得分 = {result.mean_score:.4f}
- 得分标准差 = {result.std_score:.4f}
- 变异系数 = {(result.std_score / abs(result.mean_score)) * 100 if result.mean_score != 0 else 0:.2f}%

📈 各折得分详情："""
    
    # 添加各折得分
    for i, score in enumerate(result.cv_scores, 1):
        result_text += f"\n- 第{i}折: {score:.4f}"
    
    # 评估模型稳定性
    stability_assessment = ""
    cv_threshold = 0.1  # 10%的变异系数阈值
    cv_value = (result.std_score / abs(result.mean_score)) if result.mean_score != 0 else 0
    
    if cv_value < cv_threshold:
        stability_assessment = f"\n\n✅ 模型稳定性：优秀（变异系数{cv_value*100:.2f}% < {cv_threshold*100:.0f}%）"
    elif cv_value < cv_threshold * 2:
        stability_assessment = f"\n\n⚠️ 模型稳定性：一般（变异系数{cv_value*100:.2f}% 在{cv_threshold*100:.0f}%-{cv_threshold*2*100:.0f}%之间）"
    else:
        stability_assessment = f"\n\n❌ 模型稳定性：较差（变异系数{cv_value*100:.2f}% > {cv_threshold*2*100:.0f}%）"
    
    result_text += stability_assessment
    result_text += f"\n\n💡 模型说明：交叉验证通过将数据分成多个子集进行训练和测试，评估模型的泛化能力和稳定性。"
    result_text += f"\n\n⚠️ 注意事项：变异系数越小表明模型越稳定，建议选择变异系数小于10%的模型。"
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        structuredContent=result.model_dump()
    )


async def handle_feature_importance(ctx, y_data, x_data, feature_names=None, method="random_forest", top_k=5, **kwargs):
    """处理特征重要性分析 - 统一输出格式"""
    result = feature_importance_analysis(y_data, x_data, feature_names, method, top_k)
    
    # 构建详细的结果文本
    result_text = f"""📊 特征重要性分析结果

🔍 分析信息：
- 分析方法 = {method}
- 显示Top特征数量 = {top_k}
- 总特征数量 = {len(result.feature_importance)}

📈 特征重要性排名："""
    
    # 添加特征重要性信息
    for i, (feature, importance) in enumerate(result.sorted_features[:top_k], 1):
        percentage = (importance / sum(result.feature_importance.values())) * 100 if sum(result.feature_importance.values()) > 0 else 0
        result_text += f"\n{i}. {feature}: {importance:.4f} ({percentage:.1f}%)"
    
    # 添加重要性分布信息
    if len(result.sorted_features) > 0:
        top_k_importance = sum(imp for _, imp in result.sorted_features[:top_k])
        total_importance = sum(result.feature_importance.values())
        top_k_percentage = (top_k_importance / total_importance) * 100 if total_importance > 0 else 0
        
        result_text += f"\n\n📊 重要性分布："
        result_text += f"\n- Top {top_k}特征累计重要性: {top_k_percentage:.1f}%"
        result_text += f"\n- 剩余特征重要性: {100 - top_k_percentage:.1f}%"
    
    result_text += f"\n\n💡 分析说明：特征重要性分析帮助识别对预测目标最重要的变量，可用于特征选择和模型解释。"
    result_text += f"\n\n⚠️ 注意事项：不同方法计算的特征重要性可能不同，建议结合业务知识进行解释。"
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        structuredContent=result.model_dump()
    )