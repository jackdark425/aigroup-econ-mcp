# -*- coding: utf-8 -*-
"""
基础参数估计处理器
"""

from typing import Dict, List, Any, Optional
from mcp.types import CallToolResult, TextContent


async def handle_ols_regression(ctx, y_data: List[float], x_data: List[List[float]], 
                                feature_names: Optional[List[str]] = None, **kwargs) -> CallToolResult:
    """处理OLS回归（基础参数估计模块）"""
    return CallToolResult(
        content=[TextContent(type="text", text="OLS回归 - 功能待实现")]
    )


async def handle_mle_estimation(ctx, data, model_type, **kwargs):
    """处理最大似然估计"""
    return CallToolResult(
        content=[TextContent(type="text", text="最大似然估计 - 功能待实现")]
    )


async def handle_gmm_estimation(ctx, data, moment_conditions, **kwargs):
    """处理广义矩估计"""
    return CallToolResult(
        content=[TextContent(type="text", text="广义矩估计 - 功能待实现")]
    )
# -*- coding: utf-8 -*-
"""
机器学习处理器
"""

from typing import Dict, List, Any, Optional
from mcp.types import CallToolResult, TextContent


async def handle_random_forest(ctx, y_data, x_data, feature_names, **kwargs):
    """处理随机森林"""
    return CallToolResult(
        content=[TextContent(type="text", text="随机森林回归 - 功能待实现")]
    )


async def handle_gradient_boosting(ctx, y_data, x_data, feature_names, **kwargs):
    """处理梯度提升"""
    return CallToolResult(
        content=[TextContent(type="text", text="梯度提升回归 - 功能待实现")]
    )


async def handle_lasso_regression(ctx, y_data, x_data, feature_names, **kwargs):
    """处理LASSO回归"""
    return CallToolResult(
        content=[TextContent(type="text", text="LASSO回归 - 功能待实现")]
    )


async def handle_ridge_regression(ctx, y_data, x_data, feature_names, **kwargs):
    """处理岭回归"""
    return CallToolResult(
        content=[TextContent(type="text", text="岭回归 - 功能待实现")]
    )


async def handle_cross_validation(ctx, y_data, x_data, feature_names, **kwargs):
    """处理交叉验证"""
    return CallToolResult(
        content=[TextContent(type="text", text="交叉验证 - 功能待实现")]
    )


async def handle_feature_importance(ctx, y_data, x_data, feature_names, **kwargs):
    """处理特征重要性"""
    return CallToolResult(
        content=[TextContent(type="text", text="特征重要性分析 - 功能待实现")]
    )
# -*- coding: utf-8 -*-
"""
时间序列模型处理器
"""

from typing import Dict, List, Any, Optional
from mcp.types import CallToolResult, TextContent


async def handle_var_model(ctx, data, max_lags, selection_criteria, **kwargs):
    """处理VAR模型"""
    return CallToolResult(
        content=[TextContent(type="text", text="VAR模型分析 - 功能待实现")]
    )


async def handle_vecm_model(ctx, data, max_lags, selection_criteria, **kwargs):
    """处理VECM模型"""
    return CallToolResult(
        content=[TextContent(type="text", text="VECM模型分析 - 功能待实现")]
    )


async def handle_garch_model(ctx, data, p, q, **kwargs):
    """处理GARCH模型"""
    return CallToolResult(
        content=[TextContent(type="text", text="GARCH模型分析 - 功能待实现")]
    )


async def handle_state_space_model(ctx, data, **kwargs):
    """处理状态空间模型"""
    return CallToolResult(
        content=[TextContent(type="text", text="状态空间模型 - 功能待实现")]
    )


async def handle_variance_decomposition(ctx, data, **kwargs):
    """处理方差分解"""
    return CallToolResult(
        content=[TextContent(type="text", text="方差分解分析 - 功能待实现")]
    )
# -*- coding: utf-8 -*-
"""
面板数据分析处理器
"""

from typing import Dict, List, Any, Optional
from mcp.types import CallToolResult, TextContent


async def handle_panel_fixed_effects(ctx, y_data, x_data, entity_ids, time_periods,
                                    feature_names=None, entity_effects=True, time_effects=False, **kwargs):
    """处理面板固定效应模型"""
    return CallToolResult(
        content=[TextContent(type="text", text="面板固定效应模型 - 功能待实现")]
    )


async def handle_panel_random_effects(ctx, y_data, x_data, entity_ids, time_periods,
                                     feature_names=None, entity_effects=True, time_effects=False, **kwargs):
    """处理面板随机效应模型"""
    return CallToolResult(
        content=[TextContent(type="text", text="面板随机效应模型 - 功能待实现")]
    )


async def handle_panel_hausman_test(ctx, y_data, x_data, entity_ids, time_periods, feature_names, **kwargs):
    """处理Hausman检验"""
    return CallToolResult(
        content=[TextContent(type="text", text="Hausman检验 - 功能待实现")]
    )


async def handle_panel_unit_root_test(ctx, data, y_data, entity_ids, time_periods, test_type, **kwargs):
    """处理面板单位根检验"""
    return CallToolResult(
        content=[TextContent(type="text", text="面板单位根检验 - 功能待实现")]
    )
# -*- coding: utf-8 -*-
"""
相关性分析处理器
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from mcp.types import CallToolResult, TextContent


async def handle_correlation_analysis(ctx, data: Dict[str, List[float]], 
                                     method: str = "pearson", **kwargs) -> CallToolResult:
    """处理相关性分析"""
    if not data or len(data) < 2:
        raise ValueError("需要至少2个变量")
    
    df = pd.DataFrame(data)
    correlation_matrix = df.corr(method=method)
    
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"{method.title()}相关性分析\n{correlation_matrix.round(4).to_string()}"
            )
        ]
    )
# -*- coding: utf-8 -*-
"""
时间序列分析处理器
"""

import pandas as pd
import numpy as np
from statsmodels.tsa import stattools
from typing import Dict, List, Any, Optional
from mcp.types import CallToolResult, TextContent


async def handle_time_series_analysis(ctx, data: List[float], **kwargs) -> CallToolResult:
    """处理时间序列分析"""
    if not data or len(data) < 5:
        raise ValueError("需要至少5个数据点")
    
    series = pd.Series(data)
    
    # 基本统计量
    basic_stats = {
        "count": len(data),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
        "skewness": float(series.skew()),
        "kurtosis": float(series.kurtosis())
    }
    
    # 平稳性检验
    adf_result = stattools.adfuller(data)
    
    result_text = f"""时间序列分析

基本统计量
- 样本数 = {basic_stats['count']}
- 均值 = {basic_stats['mean']:.4f}
- 标准差 = {basic_stats['std']:.4f}
- 最小值 = {basic_stats['min']:.4f}
- 最大值 = {basic_stats['max']:.4f}
- 中位数 = {basic_stats['median']:.4f}
- 偏度 = {basic_stats['skewness']:.4f}
- 峰度 = {basic_stats['kurtosis']:.4f}

平稳性检验
- ADF统计量 = {adf_result[0]:.4f}
- ADF p值 = {adf_result[1]:.4f}
- 平稳性 = {'平稳' if adf_result[1] < 0.05 else '非平稳'}"""
    
    result_data = {
        "basic_statistics": basic_stats,
        "adf_statistic": float(adf_result[0]),
        "adf_pvalue": float(adf_result[1]),
        "stationary": bool(adf_result[1] < 0.05)
    }
    
    return CallToolResult(
        content=[TextContent(type="text", text=result_text)],
        structuredContent=result_data
    )
# -*- coding: utf-8 -*-
"""
假设检验处理器
"""

import numpy as np
from scipy import stats
from statsmodels.tsa import stattools
from typing import Dict, List, Any, Optional
from mcp.types import CallToolResult, TextContent


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
                text=f"{test_type.upper()}检验结果\n"
                     f"统计量 = {test_result['statistic']:.4f}\n"
                     f"p值 = {test_result['p_value']:.4f}\n"
                     f"{'显著' if test_result['significant'] else '不显著'} (5%显著性水平)\n"
                     f"{ci_text}"
            )
        ],
        structuredContent=test_result
    )
# -*- coding: utf-8 -*-
"""
回归分析处理器
"""

import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Any, Optional
from mcp.types import CallToolResult, TextContent


async def handle_ols_regression(ctx, y_data: List[float], x_data: List[List[float]], 
                                feature_names: Optional[List[str]] = None, **kwargs) -> CallToolResult:
    """处理OLS回归"""
    if not y_data or not x_data:
        raise ValueError("需要提供因变量和自变量数据")
    
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
                text=f"OLS回归结果\n"
                     f"R² = {result_data['rsquared']:.4f}\n"
                     f"调整R² = {result_data['rsquared_adj']:.4f}\n"
                     f"F统计量 = {result_data['f_statistic']:.4f} (p = {result_data['f_pvalue']:.4f})\n"
                     f"AIC = {result_data['aic']:.2f}, BIC = {result_data['bic']:.2f}\n\n"
                     f"系数表\n{model.summary().tables[1]}"
            )
        ],
        structuredContent=result_data
    )


async def handle_multiple_regression(ctx, y_data: List[float], x_data: List[List[float]], 
                                   feature_names: Optional[List[str]] = None, **kwargs) -> CallToolResult:
    """处理多元回归（作为占位符）"""
    return await handle_ols_regression(ctx, y_data, x_data, feature_names, **kwargs)
# -*- coding: utf-8 -*-
"""
描述性统计处理器
"""

import pandas as pd
from typing import Dict, List, Any
from mcp.types import CallToolResult, TextContent


async def handle_descriptive_statistics(ctx, data: Dict[str, List[float]], **kwargs) -> CallToolResult:
    """处理描述性统计"""
    if not data:
        raise ValueError("需要提供数据")
    
    df = pd.DataFrame(data)
    
    # 计算基本统计量
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
                text=f"\n描述性统计\n"
                     f"均值: {result_data['mean']:.4f}\n"
                     f"标准差: {result_data['std']:.4f}\n"
                     f"最小值: {result_data['min']:.4f}\n"
                     f"最大值: {result_data['max']:.4f}\n"
                     f"中位数: {result_data['median']:.4f}\n"
                     f"偏度: {result_data['skewness']:.4f}\n"
                     f"峰度: {result_data['kurtosis']:.4f}\n\n"
                     f"相关性矩阵\n{correlation_matrix.to_string()}"
            )
        ],
        structuredContent=result_data
    )
# -*- coding: utf-8 -*-
"""
工具处理器模块
处理各种计量经济学工具的MCP调用
"""

# 导入各个工具处理器
from .descriptive_statistics_handler import handle_descriptive_statistics
from .regression_handler import handle_ols_regression as handle_old_ols_regression
from .regression_handler import handle_multiple_regression
from .hypothesis_testing_handler import handle_hypothesis_testing
from .time_series_handler import handle_time_series_analysis
from .correlation_analysis_handler import handle_correlation_analysis
from .panel_data_handler import (
    handle_panel_fixed_effects,
    handle_panel_random_effects,
    handle_panel_hausman_test,
    handle_panel_unit_root_test
)
from .time_series_model_handler import (
    handle_var_model,
    handle_vecm_model,
    handle_garch_model,
    handle_state_space_model,
    handle_variance_decomposition
)
from .machine_learning_handler import (
    handle_random_forest,
    handle_gradient_boosting,
    handle_lasso_regression,
    handle_ridge_regression,
    handle_cross_validation,
    handle_feature_importance
)

# 新增的基础与参数估计模块处理器
from .basic_parametric_estimation_handler import (
    handle_ols_regression,
    handle_mle_estimation,
    handle_gmm_estimation
)

__all__ = [
    "handle_descriptive_statistics",
    "handle_old_ols_regression",
    "handle_multiple_regression",
    "handle_hypothesis_testing",
    "handle_time_series_analysis",
    "handle_correlation_analysis",
    "handle_panel_fixed_effects",
    "handle_panel_random_effects",
    "handle_panel_hausman_test",
    "handle_panel_unit_root_test",
    "handle_var_model",
    "handle_vecm_model",
    "handle_garch_model",
    "handle_state_space_model",
    "handle_variance_decomposition",
    "handle_random_forest",
    "handle_gradient_boosting",
    "handle_lasso_regression",
    "handle_ridge_regression",
    "handle_cross_validation",
    "handle_feature_importance",
    # 新增的基础与参数估计模块处理器
    "handle_ols_regression",
    "handle_mle_estimation",
    "handle_gmm_estimation"
]