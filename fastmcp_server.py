"""
AIGroup 计量经济学 MCP 服务器 v2.2 - 组件化架构
自动发现和注册工具组件
"""

import sys
import os
from typing import List, Optional, Union
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

# 设置Windows控制台编码
if sys.platform == "win32":
    try:
        # 尝试设置UTF-8编码
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        # 如果失败，使用ASCII字符
        pass

# 导入所有工具组
from tools.mcp_tool_groups.basic_parametric_tools import BasicParametricTools
from tools.mcp_tool_groups.time_series_tools import TimeSeriesTools
from tools.mcp_tool_groups.model_specification_tools import ModelSpecificationTools

# 创建 FastMCP 服务器实例
mcp = FastMCP("aigroup-econ-mcp")

# 注册基础参数估计工具
@mcp.tool()
async def basic_parametric_estimation_ols(
    y_data: Optional[List[float]] = None,
    x_data: Optional[Union[List[float], List[List[float]]]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    confidence_level: float = 0.95,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """OLS Regression Analysis"""
    return await BasicParametricTools.ols_tool(y_data, x_data, file_path, feature_names, constant, confidence_level, output_format, save_path, ctx)

@mcp.tool()
async def basic_parametric_estimation_mle(
    data: Optional[List[float]] = None,
    file_path: Optional[str] = None,
    distribution: str = "normal",
    initial_params: Optional[List[float]] = None,
    confidence_level: float = 0.95,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Maximum Likelihood Estimation"""
    return await BasicParametricTools.mle_tool(data, file_path, distribution, initial_params, confidence_level, output_format, save_path, ctx)

@mcp.tool()
async def basic_parametric_estimation_gmm(
    y_data: Optional[List[float]] = None,
    x_data: Optional[Union[List[float], List[List[float]]]] = None,
    file_path: Optional[str] = None,
    instruments: Optional[Union[List[float], List[List[float]]]] = None,
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    confidence_level: float = 0.95,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Generalized Method of Moments"""
    return await BasicParametricTools.gmm_tool(y_data, x_data, file_path, instruments, feature_names, constant, confidence_level, output_format, save_path, ctx)

# 注册时间序列工具
@mcp.tool()
async def time_series_arima_model(
    data: Optional[List[float]] = None,
    file_path: Optional[str] = None,
    order: tuple = (1, 1, 1),
    forecast_steps: int = 1,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """ARIMA Time Series Model"""
    return await TimeSeriesTools.arima_tool(data, file_path, order, forecast_steps, output_format, save_path, ctx)

@mcp.tool()
async def time_series_exponential_smoothing(
    data: Optional[List[float]] = None,
    file_path: Optional[str] = None,
    trend: bool = True,
    seasonal: bool = False,
   seasonal_periods: Optional[int] = None,
    forecast_steps: int = 1,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Exponential Smoothing Model"""
    return await TimeSeriesTools.exp_smoothing_tool(data, file_path, trend, seasonal, seasonal_periods, forecast_steps, output_format, save_path, ctx)

@mcp.tool()
async def time_series_garch_model(
    data: Optional[List[float]] = None,
    file_path: Optional[str] = None,
    order: tuple = (1, 1),
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """GARCH Volatility Model"""
    return await TimeSeriesTools.garch_tool(data, file_path, order, output_format, save_path, ctx)

@mcp.tool()
async def time_series_unit_root_tests(
    data: Optional[List[float]] = None,
    file_path: Optional[str] = None,
    test_type: str = "adf",
    max_lags: Optional[int] = None,
    regression_type: str = "c",
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Unit Root Tests"""
    return await TimeSeriesTools.unit_root_tool(data, file_path, test_type, max_lags, regression_type, output_format, save_path, ctx)

@mcp.tool()
async def time_series_var_svar_model(
    data: Optional[List[List[float]]] = None,
    file_path: Optional[str] = None,
    model_type: str = "var",
    lags: int = 1,
    variables: Optional[List[str]] = None,
    a_matrix: Optional[List[List[float]]] = None,
    b_matrix: Optional[List[List[float]]] = None,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """VAR/SVAR Model"""
    return await TimeSeriesTools.var_svar_tool(data, file_path, model_type, lags, variables, a_matrix, b_matrix, output_format, save_path, ctx)

@mcp.tool()
async def time_series_cointegration_analysis(
    data: Optional[List[List[float]]] = None,
    file_path: Optional[str] = None,
    analysis_type: str = "johansen",
    variables: Optional[List[str]] = None,
    coint_rank: int = 1,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Cointegration Analysis"""
    return await TimeSeriesTools.cointegration_tool(data, file_path, analysis_type, variables, coint_rank, output_format, save_path, ctx)

@mcp.tool()
async def panel_data_dynamic_model(
    y_data: Optional[List[float]] = None,
    x_data: Optional[List[List[float]]] = None,
    entity_ids: Optional[List[int]] = None,
    time_periods: Optional[List[int]] = None,
    file_path: Optional[str] = None,
    model_type: str = "diff_gmm",
    lags: int = 1,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Dynamic Panel Data Model"""
    return await TimeSeriesTools.dynamic_panel_tool(y_data, x_data, entity_ids, time_periods, file_path, model_type, lags, output_format, save_path, ctx)

@mcp.tool()
async def panel_data_diagnostics(
    test_type: str = "hausman",
    fe_coefficients: Optional[List[float]] = None,
    re_coefficients: Optional[List[float]] = None,
    fe_covariance: Optional[List[List[float]]] = None,
    re_covariance: Optional[List[List[float]]] = None,
    pooled_ssrs: Optional[float] = None,
    fixed_ssrs: Optional[float] = None,
    random_ssrs: Optional[float] = None,
    n_individuals: Optional[int] = None,
    n_params: Optional[int] = None,
    n_obs: Optional[int] = None,
    n_periods: Optional[int] = None,
    residuals: Optional[List[List[float]]] = None,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Panel Data Diagnostic Tests (Hausman, Pooling F, LM, Within Correlation)"""
    return await TimeSeriesTools.panel_diagnostics_tool(test_type, fe_coefficients, re_coefficients, fe_covariance, re_covariance, pooled_ssrs, fixed_ssrs, random_ssrs, n_individuals, n_params, n_obs, n_periods, residuals, output_format, save_path, ctx)

@mcp.tool()
async def panel_var_model(
    data: Optional[List[List[float]]] = None,
    entity_ids: Optional[List[int]] = None,
    time_periods: Optional[List[int]] = None,
    file_path: Optional[str] = None,
    lags: int = 1,
    variables: Optional[List[str]] = None,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Panel Vector Autoregression Model"""
    return await TimeSeriesTools.panel_var_tool(data, entity_ids, time_periods, file_path, lags, variables, output_format, save_path, ctx)

@mcp.tool()
async def structural_break_tests(
    data: Optional[List[float]] = None,
    file_path: Optional[str] = None,
    test_type: str = "chow",
    break_point: Optional[int] = None,
    max_breaks: int = 5,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Structural Break Tests (Chow, Quandt-Andrews, Bai-Perron)"""
    return await TimeSeriesTools.structural_break_tool(data, file_path, test_type, break_point, max_breaks, output_format, save_path, ctx)

@mcp.tool()
async def time_varying_parameter_models(
    y_data: Optional[List[float]] = None,
    x_data: Optional[List[List[float]]] = None,
    file_path: Optional[str] = None,
    model_type: str = "tar",
    threshold_variable: Optional[List[float]] = None,
    n_regimes: int = 2,
    star_type: str = "logistic",
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Time-Varying Parameter Models (TAR, STAR, Markov Switching)"""
    return await TimeSeriesTools.time_varying_parameter_tool(y_data, x_data, file_path, model_type, threshold_variable, n_regimes, star_type, output_format, save_path, ctx)

# 注册模型规范、诊断和稳健推断工具
@mcp.tool()
async def model_diagnostic_tests(
    y_data: Optional[List[float]] = None,
    x_data: Optional[Union[List[float], List[List[float]]]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Model Diagnostic Tests (Heteroskedasticity, Autocorrelation, Normality, VIF)"""
    return await ModelSpecificationTools.diagnostic_tests_tool(y_data, x_data, file_path, feature_names, constant, output_format, save_path, ctx)

@mcp.tool()
async def generalized_least_squares(
    y_data: Optional[List[float]] = None,
    x_data: Optional[Union[List[float], List[List[float]]]] = None,
    file_path: Optional[str] = None,
    sigma: Optional[List[List[float]]] = None,
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    confidence_level: float = 0.95,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Generalized Least Squares (GLS) Regression"""
    return await ModelSpecificationTools.gls_tool(y_data, x_data, file_path, sigma, feature_names, constant, confidence_level, output_format, save_path, ctx)

@mcp.tool()
async def weighted_least_squares(
    y_data: Optional[List[float]] = None,
    x_data: Optional[Union[List[float], List[List[float]]]] = None,
    file_path: Optional[str] = None,
    weights: Optional[List[float]] = None,
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    confidence_level: float = 0.95,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Weighted Least Squares (WLS) Regression"""
    return await ModelSpecificationTools.wls_tool(y_data, x_data, file_path, weights, feature_names, constant, confidence_level, output_format, save_path, ctx)

@mcp.tool()
async def robust_errors_regression(
    y_data: Optional[List[float]] = None,
    x_data: Optional[Union[List[float], List[List[float]]]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    confidence_level: float = 0.95,
    cov_type: str = "HC1",
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Robust Standard Errors Regression (Heteroskedasticity-Robust)"""
    return await ModelSpecificationTools.robust_errors_tool(y_data, x_data, file_path, feature_names, constant, confidence_level, cov_type, output_format, save_path, ctx)

@mcp.tool()
async def model_selection_criteria(
    y_data: Optional[List[float]] = None,
    x_data: Optional[Union[List[float], List[List[float]]]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    cv_folds: Optional[int] = None,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Model Selection Criteria (AIC, BIC, HQIC, Cross-Validation)"""
    return await ModelSpecificationTools.model_selection_tool(y_data, x_data, file_path, feature_names, constant, cv_folds, output_format, save_path, ctx)

@mcp.tool()
async def regularized_regression(
    y_data: Optional[List[float]] = None,
    x_data: Optional[Union[List[float], List[List[float]]]] = None,
    file_path: Optional[str] = None,
    method: str = "ridge",
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    feature_names: Optional[List[str]] = None,
    fit_intercept: bool = True,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Regularized Regression (Ridge, LASSO, Elastic Net)"""
    return await ModelSpecificationTools.regularization_tool(y_data, x_data, file_path, method, alpha, l1_ratio, feature_names, fit_intercept, output_format, save_path, ctx)

@mcp.tool()
async def simultaneous_equations_model(
    y_data: Optional[List[List[float]]] = None,
    x_data: Optional[List[List[List[float]]]] = None,
    file_path: Optional[str] = None,
    instruments: Optional[List[List[float]]] = None,
    equation_names: Optional[List[str]] = None,
    endogenous_vars: Optional[List[str]] = None,
    exogenous_vars: Optional[List[str]] = None,
    constant: bool = True,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """Simultaneous Equations Model (2SLS)"""
    return await ModelSpecificationTools.simultaneous_equations_tool(y_data, x_data, file_path, instruments, equation_names, endogenous_vars, exogenous_vars, constant, output_format, save_path, ctx)


@mcp.resource("guide://econometrics")
def get_econometrics_guide() -> str:
    """Get complete econometrics tools guide"""
    try:
        with open("resources/MCP_MASTER_GUIDE.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "完整使用指南文件未找到，请检查 resources/MCP_MASTER_GUIDE.md 文件是否存在。"


def main():
    """Start FastMCP server"""
    print("=" * 60)
    print("AIGroup Econometrics MCP Server v2.2.0")
    print("=" * 60)
    print("\n架构: 组件化")
    print("\n已注册工具组:")
    print(f"  - {BasicParametricTools.name} ({len(BasicParametricTools.get_tools())} tools)")
    print(f"  - {TimeSeriesTools.name} ({len(TimeSeriesTools.get_tools())} tools)")
    print(f"  - {ModelSpecificationTools.name} ({len(ModelSpecificationTools.get_tools())} tools)")
    
    total_tools = len(BasicParametricTools.get_tools()) + len(TimeSeriesTools.get_tools()) + len(ModelSpecificationTools.get_tools())
    print(f"\n总工具数: {total_tools}")
    print("\n支持格式:")
    print("  输入: txt/json/csv/excel (.xlsx, .xls)")
    print("  输出: json/markdown/txt")
    
    print("\n优势:")
    print("  * 组件化设计")
    print("  * 易于扩展")
    print("  * DRY原则")
    
    print("\n启动服务器...")
    print("=" * 60)
    
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()