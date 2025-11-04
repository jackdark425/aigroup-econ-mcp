"""
AIGroup 计量经济学 MCP 服务器 - 优化版
使用组件化架构，代码量减少80%，同时自动支持文件输入
"""

from typing import Dict, Any, Optional, List, Annotated
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from mcp.types import CallToolResult, TextContent

# 导入工具处理器
from .tools.tool_handlers import (
    handle_descriptive_statistics,
    handle_ols_regression,
    handle_hypothesis_testing,
    handle_time_series_analysis,
    handle_correlation_analysis,
    handle_panel_fixed_effects,
    handle_panel_random_effects,
    handle_panel_hausman_test,
    handle_panel_unit_root_test,
    handle_var_model,
    handle_vecm_model,
    handle_garch_model,
    handle_state_space_model,
    handle_variance_decomposition,
    handle_random_forest,
    handle_gradient_boosting,
    handle_lasso_regression,
    handle_ridge_regression,
    handle_cross_validation,
    handle_feature_importance
)

# 导入装饰器和工具描述
from .tools.base import with_file_support_decorator as econometric_tool
from .tools.tool_descriptions import (
    get_tool_description,
    get_field_description,
    DESCRIPTIVE_STATISTICS,
    OLS_REGRESSION,
    HYPOTHESIS_TESTING,
    TIME_SERIES_ANALYSIS,
    CORRELATION_ANALYSIS,
    PANEL_FIXED_EFFECTS,
    PANEL_RANDOM_EFFECTS,
    PANEL_HAUSMAN_TEST,
    PANEL_UNIT_ROOT_TEST,
    VAR_MODEL_ANALYSIS,
    VECM_MODEL_ANALYSIS,
    GARCH_MODEL_ANALYSIS,
    STATE_SPACE_MODEL_ANALYSIS,
    VARIANCE_DECOMPOSITION_ANALYSIS,
    RANDOM_FOREST_REGRESSION_ANALYSIS,
    GRADIENT_BOOSTING_REGRESSION_ANALYSIS,
    LASSO_REGRESSION_ANALYSIS,
    RIDGE_REGRESSION_ANALYSIS,
    CROSS_VALIDATION_ANALYSIS,
    FEATURE_IMPORTANCE_ANALYSIS_TOOL
)


# 应用上下文
@dataclass
class AppContext:
    """应用上下文，包含共享资源"""
    config: Dict[str, Any]
    version: str = "1.0.0"


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """服务器生命周期管理"""
    config = {
        "max_sample_size": 10000,
        "default_significance_level": 0.05,
        "supported_tests": ["t_test", "f_test", "chi_square", "adf"],
        "data_types": ["cross_section", "time_series", "panel"]
    }
    try:
        yield AppContext(config=config, version="1.0.0")
    finally:
        pass


# 创建MCP服务器实例
mcp = FastMCP(
    name="aigroup-econ-mcp",
    instructions="Econometrics MCP Server - Provides data analysis with automatic file input support",
    lifespan=lifespan
)


# ============================================================================
# 基础统计工具 (5个) - 自动支持文件输入
# ============================================================================

@mcp.tool()
@econometric_tool('multi_var_dict')
async def descriptive_statistics(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=DESCRIPTIVE_STATISTICS.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=DESCRIPTIVE_STATISTICS.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=DESCRIPTIVE_STATISTICS.get_field_description("file_format"))] = "auto",
    data: Annotated[Optional[Dict[str, List[float]]], Field(default=None, description=DESCRIPTIVE_STATISTICS.get_field_description("data"))] = None
) -> CallToolResult:
    """计算描述性统计量"""
    return await handle_descriptive_statistics(ctx, data=data)


@mcp.tool()
@econometric_tool('regression')
async def ols_regression(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=OLS_REGRESSION.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=OLS_REGRESSION.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=OLS_REGRESSION.get_field_description("file_format"))] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None, description=OLS_REGRESSION.get_field_description("y_data"))] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None, description=OLS_REGRESSION.get_field_description("x_data"))] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None, description=OLS_REGRESSION.get_field_description("feature_names"))] = None
) -> CallToolResult:
    """OLS回归分析"""
    return await handle_ols_regression(ctx, y_data=y_data, x_data=x_data, feature_names=feature_names)


@mcp.tool()
@econometric_tool('single_var')
async def hypothesis_testing(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=HYPOTHESIS_TESTING.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=HYPOTHESIS_TESTING.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=HYPOTHESIS_TESTING.get_field_description("file_format"))] = "auto",
    data: Annotated[Optional[List[float]], Field(default=None, description=HYPOTHESIS_TESTING.get_field_description("data"))] = None,
    data2: Annotated[Optional[List[float]], Field(default=None, description=HYPOTHESIS_TESTING.get_field_description("data2"))] = None,
    test_type: Annotated[str, Field(default="t_test", description=HYPOTHESIS_TESTING.get_field_description("test_type"))] = "t_test"
) -> CallToolResult:
    """假设检验 - 支持文件或直接数据输入"""
    return await handle_hypothesis_testing(ctx, data1=data, data2=data2, test_type=test_type)


@mcp.tool()
@econometric_tool('single_var')
async def time_series_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=TIME_SERIES_ANALYSIS.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=TIME_SERIES_ANALYSIS.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=TIME_SERIES_ANALYSIS.get_field_description("file_format"))] = "auto",
    data: Annotated[Optional[List[float]], Field(default=None, description=TIME_SERIES_ANALYSIS.get_field_description("data"))] = None
) -> CallToolResult:
    """时间序列分析 - 支持文件或直接数据输入"""
    return await handle_time_series_analysis(ctx, data=data)


@mcp.tool()
@econometric_tool('multi_var_dict')
async def correlation_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=CORRELATION_ANALYSIS.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=CORRELATION_ANALYSIS.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=CORRELATION_ANALYSIS.get_field_description("file_format"))] = "auto",
    data: Annotated[Optional[Dict[str, List[float]]], Field(default=None, description=CORRELATION_ANALYSIS.get_field_description("data"))] = None,
    method: Annotated[str, Field(default="pearson", description=CORRELATION_ANALYSIS.get_field_description("method"))] = "pearson"
) -> CallToolResult:
    """相关性分析 - 支持文件或直接数据输入"""
    return await handle_correlation_analysis(ctx, data=data, method=method)


# ============================================================================
# 面板数据工具 (4个) - 自动支持文件输入
# ============================================================================

@mcp.tool()
@econometric_tool('panel')
async def panel_fixed_effects(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=PANEL_FIXED_EFFECTS.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=PANEL_FIXED_EFFECTS.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=PANEL_FIXED_EFFECTS.get_field_description("file_format"))] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None, description=PANEL_FIXED_EFFECTS.get_field_description("y_data"))] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None, description=PANEL_FIXED_EFFECTS.get_field_description("x_data"))] = None,
    entity_ids: Annotated[Optional[List[str]], Field(default=None, description=PANEL_FIXED_EFFECTS.get_field_description("entity_ids"))] = None,
    time_periods: Annotated[Optional[List[str]], Field(default=None, description=PANEL_FIXED_EFFECTS.get_field_description("time_periods"))] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None, description=PANEL_FIXED_EFFECTS.get_field_description("feature_names"))] = None,
    entity_effects: Annotated[bool, Field(default=True, description=PANEL_FIXED_EFFECTS.get_field_description("entity_effects"))] = True,
    time_effects: Annotated[bool, Field(default=False, description=PANEL_FIXED_EFFECTS.get_field_description("time_effects"))] = False
) -> CallToolResult:
    """固定效应模型 - 支持文件输入"""
    return await handle_panel_fixed_effects(ctx, y_data, x_data, entity_ids, time_periods,
                                           feature_names, entity_effects, time_effects)


@mcp.tool()
@econometric_tool('panel')
async def panel_random_effects(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=PANEL_RANDOM_EFFECTS.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=PANEL_RANDOM_EFFECTS.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=PANEL_RANDOM_EFFECTS.get_field_description("file_format"))] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None, description=PANEL_RANDOM_EFFECTS.get_field_description("y_data"))] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None, description=PANEL_RANDOM_EFFECTS.get_field_description("x_data"))] = None,
    entity_ids: Annotated[Optional[List[str]], Field(default=None, description=PANEL_RANDOM_EFFECTS.get_field_description("entity_ids"))] = None,
    time_periods: Annotated[Optional[List[str]], Field(default=None, description=PANEL_RANDOM_EFFECTS.get_field_description("time_periods"))] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None, description=PANEL_RANDOM_EFFECTS.get_field_description("feature_names"))] = None,
    entity_effects: Annotated[bool, Field(default=True, description=PANEL_RANDOM_EFFECTS.get_field_description("entity_effects"))] = True,
    time_effects: Annotated[bool, Field(default=False, description=PANEL_RANDOM_EFFECTS.get_field_description("time_effects"))] = False
) -> CallToolResult:
    """随机效应模型 - 支持文件输入"""
    return await handle_panel_random_effects(ctx, y_data, x_data, entity_ids, time_periods,
                                            feature_names, entity_effects, time_effects)


@mcp.tool()
@econometric_tool('panel')
async def panel_hausman_test(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=PANEL_HAUSMAN_TEST.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=PANEL_HAUSMAN_TEST.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=PANEL_HAUSMAN_TEST.get_field_description("file_format"))] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None, description=PANEL_HAUSMAN_TEST.get_field_description("y_data"))] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None, description=PANEL_HAUSMAN_TEST.get_field_description("x_data"))] = None,
    entity_ids: Annotated[Optional[List[str]], Field(default=None, description=PANEL_HAUSMAN_TEST.get_field_description("entity_ids"))] = None,
    time_periods: Annotated[Optional[List[str]], Field(default=None, description=PANEL_HAUSMAN_TEST.get_field_description("time_periods"))] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None, description=PANEL_HAUSMAN_TEST.get_field_description("feature_names"))] = None
) -> CallToolResult:
    """Hausman检验 - 支持文件输入"""
    return await handle_panel_hausman_test(ctx, y_data, x_data, entity_ids, time_periods, feature_names)


@mcp.tool()
@econometric_tool('panel')  # 保持panel类型以获取entity_ids和time_periods
async def panel_unit_root_test(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=PANEL_UNIT_ROOT_TEST.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=PANEL_UNIT_ROOT_TEST.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=PANEL_UNIT_ROOT_TEST.get_field_description("file_format"))] = "auto",
    data: Annotated[Optional[List[float]], Field(default=None, description=PANEL_UNIT_ROOT_TEST.get_field_description("data"))] = None,
    y_data: Annotated[Optional[List[float]], Field(default=None, description=PANEL_UNIT_ROOT_TEST.get_field_description("y_data"))] = None,  # 从panel转换来的
    x_data: Annotated[Optional[List[List[float]]], Field(default=None, description=PANEL_UNIT_ROOT_TEST.get_field_description("x_data"))] = None,  # 从panel转换来的，忽略
    entity_ids: Annotated[Optional[List[str]], Field(default=None, description=PANEL_UNIT_ROOT_TEST.get_field_description("entity_ids"))] = None,
    time_periods: Annotated[Optional[List[str]], Field(default=None, description=PANEL_UNIT_ROOT_TEST.get_field_description("time_periods"))] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None, description=PANEL_UNIT_ROOT_TEST.get_field_description("feature_names"))] = None,  # 从panel转换来的，忽略
    test_type: Annotated[str, Field(default="levinlin", description=PANEL_UNIT_ROOT_TEST.get_field_description("test_type"))] = "levinlin"
) -> CallToolResult:
    """面板单位根检验 - 支持文件输入"""
    # 传递所有参数给handler
    return await handle_panel_unit_root_test(
        ctx,
        data=data,
        y_data=y_data,
        entity_ids=entity_ids,
        time_periods=time_periods,
        test_type=test_type
    )


# ============================================================================
# 高级时间序列工具 (5个) - 自动支持文件输入
# ============================================================================

@mcp.tool()
@econometric_tool('time_series')
async def var_model_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=VAR_MODEL_ANALYSIS.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=VAR_MODEL_ANALYSIS.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=VAR_MODEL_ANALYSIS.get_field_description("file_format"))] = "auto",
    data: Annotated[Optional[Dict[str, List[float]]], Field(default=None, description=VAR_MODEL_ANALYSIS.get_field_description("data"))] = None,
    max_lags: Annotated[int, Field(default=5, description=VAR_MODEL_ANALYSIS.get_field_description("max_lags"))] = 5,
    ic: Annotated[str, Field(default="aic", description=VAR_MODEL_ANALYSIS.get_field_description("ic"))] = "aic"
) -> CallToolResult:
    """VAR模型分析 - 支持文件输入"""
    return await handle_var_model(ctx, data, max_lags, ic)


@mcp.tool()
@econometric_tool('time_series')
async def vecm_model_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=VECM_MODEL_ANALYSIS.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=VECM_MODEL_ANALYSIS.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=VECM_MODEL_ANALYSIS.get_field_description("file_format"))] = "auto",
    data: Annotated[Optional[Dict[str, List[float]]], Field(default=None, description=VECM_MODEL_ANALYSIS.get_field_description("data"))] = None,
    coint_rank: Annotated[int, Field(default=1, description=VECM_MODEL_ANALYSIS.get_field_description("coint_rank"))] = 1,
    deterministic: Annotated[str, Field(default="co", description=VECM_MODEL_ANALYSIS.get_field_description("deterministic"))] = "co",
    max_lags: Annotated[int, Field(default=5, description=VECM_MODEL_ANALYSIS.get_field_description("max_lags"))] = 5
) -> CallToolResult:
    """VECM模型分析 - 支持文件输入"""
    return await handle_vecm_model(ctx, data, coint_rank, deterministic, max_lags)


@mcp.tool()
@econometric_tool('single_var')
async def garch_model_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=GARCH_MODEL_ANALYSIS.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=GARCH_MODEL_ANALYSIS.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=GARCH_MODEL_ANALYSIS.get_field_description("file_format"))] = "auto",
    data: Annotated[Optional[List[float]], Field(default=None, description=GARCH_MODEL_ANALYSIS.get_field_description("data"))] = None,
    order: Annotated[tuple, Field(default=(1, 1), description=GARCH_MODEL_ANALYSIS.get_field_description("order"))] = (1, 1),
    dist: Annotated[str, Field(default="normal", description=GARCH_MODEL_ANALYSIS.get_field_description("dist"))] = "normal"
) -> CallToolResult:
    """GARCH模型分析 - 支持文件输入"""
    return await handle_garch_model(ctx, data, order, dist)


@mcp.tool()
@econometric_tool('single_var')
async def state_space_model_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=STATE_SPACE_MODEL_ANALYSIS.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=STATE_SPACE_MODEL_ANALYSIS.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=STATE_SPACE_MODEL_ANALYSIS.get_field_description("file_format"))] = "auto",
    data: Annotated[Optional[List[float]], Field(default=None, description=STATE_SPACE_MODEL_ANALYSIS.get_field_description("data"))] = None,
    state_dim: Annotated[int, Field(default=1, description=STATE_SPACE_MODEL_ANALYSIS.get_field_description("state_dim"))] = 1,
    observation_dim: Annotated[int, Field(default=1, description=STATE_SPACE_MODEL_ANALYSIS.get_field_description("observation_dim"))] = 1,
    trend: Annotated[bool, Field(default=True, description=STATE_SPACE_MODEL_ANALYSIS.get_field_description("trend"))] = True,
    seasonal: Annotated[bool, Field(default=False, description=STATE_SPACE_MODEL_ANALYSIS.get_field_description("seasonal"))] = False,
    period: Annotated[int, Field(default=12, description=STATE_SPACE_MODEL_ANALYSIS.get_field_description("period"))] = 12
) -> CallToolResult:
    """状态空间模型分析 - 支持文件输入"""
    return await handle_state_space_model(ctx, data, state_dim, observation_dim, trend, seasonal, period)


@mcp.tool()
@econometric_tool('time_series')
async def variance_decomposition_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=VARIANCE_DECOMPOSITION_ANALYSIS.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=VARIANCE_DECOMPOSITION_ANALYSIS.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=VARIANCE_DECOMPOSITION_ANALYSIS.get_field_description("file_format"))] = "auto",
    data: Annotated[Optional[Dict[str, List[float]]], Field(default=None, description=VARIANCE_DECOMPOSITION_ANALYSIS.get_field_description("data"))] = None,
    periods: Annotated[int, Field(default=10, description=VARIANCE_DECOMPOSITION_ANALYSIS.get_field_description("periods"))] = 10,
    max_lags: Annotated[int, Field(default=5, description=VARIANCE_DECOMPOSITION_ANALYSIS.get_field_description("max_lags"))] = 5
) -> CallToolResult:
    """方差分解分析 - 支持文件输入"""
    return await handle_variance_decomposition(ctx, data, periods, max_lags)


# ============================================================================
# 机器学习工具 (6个) - 自动支持文件输入
# ============================================================================

@mcp.tool()
@econometric_tool('regression')
async def random_forest_regression_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=RANDOM_FOREST_REGRESSION_ANALYSIS.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=RANDOM_FOREST_REGRESSION_ANALYSIS.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=RANDOM_FOREST_REGRESSION_ANALYSIS.get_field_description("file_format"))] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None, description=RANDOM_FOREST_REGRESSION_ANALYSIS.get_field_description("y_data"))] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None, description=RANDOM_FOREST_REGRESSION_ANALYSIS.get_field_description("x_data"))] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None, description=RANDOM_FOREST_REGRESSION_ANALYSIS.get_field_description("feature_names"))] = None,
    n_estimators: Annotated[int, Field(default=100, description=RANDOM_FOREST_REGRESSION_ANALYSIS.get_field_description("n_estimators"))] = 100,
    max_depth: Annotated[Optional[int], Field(default=None, description=RANDOM_FOREST_REGRESSION_ANALYSIS.get_field_description("max_depth"))] = None
) -> CallToolResult:
    """随机森林回归 - 支持文件输入"""
    return await handle_random_forest(ctx, y_data, x_data, feature_names, n_estimators, max_depth)


@mcp.tool()
@econometric_tool('regression')
async def gradient_boosting_regression_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=GRADIENT_BOOSTING_REGRESSION_ANALYSIS.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=GRADIENT_BOOSTING_REGRESSION_ANALYSIS.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=GRADIENT_BOOSTING_REGRESSION_ANALYSIS.get_field_description("file_format"))] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None, description=GRADIENT_BOOSTING_REGRESSION_ANALYSIS.get_field_description("y_data"))] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None, description=GRADIENT_BOOSTING_REGRESSION_ANALYSIS.get_field_description("x_data"))] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None, description=GRADIENT_BOOSTING_REGRESSION_ANALYSIS.get_field_description("feature_names"))] = None,
    n_estimators: Annotated[int, Field(default=100, description=GRADIENT_BOOSTING_REGRESSION_ANALYSIS.get_field_description("n_estimators"))] = 100,
    learning_rate: Annotated[float, Field(default=0.1, description=GRADIENT_BOOSTING_REGRESSION_ANALYSIS.get_field_description("learning_rate"))] = 0.1,
    max_depth: Annotated[int, Field(default=3, description=GRADIENT_BOOSTING_REGRESSION_ANALYSIS.get_field_description("max_depth"))] = 3
) -> CallToolResult:
    """梯度提升树回归 - 支持文件输入"""
    return await handle_gradient_boosting(ctx, y_data, x_data, feature_names, n_estimators, learning_rate, max_depth)


@mcp.tool()
@econometric_tool('regression')
async def lasso_regression_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=LASSO_REGRESSION_ANALYSIS.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=LASSO_REGRESSION_ANALYSIS.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=LASSO_REGRESSION_ANALYSIS.get_field_description("file_format"))] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None, description=LASSO_REGRESSION_ANALYSIS.get_field_description("y_data"))] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None, description=LASSO_REGRESSION_ANALYSIS.get_field_description("x_data"))] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None, description=LASSO_REGRESSION_ANALYSIS.get_field_description("feature_names"))] = None,
    alpha: Annotated[float, Field(default=1.0, description=LASSO_REGRESSION_ANALYSIS.get_field_description("alpha"))] = 1.0
) -> CallToolResult:
    """Lasso回归 - 支持文件输入"""
    return await handle_lasso_regression(ctx, y_data, x_data, feature_names, alpha)


@mcp.tool()
@econometric_tool('regression')
async def ridge_regression_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=RIDGE_REGRESSION_ANALYSIS.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=RIDGE_REGRESSION_ANALYSIS.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=RIDGE_REGRESSION_ANALYSIS.get_field_description("file_format"))] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None, description=RIDGE_REGRESSION_ANALYSIS.get_field_description("y_data"))] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None, description=RIDGE_REGRESSION_ANALYSIS.get_field_description("x_data"))] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None, description=RIDGE_REGRESSION_ANALYSIS.get_field_description("feature_names"))] = None,
    alpha: Annotated[float, Field(default=1.0, description=RIDGE_REGRESSION_ANALYSIS.get_field_description("alpha"))] = 1.0
) -> CallToolResult:
    """Ridge回归 - 支持文件输入"""
    return await handle_ridge_regression(ctx, y_data, x_data, feature_names, alpha)


@mcp.tool()
@econometric_tool('regression')
async def cross_validation_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=CROSS_VALIDATION_ANALYSIS.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=CROSS_VALIDATION_ANALYSIS.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=CROSS_VALIDATION_ANALYSIS.get_field_description("file_format"))] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None, description=CROSS_VALIDATION_ANALYSIS.get_field_description("y_data"))] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None, description=CROSS_VALIDATION_ANALYSIS.get_field_description("x_data"))] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None, description=CROSS_VALIDATION_ANALYSIS.get_field_description("feature_names"))] = None,
    model_type: Annotated[str, Field(default="random_forest", description=CROSS_VALIDATION_ANALYSIS.get_field_description("model_type"))] = "random_forest",
    cv_folds: Annotated[int, Field(default=5, description=CROSS_VALIDATION_ANALYSIS.get_field_description("cv_folds"))] = 5,
    scoring: Annotated[str, Field(default="r2", description=CROSS_VALIDATION_ANALYSIS.get_field_description("scoring"))] = "r2"
) -> CallToolResult:
    """交叉验证 - 支持文件输入"""
    return await handle_cross_validation(ctx, y_data, x_data, model_type, cv_folds, scoring)


@mcp.tool()
@econometric_tool('regression')
async def feature_importance_analysis_tool(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=FEATURE_IMPORTANCE_ANALYSIS_TOOL.get_field_description("file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=FEATURE_IMPORTANCE_ANALYSIS_TOOL.get_field_description("file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=FEATURE_IMPORTANCE_ANALYSIS_TOOL.get_field_description("file_format"))] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None, description=FEATURE_IMPORTANCE_ANALYSIS_TOOL.get_field_description("y_data"))] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None, description=FEATURE_IMPORTANCE_ANALYSIS_TOOL.get_field_description("x_data"))] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None, description=FEATURE_IMPORTANCE_ANALYSIS_TOOL.get_field_description("feature_names"))] = None,
    method: Annotated[str, Field(default="random_forest", description=FEATURE_IMPORTANCE_ANALYSIS_TOOL.get_field_description("method"))] = "random_forest",
    top_k: Annotated[int, Field(default=5, description=FEATURE_IMPORTANCE_ANALYSIS_TOOL.get_field_description("top_k"))] = 5
) -> CallToolResult:
    """特征重要性分析 - 支持文件输入"""
    return await handle_feature_importance(ctx, y_data, x_data, feature_names, method, top_k)


def create_mcp_server() -> FastMCP:
    """创建并返回MCP服务器实例"""
    return mcp
def main():
    """启动MCP服务器"""
    print("启动AIGroup计量经济学MCP服务器...")
    print("传输协议: stdio")
    print("等待MCP客户端连接...")
    server = create_mcp_server()
    server.run(transport='stdio')


if __name__ == "__main__":
    main()