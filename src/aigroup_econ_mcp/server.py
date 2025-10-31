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

# 导入装饰器
from .tools.base import with_file_support_decorator as econometric_tool


# 应用上下文
@dataclass
class AppContext:
    """应用上下文，包含共享资源"""
    config: Dict[str, Any]
    version: str = "0.2.0"


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
        yield AppContext(config=config, version="0.2.0")
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
    file_path: Annotated[Optional[str], Field(default=None, description="CSV/JSON文件路径")] = None,
    file_content: Annotated[Optional[str], Field(default=None, description="CSV/JSON文件内容")] = None,
    file_format: Annotated[str, Field(default="auto", description="文件格式(csv/json/auto)")] = "auto",
    data: Annotated[Optional[Dict[str, List[float]]], Field(default=None, description="数据字典(直接数据输入)")] = None
) -> CallToolResult:
    """
    计算描述性统计量
    
    支持三种输入方式（按优先级）：
    1. file_path: 文件路径 (如 "data.csv")
    2. file_content: 文件内容字符串
    3. data: 直接传入数据字典
    """
    return await handle_descriptive_statistics(ctx, data=data)


@mcp.tool()
@econometric_tool('regression')
async def ols_regression(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description="CSV/JSON文件路径")] = None,
    file_content: Annotated[Optional[str], Field(default=None, description="CSV/JSON文件内容")] = None,
    file_format: Annotated[str, Field(default="auto", description="文件格式")] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None, description="因变量(直接输入)")] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None, description="自变量(直接输入)")] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None, description="特征名称")] = None
) -> CallToolResult:
    """
    OLS回归分析
    
    支持文件输入或直接数据输入。文件格式示例：
    CSV: 最后一列为因变量，其余列为自变量
    """
    return await handle_ols_regression(ctx, y_data=y_data, x_data=x_data, feature_names=feature_names)


@mcp.tool()
@econometric_tool('single_var')
async def hypothesis_testing(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description="文件路径")] = None,
    file_content: Annotated[Optional[str], Field(default=None, description="文件内容")] = None,
    file_format: Annotated[str, Field(default="auto", description="文件格式")] = "auto",
    data: Annotated[Optional[List[float]], Field(default=None, description="第一组数据")] = None,
    data2: Annotated[Optional[List[float]], Field(default=None, description="第二组数据")] = None,
    test_type: Annotated[str, Field(default="t_test", description="检验类型(t_test/adf)")] = "t_test"
) -> CallToolResult:
    """假设检验 - 支持文件或直接数据输入"""
    return await handle_hypothesis_testing(ctx, data1=data, data2=data2, test_type=test_type)


@mcp.tool()
@econometric_tool('single_var')
async def time_series_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description="文件路径")] = None,
    file_content: Annotated[Optional[str], Field(default=None, description="文件内容")] = None,
    file_format: Annotated[str, Field(default="auto", description="文件格式")] = "auto",
    data: Annotated[Optional[List[float]], Field(default=None, description="时间序列数据")] = None
) -> CallToolResult:
    """时间序列分析 - 支持文件或直接数据输入"""
    return await handle_time_series_analysis(ctx, data=data)


@mcp.tool()
@econometric_tool('multi_var_dict')
async def correlation_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description="文件路径")] = None,
    file_content: Annotated[Optional[str], Field(default=None, description="文件内容")] = None,
    file_format: Annotated[str, Field(default="auto", description="文件格式")] = "auto",
    data: Annotated[Optional[Dict[str, List[float]]], Field(default=None, description="多变量数据")] = None,
    method: Annotated[str, Field(default="pearson", description="相关系数类型")] = "pearson"
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
    file_path: Annotated[Optional[str], Field(
        default=None, 
        description="CSV文件路径。CSV格式要求：必须包含实体ID列(列名含entity_id/id/entity/firm/company/country/region之一)和时间列(列名含time_period/time/date/year/month/period/quarter之一)"
    )] = None,
    file_content: Annotated[Optional[str], Field(default=None)] = None,
    file_format: Annotated[str, Field(default="auto")] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None)] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None)] = None,
    entity_ids: Annotated[Optional[List[str]], Field(default=None)] = None,
    time_periods: Annotated[Optional[List[str]], Field(default=None)] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None)] = None,
    entity_effects: Annotated[bool, Field(default=True)] = True,
    time_effects: Annotated[bool, Field(default=False)] = False
) -> CallToolResult:
    """固定效应模型 - 支持文件输入"""
    return await handle_panel_fixed_effects(ctx, y_data, x_data, entity_ids, time_periods, 
                                           feature_names, entity_effects, time_effects)


@mcp.tool()
@econometric_tool('panel')
async def panel_random_effects(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(
        default=None, 
        description="CSV文件路径。CSV格式要求：必须包含实体ID列(列名含entity_id/id/entity/firm/company/country/region之一)和时间列(列名含time_period/time/date/year/month/period/quarter之一)"
    )] = None,
    file_content: Annotated[Optional[str], Field(default=None)] = None,
    file_format: Annotated[str, Field(default="auto")] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None)] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None)] = None,
    entity_ids: Annotated[Optional[List[str]], Field(default=None)] = None,
    time_periods: Annotated[Optional[List[str]], Field(default=None)] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None)] = None,
    entity_effects: Annotated[bool, Field(default=True)] = True,
    time_effects: Annotated[bool, Field(default=False)] = False
) -> CallToolResult:
    """随机效应模型 - 支持文件输入"""
    return await handle_panel_random_effects(ctx, y_data, x_data, entity_ids, time_periods,
                                            feature_names, entity_effects, time_effects)


@mcp.tool()
@econometric_tool('panel')
async def panel_hausman_test(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(
        default=None, 
        description="CSV文件路径。CSV格式要求：必须包含实体ID列(列名含entity_id/id/entity/firm/company/country/region之一)和时间列(列名含time_period/time/date/year/month/period/quarter之一)"
    )] = None,
    file_content: Annotated[Optional[str], Field(default=None)] = None,
    file_format: Annotated[str, Field(default="auto")] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None)] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None)] = None,
    entity_ids: Annotated[Optional[List[str]], Field(default=None)] = None,
    time_periods: Annotated[Optional[List[str]], Field(default=None)] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None)] = None
) -> CallToolResult:
    """Hausman检验 - 支持文件输入"""
    return await handle_panel_hausman_test(ctx, y_data, x_data, entity_ids, time_periods, feature_names)


@mcp.tool()
@econometric_tool('panel')  # 保持panel类型以获取entity_ids和time_periods
async def panel_unit_root_test(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(
        default=None, 
        description="CSV文件路径。CSV格式要求：必须包含实体ID列(列名含entity_id/id/entity/firm/company/country/region之一)和时间列(列名含time_period/time/date/year/month/period/quarter之一)。数据量要求：至少3个实体，每个实体至少5个时间点"
    )] = None,
    file_content: Annotated[Optional[str], Field(default=None)] = None,
    file_format: Annotated[str, Field(default="auto")] = "auto",
    data: Annotated[Optional[List[float]], Field(default=None)] = None,
    y_data: Annotated[Optional[List[float]], Field(default=None)] = None,  # 从panel转换来的
    x_data: Annotated[Optional[List[List[float]]], Field(default=None)] = None,  # 从panel转换来的，忽略
    entity_ids: Annotated[Optional[List[str]], Field(default=None)] = None,
    time_periods: Annotated[Optional[List[str]], Field(default=None)] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None)] = None,  # 从panel转换来的，忽略
    test_type: Annotated[str, Field(default="levinlin")] = "levinlin"
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
    file_path: Annotated[Optional[str], Field(default=None, description="文件路径")] = None,
    file_content: Annotated[Optional[str], Field(default=None)] = None,
    file_format: Annotated[str, Field(default="auto")] = "auto",
    data: Annotated[Optional[Dict[str, List[float]]], Field(default=None)] = None,
    max_lags: Annotated[int, Field(default=5)] = 5,
    ic: Annotated[str, Field(default="aic")] = "aic"
) -> CallToolResult:
    """VAR模型分析 - 支持文件输入"""
    return await handle_var_model(ctx, data, max_lags, ic)


@mcp.tool()
@econometric_tool('time_series')
async def vecm_model_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description="文件路径")] = None,
    file_content: Annotated[Optional[str], Field(default=None)] = None,
    file_format: Annotated[str, Field(default="auto")] = "auto",
    data: Annotated[Optional[Dict[str, List[float]]], Field(default=None)] = None,
    coint_rank: Annotated[int, Field(default=1)] = 1,
    deterministic: Annotated[str, Field(default="co")] = "co",
    max_lags: Annotated[int, Field(default=5)] = 5
) -> CallToolResult:
    """VECM模型分析 - 支持文件输入"""
    return await handle_vecm_model(ctx, data, coint_rank, deterministic, max_lags)


@mcp.tool()
@econometric_tool('single_var')
async def garch_model_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description="文件路径")] = None,
    file_content: Annotated[Optional[str], Field(default=None)] = None,
    file_format: Annotated[str, Field(default="auto")] = "auto",
    data: Annotated[Optional[List[float]], Field(default=None)] = None,
    order: Annotated[tuple, Field(default=(1, 1))] = (1, 1),
    dist: Annotated[str, Field(default="normal")] = "normal"
) -> CallToolResult:
    """GARCH模型分析 - 支持文件输入"""
    return await handle_garch_model(ctx, data, order, dist)


@mcp.tool()
@econometric_tool('single_var')
async def state_space_model_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description="文件路径")] = None,
    file_content: Annotated[Optional[str], Field(default=None)] = None,
    file_format: Annotated[str, Field(default="auto")] = "auto",
    data: Annotated[Optional[List[float]], Field(default=None)] = None,
    state_dim: Annotated[int, Field(default=1)] = 1,
    observation_dim: Annotated[int, Field(default=1)] = 1,
    trend: Annotated[bool, Field(default=True)] = True,
    seasonal: Annotated[bool, Field(default=False)] = False,
    period: Annotated[int, Field(default=12)] = 12
) -> CallToolResult:
    """状态空间模型分析 - 支持文件输入"""
    return await handle_state_space_model(ctx, data, state_dim, observation_dim, trend, seasonal, period)


@mcp.tool()
@econometric_tool('time_series')
async def variance_decomposition_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description="文件路径")] = None,
    file_content: Annotated[Optional[str], Field(default=None)] = None,
    file_format: Annotated[str, Field(default="auto")] = "auto",
    data: Annotated[Optional[Dict[str, List[float]]], Field(default=None)] = None,
    periods: Annotated[int, Field(default=10)] = 10,
    max_lags: Annotated[int, Field(default=5)] = 5
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
    file_path: Annotated[Optional[str], Field(default=None, description="文件路径")] = None,
    file_content: Annotated[Optional[str], Field(default=None)] = None,
    file_format: Annotated[str, Field(default="auto")] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None)] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None)] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None)] = None,
    n_estimators: Annotated[int, Field(default=100)] = 100,
    max_depth: Annotated[Optional[int], Field(default=None)] = None
) -> CallToolResult:
    """随机森林回归 - 支持文件输入"""
    return await handle_random_forest(ctx, y_data, x_data, feature_names, n_estimators, max_depth)


@mcp.tool()
@econometric_tool('regression')
async def gradient_boosting_regression_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description="文件路径")] = None,
    file_content: Annotated[Optional[str], Field(default=None)] = None,
    file_format: Annotated[str, Field(default="auto")] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None)] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None)] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None)] = None,
    n_estimators: Annotated[int, Field(default=100)] = 100,
    learning_rate: Annotated[float, Field(default=0.1)] = 0.1,
    max_depth: Annotated[int, Field(default=3)] = 3
) -> CallToolResult:
    """梯度提升树回归 - 支持文件输入"""
    return await handle_gradient_boosting(ctx, y_data, x_data, feature_names, n_estimators, learning_rate, max_depth)


@mcp.tool()
@econometric_tool('regression')
async def lasso_regression_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description="文件路径")] = None,
    file_content: Annotated[Optional[str], Field(default=None)] = None,
    file_format: Annotated[str, Field(default="auto")] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None)] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None)] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None)] = None,
    alpha: Annotated[float, Field(default=1.0)] = 1.0
) -> CallToolResult:
    """Lasso回归 - 支持文件输入"""
    return await handle_lasso_regression(ctx, y_data, x_data, feature_names, alpha)


@mcp.tool()
@econometric_tool('regression')
async def ridge_regression_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description="文件路径")] = None,
    file_content: Annotated[Optional[str], Field(default=None)] = None,
    file_format: Annotated[str, Field(default="auto")] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None)] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None)] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None)] = None,
    alpha: Annotated[float, Field(default=1.0)] = 1.0
) -> CallToolResult:
    """Ridge回归 - 支持文件输入"""
    return await handle_ridge_regression(ctx, y_data, x_data, feature_names, alpha)


@mcp.tool()
@econometric_tool('regression')
async def cross_validation_analysis(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description="文件路径")] = None,
    file_content: Annotated[Optional[str], Field(default=None)] = None,
    file_format: Annotated[str, Field(default="auto")] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None)] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None)] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None)] = None,
    model_type: Annotated[str, Field(default="random_forest")] = "random_forest",
    cv_folds: Annotated[int, Field(default=5)] = 5,
    scoring: Annotated[str, Field(default="r2")] = "r2"
) -> CallToolResult:
    """交叉验证 - 支持文件输入"""
    return await handle_cross_validation(ctx, y_data, x_data, model_type, cv_folds, scoring)


@mcp.tool()
@econometric_tool('regression')
async def feature_importance_analysis_tool(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description="文件路径")] = None,
    file_content: Annotated[Optional[str], Field(default=None)] = None,
    file_format: Annotated[str, Field(default="auto")] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None)] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None)] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None)] = None,
    method: Annotated[str, Field(default="random_forest")] = "random_forest",
    top_k: Annotated[int, Field(default=5)] = 5
) -> CallToolResult:
    """特征重要性分析 - 支持文件输入"""
    return await handle_feature_importance(ctx, y_data, x_data, feature_names, method, top_k)


def create_mcp_server() -> FastMCP:
    """创建并返回MCP服务器实例"""
    return mcp