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

# 导入最新的三个工具处理器
from tools.basic_parametric_estimation_handler import (
    handle_ols_regression,
    handle_mle_estimation,
    handle_gmm_estimation
)

# 导入装饰器和工具描述
from tools.decorators import with_file_support_decorator as econometric_tool
from tools.tool_descriptions import (
    get_tool_description,
    get_field_description,
    BASIC_PARAMETRIC_ESTIMATION_OLS,
    BASIC_PARAMETRIC_ESTIMATION_MLE,
    BASIC_PARAMETRIC_ESTIMATION_GMM
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
# 最新的基础参数估计工具 (3个) - 自动支持文件输入
# ============================================================================

@mcp.tool()
@econometric_tool('regression')
async def basic_parametric_estimation_ols(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=get_field_description("basic_parametric_estimation_ols", "file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=get_field_description("basic_parametric_estimation_ols", "file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=get_field_description("basic_parametric_estimation_ols", "file_format"))] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None, description=get_field_description("basic_parametric_estimation_ols", "y_data"))] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None, description=get_field_description("basic_parametric_estimation_ols", "x_data"))] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None, description=get_field_description("basic_parametric_estimation_ols", "feature_names"))] = None,
    constant: Annotated[bool, Field(default=True, description=get_field_description("basic_parametric_estimation_ols", "constant"))] = True,
    confidence_level: Annotated[float, Field(default=0.95, description=get_field_description("basic_parametric_estimation_ols", "confidence_level"))] = 0.95
) -> CallToolResult:
    """普通最小二乘法(OLS)参数估计"""
    return await handle_ols_regression(ctx, y_data=y_data, x_data=x_data, feature_names=feature_names, constant=constant, confidence_level=confidence_level)


@mcp.tool()
@econometric_tool('single_var')
async def basic_parametric_estimation_mle(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=get_field_description("basic_parametric_estimation_mle", "file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=get_field_description("basic_parametric_estimation_mle", "file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=get_field_description("basic_parametric_estimation_mle", "file_format"))] = "auto",
    data: Annotated[Optional[List[float]], Field(default=None, description=get_field_description("basic_parametric_estimation_mle", "data"))] = None,
    distribution: Annotated[str, Field(default="normal", description=get_field_description("basic_parametric_estimation_mle", "distribution"))] = "normal",
    initial_params: Annotated[Optional[List[float]], Field(default=None, description=get_field_description("basic_parametric_estimation_mle", "initial_params"))] = None,
    confidence_level: Annotated[float, Field(default=0.95, description=get_field_description("basic_parametric_estimation_mle", "confidence_level"))] = 0.95
) -> CallToolResult:
    """最大似然估计(MLE)"""
    return await handle_mle_estimation(ctx, data=data, distribution=distribution, initial_params=initial_params, confidence_level=confidence_level)


@mcp.tool()
@econometric_tool('regression')
async def basic_parametric_estimation_gmm(
    ctx: Context[ServerSession, AppContext],
    file_path: Annotated[Optional[str], Field(default=None, description=get_field_description("basic_parametric_estimation_gmm", "file_path"))] = None,
    file_content: Annotated[Optional[str], Field(default=None, description=get_field_description("basic_parametric_estimation_gmm", "file_content"))] = None,
    file_format: Annotated[str, Field(default="auto", description=get_field_description("basic_parametric_estimation_gmm", "file_format"))] = "auto",
    y_data: Annotated[Optional[List[float]], Field(default=None, description=get_field_description("basic_parametric_estimation_gmm", "y_data"))] = None,
    x_data: Annotated[Optional[List[List[float]]], Field(default=None, description=get_field_description("basic_parametric_estimation_gmm", "x_data"))] = None,
    instruments: Annotated[Optional[List[List[float]]], Field(default=None, description=get_field_description("basic_parametric_estimation_gmm", "instruments"))] = None,
    feature_names: Annotated[Optional[List[str]], Field(default=None, description=get_field_description("basic_parametric_estimation_gmm", "feature_names"))] = None,
    constant: Annotated[bool, Field(default=True, description=get_field_description("basic_parametric_estimation_gmm", "constant"))] = True,
    confidence_level: Annotated[float, Field(default=0.95, description=get_field_description("basic_parametric_estimation_gmm", "confidence_level"))] = 0.95
) -> CallToolResult:
    """广义矩估计(GMM)"""
    return await handle_gmm_estimation(ctx, y_data=y_data, x_data=x_data, instruments=instruments, feature_names=feature_names, constant=constant, confidence_level=confidence_level)


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