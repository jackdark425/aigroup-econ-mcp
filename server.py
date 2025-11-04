"""
AIGroup 计量经济学 MCP 服务器 - 基于 MCP Python SDK 重构
提供 OLS、MLE、GMM 三个基础参数估计工具
"""

import asyncio
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass

import mcp.server
import mcp.server.stdio
import mcp.types as types
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# 导入工具实现
from tools.ols_tool import ols_regression
from tools.mle_tool import mle_estimation
from tools.gmm_tool import gmm_estimation


@dataclass
class AppContext:
    """应用上下文，包含共享资源"""
    config: Dict[str, Any]
    version: str = "1.4.1"


@asynccontextmanager
async def lifespan(server: mcp.server.Server):
    """服务器生命周期管理"""
    config = {
        "max_sample_size": 10000,
        "default_significance_level": 0.05,
        "supported_distributions": ["normal", "poisson", "exponential"],
        "data_types": ["cross_section", "time_series", "panel"]
    }
    try:
        yield AppContext(config=config)
    finally:
        pass


# 创建 MCP 服务器实例
server = mcp.server.Server("aigroup-econ-mcp")


# ============================================================================
# 基础参数估计工具 (3个)
# ============================================================================

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """列出所有可用工具"""
    return [
        types.Tool(
            name="basic_parametric_estimation_ols",
            description="普通最小二乘法(OLS)参数估计",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "file_content": {
                        "type": "string",
                        "description": "文件内容字符串"
                    },
                    "file_format": {
                        "type": "string",
                        "description": "文件格式",
                        "default": "auto"
                    },
                    "y_data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "因变量数据"
                    },
                    "x_data": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"}
                        },
                        "description": "自变量数据矩阵"
                    },
                    "feature_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "特征名称列表"
                    },
                    "constant": {
                        "type": "boolean",
                        "description": "是否包含常数项",
                        "default": True
                    },
                    "confidence_level": {
                        "type": "number",
                        "description": "置信水平",
                        "default": 0.95
                    }
                }
            }
        ),
        types.Tool(
            name="basic_parametric_estimation_mle",
            description="最大似然估计(MLE)",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "file_content": {
                        "type": "string",
                        "description": "文件内容字符串"
                    },
                    "file_format": {
                        "type": "string",
                        "description": "文件格式",
                        "default": "auto"
                    },
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "用于估计的数据"
                    },
                    "distribution": {
                        "type": "string",
                        "description": "分布类型: normal(正态分布), poisson(泊松分布), exponential(指数分布)",
                        "default": "normal"
                    },
                    "initial_params": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "初始参数值"
                    },
                    "confidence_level": {
                        "type": "number",
                        "description": "置信水平",
                        "default": 0.95
                    }
                }
            }
        ),
        types.Tool(
            name="basic_parametric_estimation_gmm",
            description="广义矩估计(GMM)",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "file_content": {
                        "type": "string",
                        "description": "文件内容字符串"
                    },
                    "file_format": {
                        "type": "string",
                        "description": "文件格式",
                        "default": "auto"
                    },
                    "y_data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "因变量数据"
                    },
                    "x_data": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"}
                        },
                        "description": "自变量数据矩阵"
                    },
                    "instruments": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"}
                        },
                        "description": "工具变量数据矩阵"
                    },
                    "feature_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "特征名称列表"
                    },
                    "constant": {
                        "type": "boolean",
                        "description": "是否包含常数项",
                        "default": True
                    },
                    "confidence_level": {
                        "type": "number",
                        "description": "置信水平",
                        "default": 0.95
                    }
                }
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> list[types.TextContent]:
    """处理工具调用"""
    try:
        if name == "basic_parametric_estimation_ols":
            result = await _handle_ols_regression(arguments)
        elif name == "basic_parametric_estimation_mle":
            result = await _handle_mle_estimation(arguments)
        elif name == "basic_parametric_estimation_gmm":
            result = await _handle_gmm_estimation(arguments)
        else:
            raise ValueError(f"未知工具: {name}")
        
        return [types.TextContent(type="text", text=result)]
    
    except Exception as e:
        error_msg = f"工具执行错误: {str(e)}"
        return [types.TextContent(type="text", text=error_msg)]


async def _handle_ols_regression(arguments: Dict[str, Any]) -> str:
    """处理OLS回归"""
    y_data = arguments.get("y_data")
    x_data = arguments.get("x_data")
    feature_names = arguments.get("feature_names")
    constant = arguments.get("constant", True)
    confidence_level = arguments.get("confidence_level", 0.95)
    
    if y_data is None or x_data is None:
        return "错误: 必须提供y_data和x_data参数"
    
    result = ols_regression(
        y_data=y_data,
        x_data=x_data,
        feature_names=feature_names,
        constant=constant,
        confidence_level=confidence_level
    )
    
    return _format_ols_result(result)


async def _handle_mle_estimation(arguments: Dict[str, Any]) -> str:
    """处理MLE估计"""
    data = arguments.get("data")
    distribution = arguments.get("distribution", "normal")
    initial_params = arguments.get("initial_params")
    confidence_level = arguments.get("confidence_level", 0.95)
    
    if data is None:
        return "错误: 必须提供data参数"
    
    result = mle_estimation(
        data=data,
        distribution=distribution,
        initial_params=initial_params,
        confidence_level=confidence_level
    )
    
    return _format_mle_result(result)


async def _handle_gmm_estimation(arguments: Dict[str, Any]) -> str:
    """处理GMM估计"""
    y_data = arguments.get("y_data")
    x_data = arguments.get("x_data")
    instruments = arguments.get("instruments")
    feature_names = arguments.get("feature_names")
    constant = arguments.get("constant", True)
    confidence_level = arguments.get("confidence_level", 0.95)
    
    if y_data is None or x_data is None:
        return "错误: 必须提供y_data和x_data参数"
    
    result = gmm_estimation(
        y_data=y_data,
        x_data=x_data,
        instruments=instruments,
        feature_names=feature_names,
        constant=constant,
        confidence_level=confidence_level
    )
    
    return _format_gmm_result(result)


def _format_ols_result(result) -> str:
    """格式化OLS结果"""
    output = "OLS回归结果:\n\n"
    output += f"观测数量: {result.n_obs}\n"
    output += f"R方: {result.r_squared:.4f}\n"
    output += f"调整R方: {result.adj_r_squared:.4f}\n"
    output += f"F统计量: {result.f_statistic:.4f} (p值: {result.f_p_value:.4f})\n\n"
    
    output += "系数估计:\n"
    output += "变量名\t系数\t标准误\tt值\tp值\t置信区间\n"
    for i, name in enumerate(result.feature_names):
        coef = result.coefficients[i]
        se = result.std_errors[i]
        t_val = result.t_values[i]
        p_val = result.p_values[i]
        ci_lower = result.conf_int_lower[i]
        ci_upper = result.conf_int_upper[i]
        
        output += f"{name}\t{coef:.4f}\t{se:.4f}\t{t_val:.4f}\t{p_val:.4f}\t[{ci_lower:.4f}, {ci_upper:.4f}]\n"
    
    return output


def _format_mle_result(result) -> str:
    """格式化MLE结果"""
    output = "最大似然估计结果:\n\n"
    output += f"观测数量: {result.n_obs}\n"
    output += f"对数似然值: {result.log_likelihood:.4f}\n"
    output += f"AIC: {result.aic:.4f}\n"
    output += f"BIC: {result.bic:.4f}\n"
    output += f"收敛状态: {'是' if result.convergence else '否'}\n\n"
    
    output += "参数估计:\n"
    output += "参数名\t估计值\t标准误\t置信区间\n"
    for i, name in enumerate(result.param_names):
        param = result.parameters[i]
        se = result.std_errors[i]
        ci_lower = result.conf_int_lower[i]
        ci_upper = result.conf_int_upper[i]
        
        output += f"{name}\t{param:.4f}\t{se:.4f}\t[{ci_lower:.4f}, {ci_upper:.4f}]\n"
    
    return output


def _format_gmm_result(result) -> str:
    """格式化GMM结果"""
    output = "广义矩估计结果:\n\n"
    output += f"观测数量: {result.n_obs}\n"
    output += f"矩条件数量: {result.n_moments}\n"
    output += f"J统计量: {result.j_statistic:.4f} (p值: {result.j_p_value:.4f})\n\n"
    
    output += "系数估计:\n"
    output += "变量名\t系数\t标准误\tt值\tp值\t置信区间\n"
    for i, name in enumerate(result.feature_names):
        coef = result.coefficients[i]
        se = result.std_errors[i]
        t_val = result.t_values[i]
        p_val = result.p_values[i]
        ci_lower = result.conf_int_lower[i]
        ci_upper = result.conf_int_upper[i]
        
        output += f"{name}\t{coef:.4f}\t{se:.4f}\t{t_val:.4f}\t{p_val:.4f}\t[{ci_lower:.4f}, {ci_upper:.4f}]\n"
    
    return output


async def main():
    """启动 MCP 服务器"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())