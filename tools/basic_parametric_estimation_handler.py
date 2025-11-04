"""
基础与参数估计模块处理器
处理OLS、MLE和GMM工具的MCP调用
"""

from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import Context

from econometrics.basic_parametric_estimation import (
    ols_regression,
    mle_estimation,
    gmm_estimation
)


async def handle_ols_regression(ctx: Context, **kwargs) -> Dict[str, Any]:
    """
    处理OLS回归工具调用
    
    Args:
        ctx: MCP上下文
        **kwargs: 工具参数
        
    Returns:
        Dict[str, Any]: OLS回归结果
    """
    # 提取参数
    y_data = kwargs.get("y_data")
    x_data = kwargs.get("x_data")
    feature_names = kwargs.get("feature_names")
    constant = kwargs.get("constant", True)
    confidence_level = kwargs.get("confidence_level", 0.95)
    
    # 调用OLS回归函数
    result = ols_regression(
        y_data=y_data,
        x_data=x_data,
        feature_names=feature_names,
        constant=constant,
        confidence_level=confidence_level
    )
    
    # 返回结果字典
    return result.dict()


async def handle_mle_estimation(ctx: Context, **kwargs) -> Dict[str, Any]:
    """
    处理最大似然估计工具调用
    
    Args:
        ctx: MCP上下文
        **kwargs: 工具参数
        
    Returns:
        Dict[str, Any]: 最大似然估计结果
    """
    # 提取参数
    data = kwargs.get("data")
    distribution = kwargs.get("distribution", "normal")
    initial_params = kwargs.get("initial_params")
    confidence_level = kwargs.get("confidence_level", 0.95)
    
    # 调用MLE估计函数
    result = mle_estimation(
        data=data,
        distribution=distribution,
        initial_params=initial_params,
        confidence_level=confidence_level
    )
    
    # 返回结果字典
    return result.dict()


async def handle_gmm_estimation(ctx: Context, **kwargs) -> Dict[str, Any]:
    """
    处理广义矩估计工具调用
    
    Args:
        ctx: MCP上下文
        **kwargs: 工具参数
        
    Returns:
        Dict[str, Any]: 广义矩估计结果
    """
    # 提取参数
    y_data = kwargs.get("y_data")
    x_data = kwargs.get("x_data")
    instruments = kwargs.get("instruments")
    feature_names = kwargs.get("feature_names")
    constant = kwargs.get("constant", True)
    confidence_level = kwargs.get("confidence_level", 0.95)
    
    # 调用GMM估计函数
    result = gmm_estimation(
        y_data=y_data,
        x_data=x_data,
        instruments=instruments,
        feature_names=feature_names,
        constant=constant,
        confidence_level=confidence_level
    )
    
    # 返回结果字典
    return result.dict()