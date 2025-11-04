"""
装饰器模块
提供工具装饰器功能
"""

from typing import Any, Callable, Type
from functools import wraps
from mcp.server.session import ServerSession
from mcp.server.fastmcp import Context
from mcp.types import CallToolResult, TextContent

from tools.validation import ValidationError, validate_econometric_data, validate_model_parameters
from tools.cache import cache_result, cache_model, global_econometric_cache
from tools.monitoring import monitor_performance, track_progress, global_performance_monitor
from tools.file_parser import FileParser
from config import get_config, econometric_config


def with_file_input(tool_type: str):
    """
    文件输入装饰器
    自动处理文件输入参数，支持CSV/JSON/TXT格式
    
    1. file_path: CSV/JSON文件路径
    2. file_content: 文件内容字符串
    
    Args:
        tool_type: 工具类型 ('single_var', 'multi_var_dict', 'regression', 'panel', 'time_series')
    
    示例:
        @with_file_input('regression')
        async def my_tool(ctx, y_data=None, x_data=None, file_path=None, file_content=None, file_format='auto', **kwargs):
            # file_path或file_content已自动解析
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 获取上下文
            ctx = args[0] if args else kwargs.get('ctx')
            file_path = kwargs.get('file_path')
            file_content = kwargs.get('file_content')
            file_format = kwargs.get('file_format', 'auto')
            
            # 处理文件路径
            if file_path:
                try:
                    await ctx.info(f"正在解析文件: {file_path}")
                    
                    # 解析文件
                    parsed = FileParser.parse_file_path(file_path, file_format)
                    
                    await ctx.info(
                        f"文件解析成功: {parsed['n_variables']}个变量, "
                        f"{parsed['n_observations']}个观测值"
                    )
                    
                    # 转换为工具格式
                    converted = FileParser.convert_to_tool_format(parsed, tool_type)
                    
                    # 更新kwargs
                    kwargs.update(converted)
                    
                    await ctx.info(f"文件数据已转换为{tool_type}格式")
                    
                except Exception as e:
                    await ctx.error(f"文件解析失败: {str(e)}")
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"文件解析失败: {str(e)}")],
                        isError=True
                    )
            
            # 如果没有文件路径但有文件内容
            elif file_content:
                try:
                    await ctx.info("正在解析文件内容...")
                    
                    # 解析文件内容
                    parsed = FileParser.parse_file_content(file_content, file_format)
                    
                    await ctx.info(
                        f"文件内容解析成功: {parsed['n_variables']}个变量, "
                        f"{parsed['n_observations']}个观测值"
                    )
                    
                    # 转换为工具格式
                    converted = FileParser.convert_to_tool_format(parsed, tool_type)
                    
                    # 更新kwargs
                    kwargs.update(converted)
                    
                    await ctx.info(f"文件数据已转换为{tool_type}格式")
                    
                except Exception as e:
                    await ctx.error(f"文件内容解析失败: {str(e)}")
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"文件内容解析失败: {str(e)}")],
                        isError=True
                    )
            
            # 执行原始函数
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def with_error_handling(func: Callable) -> Callable:
    """
    错误处理装饰器
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        ctx = args[0] if args else kwargs.get('ctx')
        tool_name = func.__name__
        
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            await ctx.error(f"{tool_name}执行失败: {str(e)}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"工具执行失败: {str(e)}")],
                isError=True
            )
    
    return wrapper


def with_logging(func: Callable) -> Callable:
    """
    日志记录装饰器
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        ctx = args[0] if args else kwargs.get('ctx')
        tool_name = func.__name__
        
        await ctx.info(f"开始执行 {tool_name}")
        result = await func(*args, **kwargs)
        await ctx.info(f"{tool_name} 执行完成")
        
        return result
    
    return wrapper


def with_file_support_decorator(
    tool_type: str,
    enable_error_handling: bool = True,
    enable_logging: bool = True
):
    """
    完整文件支持装饰器
    
    Args:
        tool_type: 工具类型
        enable_error_handling: 是否启用错误处理
        enable_logging: 是否启用日志记录
    
    示例:
        @with_file_support_decorator('regression')
        async def ols_regression(ctx, y_data=None, x_data=None, **kwargs):
            # 工具实现
            pass
    """
    def decorator(func: Callable) -> Callable:
        wrapped = func
        
        if enable_error_handling:
            wrapped = with_error_handling(wrapped)
        
        wrapped = with_file_input(tool_type)(wrapped)
        
        if enable_logging:
            wrapped = with_logging(wrapped)
        
        return wrapped
    
    return decorator


def econometric_tool_with_optimization(tool_name: str = None):
    """
    计量经济学工具优化装饰器
    
    自动应用性能监控、缓存和错误处理
    
    Args:
        tool_name: 工具名称
        
    Returns:
        Callable: 装饰器
        
    示例:
        @econometric_tool_with_optimization('my_analysis')
        def my_analysis(data):
            # 工具实现
            pass
    """
    def decorator(func):
        from .tool_base import EconometricTool
        name = tool_name or func.__name__
        tool = EconometricTool(name)
        return tool.create_optimized_function(func)
    
    return decorator


def validate_input(data_type: str = "generic"):
    """
    输入验证装饰器
    
    Args:
        data_type: 数据类型
        
    Returns:
        Callable: 装饰器
    """
    def decorator(func):
        from .tool_base import EconometricTool
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 验证位置参数
            if args:
                validated_args = list(args)
                validated_args[0] = EconometricTool(func.__name__)._validate_input_data(args[0], data_type)
                args = tuple(validated_args)
            
            # 验证关键字参数
            validated_kwargs = {}
            for key, value in kwargs.items():
                if key in ["data", "y_data", "x_data"]:
                    validated_kwargs[key] = EconometricTool(func.__name__)._validate_input_data(value, data_type)
                else:
                    validated_kwargs[key] = value
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# 导出
__all__ = [
    "with_file_input",
    "with_error_handling", 
    "with_logging",
    "with_file_support_decorator",
    "econometric_tool_with_optimization",
    "validate_input"
]