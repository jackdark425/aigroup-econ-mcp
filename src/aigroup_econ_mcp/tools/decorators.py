"""
工具装饰器模块
提供自动文件输入处理、错误处理等功能
"""

from typing import Callable, Optional, Dict, Any, List
from functools import wraps
from mcp.server.session import ServerSession
from mcp.server.fastmcp import Context
from mcp.types import CallToolResult, TextContent

from .file_parser import FileParser


def with_file_input(tool_type: str):
    """
    为工具函数添加文件输入支持的装饰器
    
    支持两种输入方式：
    1. file_path: CSV/JSON文件路径
    2. file_content: 文件内容字符串
    
    Args:
        tool_type: 工具类型 ('single_var', 'multi_var_dict', 'regression', 'panel', 'time_series')
    
    使用示例:
        @with_file_input('regression')
        async def my_tool(ctx, y_data=None, x_data=None, file_path=None, file_content=None, file_format='auto', **kwargs):
            # 如果提供了file_path或file_content，数据会被自动填充
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 提取上下文和文件参数
            ctx = args[0] if args else kwargs.get('ctx')
            file_path = kwargs.get('file_path')
            file_content = kwargs.get('file_content')
            file_format = kwargs.get('file_format', 'auto')
            
            # 优先处理file_path
            if file_path:
                try:
                    await ctx.info(f"检测到文件路径输入: {file_path}")
                    
                    # 从文件路径解析
                    parsed = FileParser.parse_file_path(file_path, file_format)
                    
                    await ctx.info(
                        f"文件解析成功：{parsed['n_variables']}个变量，"
                        f"{parsed['n_observations']}个观测"
                    )
                    
                    # 转换为工具格式
                    converted = FileParser.convert_to_tool_format(parsed, tool_type)
                    
                    # 更新kwargs
                    kwargs.update(converted)
                    
                    await ctx.info(f"数据已转换为{tool_type}格式")
                    
                except Exception as e:
                    await ctx.error(f"文件解析失败: {str(e)}")
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"文件解析错误: {str(e)}")],
                        isError=True
                    )
            
            # 如果没有file_path但有file_content，处理文件内容
            elif file_content:
                try:
                    await ctx.info("检测到文件内容输入，开始解析...")
                    
                    # 解析文件内容
                    parsed = FileParser.parse_file_content(file_content, file_format)
                    
                    await ctx.info(
                        f"文件解析成功：{parsed['n_variables']}个变量，"
                        f"{parsed['n_observations']}个观测"
                    )
                    
                    # 转换为工具格式
                    converted = FileParser.convert_to_tool_format(parsed, tool_type)
                    
                    # 更新kwargs
                    kwargs.update(converted)
                    
                    await ctx.info(f"数据已转换为{tool_type}格式")
                    
                except Exception as e:
                    await ctx.error(f"文件解析失败: {str(e)}")
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"文件解析错误: {str(e)}")],
                        isError=True
                    )
            
            # 调用原函数
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def with_error_handling(func: Callable) -> Callable:
    """
    为工具函数添加统一错误处理的装饰器
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        ctx = args[0] if args else kwargs.get('ctx')
        tool_name = func.__name__
        
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            await ctx.error(f"{tool_name}执行出错: {str(e)}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"错误: {str(e)}")],
                isError=True
            )
    
    return wrapper


def with_logging(func: Callable) -> Callable:
    """
    为工具函数添加日志记录的装饰器
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


def econometric_tool(
    tool_type: str,
    with_file_support: bool = True,
    with_error_handling: bool = True,
    with_logging: bool = True
):
    """
    组合装饰器：为计量经济学工具添加所有标准功能
    
    Args:
        tool_type: 工具类型
        with_file_support: 是否启用文件输入支持
        with_error_handling: 是否启用错误处理
        with_logging: 是否启用日志记录
    
    使用示例:
        @econometric_tool('regression')
        async def ols_regression(ctx, y_data=None, x_data=None, **kwargs):
            # 只需要编写核心业务逻辑
            pass
    """
    def decorator(func: Callable) -> Callable:
        wrapped = func
        
        if with_error_handling:
            wrapped = globals()['with_error_handling'](wrapped)
        
        if with_file_support:
            wrapped = with_file_input(tool_type)(wrapped)
        
        if with_logging:
            wrapped = globals()['with_logging'](wrapped)
        
        return wrapped
    
    return decorator