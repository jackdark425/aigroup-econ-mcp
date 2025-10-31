"""
工具模块基类和装饰器
整合了工具基类、装饰器、错误处理和优化组件
"""

import functools
from typing import Any, Dict, List, Optional, Callable, Type
from functools import wraps
from mcp.server.session import ServerSession
from mcp.server.fastmcp import Context
from mcp.types import CallToolResult, TextContent

from .validation import ValidationError, validate_econometric_data, validate_model_parameters
from .cache import cache_result, cache_model, global_econometric_cache
from .monitoring import monitor_performance, track_progress, global_performance_monitor
from .file_parser import FileParser
from ..config import get_config, econometric_config


# ============================================================================
# 错误类定义
# ============================================================================

class EconometricToolError(Exception):
    """计量经济学工具错误基类"""
    
    def __init__(self, message: str, tool_name: str = None, original_error: Exception = None):
        self.message = message
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(self.message)
    
    def __str__(self):
        base_msg = f"计量经济学工具错误"
        if self.tool_name:
            base_msg += f" ({self.tool_name})"
        base_msg += f": {self.message}"
        if self.original_error:
            base_msg += f"\n原始错误: {self.original_error}"
        return base_msg


class DataValidationError(EconometricToolError):
    """数据验证错误"""
    pass


class ModelFittingError(EconometricToolError):
    """模型拟合错误"""
    pass


class ConfigurationError(EconometricToolError):
    """配置错误"""
    pass


# ============================================================================
# 装饰器函数
# ============================================================================

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


def with_file_support_decorator(
    tool_type: str,
    enable_error_handling: bool = True,
    enable_logging: bool = True
):
    """
    组合装饰器：为计量经济学工具添加文件支持和其他标准功能
    
    Args:
        tool_type: 工具类型
        enable_error_handling: 是否启用错误处理
        enable_logging: 是否启用日志记录
    
    使用示例:
        @with_file_support_decorator('regression')
        async def ols_regression(ctx, y_data=None, x_data=None, **kwargs):
            # 只需要编写核心业务逻辑
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


# ============================================================================
# 工具基类
# ============================================================================

class EconometricTool:
    """
    计量经济学工具基类
    提供统一的参数验证、缓存、性能监控和错误处理
    """
    
    def __init__(self, tool_name: str):
        """
        初始化工具
        
        Args:
            tool_name: 工具名称
        """
        self.tool_name = tool_name
        self._cache_enabled = get_config("cache_enabled", True)
        self._monitoring_enabled = get_config("monitoring_enabled", True)
        self._validation_strict = get_config("data_validation_strict", True)
    
    def _validate_input_data(self, data: Any, data_type: str = "generic") -> Any:
        """
        验证输入数据
        
        Args:
            data: 输入数据
            data_type: 数据类型
            
        Returns:
            Any: 验证后的数据
            
        Raises:
            DataValidationError: 数据验证失败
        """
        try:
            if data_type == "econometric":
                return validate_econometric_data(data)
            elif data_type == "time_series":
                from .validation import validate_time_series_data
                return validate_time_series_data(data)
            elif data_type == "model_parameters":
                return validate_model_parameters(data)
            else:
                return data
        except ValidationError as e:
            error_msg = f"数据验证失败: {e.message}"
            if self._validation_strict:
                raise DataValidationError(error_msg, self.tool_name, e)
            else:
                # 在非严格模式下记录警告并继续
                import warnings
                warnings.warn(f"{self.tool_name}: {error_msg}")
                return data
    
    def _handle_errors(self, func: Callable) -> Callable:
        """
        错误处理装饰器
        
        Args:
            func: 被装饰的函数
            
        Returns:
            Callable: 装饰后的函数
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (DataValidationError, ModelFittingError, ConfigurationError):
                # 重新抛出已知的错误类型
                raise
            except ValidationError as e:
                # 转换验证错误
                raise DataValidationError(f"参数验证失败: {e.message}", self.tool_name, e)
            except Exception as e:
                # 处理其他未知错误
                error_msg = f"执行过程中发生未知错误: {str(e)}"
                raise EconometricToolError(error_msg, self.tool_name, e)
        
        return wrapper
    
    def _apply_optimizations(self, func: Callable) -> Callable:
        """
        应用优化装饰器
        
        Args:
            func: 被装饰的函数
            
        Returns:
            Callable: 优化后的函数
        """
        # 应用性能监控
        if self._monitoring_enabled:
            func = monitor_performance(self.tool_name)(func)
        
        # 应用缓存
        if self._cache_enabled:
            cache_config = econometric_config.get_cache_config()
            func = cache_result(ttl=cache_config["ttl"], max_size=cache_config["max_size"])(func)
        
        # 应用错误处理
        func = self._handle_errors(func)
        
        return func
    
    def create_optimized_function(self, func: Callable) -> Callable:
        """
        创建优化函数
        
        Args:
            func: 原始函数
            
        Returns:
            Callable: 优化后的函数
        """
        return self._apply_optimizations(func)
    
    def execute_with_progress(self, total_steps: int, description: str = ""):
        """
        进度跟踪上下文管理器
        
        Args:
            total_steps: 总步骤数
            description: 进度描述
            
        Returns:
            ContextManager: 进度跟踪上下文管理器
        """
        return track_progress(total_steps, f"{self.tool_name}: {description}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            Dict[str, Any]: 性能统计信息
        """
        if not self._monitoring_enabled:
            return {}
        
        stats = global_performance_monitor.get_function_stats(self.tool_name)
        if stats:
            return {
                "execution_time": stats.execution_time,
                "peak_memory_mb": stats.peak_memory_mb,
                "cpu_percent": stats.cpu_percent,
                "timestamp": stats.timestamp
            }
        return {}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计
        
        Returns:
            Dict[str, Any]: 缓存统计信息
        """
        if not self._cache_enabled:
            return {}
        
        return global_econometric_cache.result_cache.get_function_cache_stats(self.tool_name) or {}


# ============================================================================
# 便捷装饰器函数
# ============================================================================

def econometric_tool_with_optimization(tool_name: str = None):
    """
    计量经济学工具装饰器（带优化）
    
    应用缓存、性能监控和错误处理
    
    Args:
        tool_name: 工具名称
        
    Returns:
        Callable: 装饰器函数
        
    使用示例:
        @econometric_tool_with_optimization('my_analysis')
        def my_analysis(data):
            # 自动获得缓存、监控和错误处理
            pass
    """
    def decorator(func):
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
        Callable: 装饰器函数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 验证第一个位置参数（通常是数据）
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
            
            return func(*args, **validated_kwargs)
        
        return wrapper
    
    return decorator


# 导出主要类和函数
__all__ = [
    # 错误类
    "EconometricToolError",
    "DataValidationError", 
    "ModelFittingError",
    "ConfigurationError",
    # 基类
    "EconometricTool",
    # 装饰器
    "with_file_input",
    "with_error_handling",
    "with_logging",
    "with_file_support_decorator",
    "econometric_tool_with_optimization",
    "validate_input"
]