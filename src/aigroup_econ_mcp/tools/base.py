"""
工具模块基类
提供统一的优化组件集成和错误处理
"""

import functools
from typing import Any, Dict, List, Optional, Callable, Type
from .validation import ValidationError, validate_econometric_data, validate_model_parameters
from .cache import cache_result, cache_model, global_econometric_cache
from .monitoring import monitor_performance, track_progress, global_performance_monitor
from ..config import get_config, econometric_config


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


# 便捷装饰器函数
def econometric_tool(tool_name: str = None):
    """
    计量经济学工具装饰器
    
    Args:
        tool_name: 工具名称
        
    Returns:
        Callable: 装饰器函数
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
    "EconometricToolError",
    "DataValidationError", 
    "ModelFittingError",
    "ConfigurationError",
    "EconometricTool",
    "econometric_tool",
    "validate_input"
]