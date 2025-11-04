"""
工具基类模块
提供计量经济学工具基类和核心功能
"""

import functools
from typing import Any, Dict, List, Optional, Callable, Type
from functools import wraps

from tools.validation import ValidationError, validate_econometric_data, validate_model_parameters
from tools.cache import cache_result, cache_model, global_econometric_cache
from tools.monitoring import monitor_performance, track_progress, global_performance_monitor
from config import get_config, econometric_config
from .exceptions import EconometricToolError, DataValidationError, ModelFittingError, ConfigurationError


class EconometricTool:
    """
    计量经济学工具基类
    提供统一的工具功能
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
                from tools.validation import validate_time_series_data
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
                # 非严格模式下发出警告
                import warnings
                warnings.warn(f"{self.tool_name}: {error_msg}")
                return data
    
    def _handle_errors(self, func: Callable) -> Callable:
        """
        错误处理包装器
        
        Args:
            func: 原始函数
            
        Returns:
            Callable: 包装后的函数
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (DataValidationError, ModelFittingError, ConfigurationError):
                # 已知异常直接抛出
                raise
            except ValidationError as e:
                # 验证错误转换为数据验证错误
                raise DataValidationError(f"数据验证失败: {e.message}", self.tool_name, e)
            except Exception as e:
                # 其他异常转换为工具错误
                error_msg = f"工具执行失败: {str(e)}"
                raise EconometricToolError(error_msg, self.tool_name, e)
        
        return wrapper
    
    def _apply_optimizations(self, func: Callable) -> Callable:
        """
        应用优化
        
        Args:
            func: 原始函数
            
        Returns:
            Callable: 优化后的函数
        """
        # 性能监控
        if self._monitoring_enabled:
            func = monitor_performance(self.tool_name)(func)
        
        # 缓存
        if self._cache_enabled:
            cache_config = econometric_config.get_cache_config()
            func = cache_result(ttl=cache_config["ttl"], max_size=cache_config["max_size"])(func)
        
        # 错误处理
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
        带进度执行的上下文管理器
        
        Args:
            total_steps: 总步骤数
            description: 描述
            
        Returns:
            ContextManager: 进度跟踪上下文管理器
        """
        return track_progress(total_steps, f"{self.tool_name}: {description}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            Dict[str, Any]: 性能统计
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
            Dict[str, Any]: 缓存统计
        """
        if not self._cache_enabled:
            return {}
        
        return global_econometric_cache.result_cache.get_function_cache_stats(self.tool_name) or {}


# 导出
__all__ = [
    "EconometricTool"
]