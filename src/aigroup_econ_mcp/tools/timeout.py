"""
超时控制模块
为复杂计算任务提供超时控制和资源管理
"""

import asyncio
import signal
import threading
import time
from typing import Any, Callable, Optional, TypeVar, Union
from functools import wraps
from contextlib import contextmanager
import warnings

T = TypeVar('T')


class TimeoutError(Exception):
    """超时错误"""
    pass


class TimeoutManager:
    """
    超时管理器
    提供同步和异步的超时控制
    """
    
    def __init__(self):
        self._timeout_config = {}
    
    def set_timeout_config(self, config: dict) -> None:
        """设置超时配置"""
        self._timeout_config = config
    
    def get_timeout_for_function(self, function_name: str, default: int = 60) -> int:
        """获取函数的超时时间"""
        # 从配置中获取超时时间
        if function_name in self._timeout_config:
            return self._timeout_config[function_name]
        
        # 根据函数类型返回默认超时
        if function_name.startswith(('descriptive_', 'correlation_')):
            return 30
        elif function_name.startswith(('ols_', 'hypothesis_')):
            return 60
        elif function_name.startswith(('time_series_', 'panel_')):
            return 120
        elif function_name.startswith(('var_', 'vecm_', 'garch_')):
            return 180
        elif function_name.startswith(('random_forest_', 'gradient_boosting_')):
            return 300
        else:
            return default
    
    async def execute_with_timeout(self, model_name: str, timeout_seconds: int, func: callable, *args, **kwargs):
        """使用超时执行函数"""
        try:
            if asyncio.iscoroutinefunction(func):
                # 异步函数
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            else:
                # 同步函数 - 在线程池中执行
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, 
                    lambda: self._execute_sync_with_timeout(func, timeout_seconds, *args, **kwargs)
                )
        except asyncio.TimeoutError:
            raise TimeoutError(f"模型 '{model_name}' 执行超时 ({timeout_seconds}秒)")
    
    def _execute_sync_with_timeout(self, func: callable, timeout_seconds: int, *args, **kwargs):
        """同步函数超时执行"""
        import threading
        import queue
        
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def worker():
            try:
                result = func(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            raise TimeoutError(f"同步函数执行超时 ({timeout_seconds}秒)")
        
        if not exception_queue.empty():
            raise exception_queue.get()
        
        return result_queue.get()
    
    @contextmanager
    def timeout_context(self, seconds: int):
        """
        同步超时上下文管理器
        
        Args:
            seconds: 超时时间（秒）
        """
        def timeout_handler(signum, frame):
            raise TimeoutError(f"操作超时 ({seconds}秒)")
        
        # 设置信号处理（仅适用于Unix系统）
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # 取消警报
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
    
    async def async_timeout_context(self, seconds: int):
        """
        异步超时上下文管理器
        
        Args:
            seconds: 超时时间（秒）
        """
        try:
            await asyncio.wait_for(asyncio.sleep(0), timeout=seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(f"异步操作超时 ({seconds}秒)")


def timeout(seconds: int = 60):
    """
    同步函数超时装饰器
    
    Args:
        seconds: 超时时间（秒）
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = TimeoutManager()
            
            # 在Windows上使用线程实现超时
            if hasattr(signal, 'SIGALRM'):
                # Unix系统使用信号
                with manager.timeout_context(seconds):
                    return func(*args, **kwargs)
            else:
                # Windows系统使用线程
                result = [None]
                exception = [None]
                
                def target():
                    try:
                        result[0] = func(*args, **kwargs)
                    except Exception as e:
                        exception[0] = e
                
                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(seconds)
                
                if thread.is_alive():
                    raise TimeoutError(f"函数 {func.__name__} 执行超时 ({seconds}秒)")
                
                if exception[0]:
                    raise exception[0]
                
                return result[0]
        
        return wrapper
    return decorator


def async_timeout(seconds: int = 60):
    """
    异步函数超时装饰器
    
    Args:
        seconds: 超时时间（秒）
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs), 
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"异步函数 {func.__name__} 执行超时 ({seconds}秒)")
        
        return wrapper
    return decorator


class ResourceMonitor:
    """
    资源监控器
    监控内存和CPU使用情况
    """
    
    def __init__(self):
        self._start_time = None
        self._peak_memory = 0
    
    @contextmanager
    def monitor_resources(self):
        """监控资源使用的上下文管理器"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        self._start_time = time.time()
        initial_memory = process.memory_info().rss
        
        try:
            yield
        finally:
            current_memory = process.memory_info().rss
            memory_increase = (current_memory - initial_memory) / 1024 / 1024  # MB
            
            execution_time = time.time() - self._start_time
            
            # 记录资源使用情况
            if memory_increase > 100:  # 超过100MB
                warnings.warn(
                    f"高内存使用警告: 内存增加 {memory_increase:.2f}MB, "
                    f"执行时间: {execution_time:.2f}秒"
                )


# 全局超时管理器实例
global_timeout_manager = TimeoutManager()


# 便捷装饰器
def with_timeout(seconds: int = 60):
    """便捷超时装饰器，自动选择同步或异步"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            return async_timeout(seconds)(func)
        else:
            return timeout(seconds)(func)
    return decorator


def econometric_timeout(function_name: str = None):
    """
    计量经济学专用超时装饰器
    根据函数类型自动设置合适的超时时间
    """
    def decorator(func):
        name = function_name or func.__name__
        timeout_seconds = global_timeout_manager.get_timeout_for_function(name)
        
        if asyncio.iscoroutinefunction(func):
            return async_timeout(timeout_seconds)(func)
        else:
            return timeout(timeout_seconds)(func)
    
    return decorator


# 导出主要类和函数
__all__ = [
    "TimeoutError",
    "TimeoutManager", 
    "timeout",
    "async_timeout",
    "with_timeout",
    "econometric_timeout",
    "ResourceMonitor",
    "global_timeout_manager"
]