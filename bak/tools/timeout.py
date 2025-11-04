"""


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
    """"""
    pass


class TimeoutManager:
    """
    
    
    """
    
    def __init__(self):
        self._timeout_config = {}
    
    def set_timeout_config(self, config: dict) -> None:
        """"""
        self._timeout_config = config
    
    def get_timeout_for_function(self, function_name: str, default: int = 60) -> int:
        """"""
        # 
        if function_name in self._timeout_config:
            return self._timeout_config[function_name]
        
        # 
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
        """"""
        try:
            if asyncio.iscoroutinefunction(func):
                # 
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            else:
                #  - 
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, 
                    lambda: self._execute_sync_with_timeout(func, timeout_seconds, *args, **kwargs)
                )
        except asyncio.TimeoutError:
            raise TimeoutError(f" '{model_name}'  ({timeout_seconds})")
    
    def _execute_sync_with_timeout(self, func: callable, timeout_seconds: int, *args, **kwargs):
        """"""
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
            raise TimeoutError(f" ({timeout_seconds})")
        
        if not exception_queue.empty():
            raise exception_queue.get()
        
        return result_queue.get()
    
    @contextmanager
    def timeout_context(self, seconds: int):
        """
        
        
        Args:
            seconds: 
        """
        def timeout_handler(signum, frame):
            raise TimeoutError(f" ({seconds})")
        
        # Unix
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # 
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
    
    async def async_timeout_context(self, seconds: int):
        """
        
        
        Args:
            seconds: 
        """
        try:
            await asyncio.wait_for(asyncio.sleep(0), timeout=seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(f" ({seconds})")


def timeout(seconds: int = 60):
    """
    
    
    Args:
        seconds: 
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = TimeoutManager()
            
            # Windows
            if hasattr(signal, 'SIGALRM'):
                # Unix
                with manager.timeout_context(seconds):
                    return func(*args, **kwargs)
            else:
                # Windows
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
                    raise TimeoutError(f" {func.__name__}  ({seconds})")
                
                if exception[0]:
                    raise exception[0]
                
                return result[0]
        
        return wrapper
    return decorator


def async_timeout(seconds: int = 60):
    """
    
    
    Args:
        seconds: 
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
                raise TimeoutError(f" {func.__name__}  ({seconds})")
        
        return wrapper
    return decorator


class ResourceMonitor:
    """
    
    CPU
    """
    
    def __init__(self):
        self._start_time = None
        self._peak_memory = 0
    
    @contextmanager
    def monitor_resources(self):
        """"""
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
            
            # 
            if memory_increase > 100:  # 100MB
                warnings.warn(
                    f":  {memory_increase:.2f}MB, "
                    f": {execution_time:.2f}"
                )


# 
global_timeout_manager = TimeoutManager()


# 
def with_timeout(seconds: int = 60):
    """"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            return async_timeout(seconds)(func)
        else:
            return timeout(seconds)(func)
    return decorator


def econometric_timeout(function_name: str = None):
    """
    
    
    """
    def decorator(func):
        name = function_name or func.__name__
        timeout_seconds = global_timeout_manager.get_timeout_for_function(name)
        
        if asyncio.iscoroutinefunction(func):
            return async_timeout(timeout_seconds)(func)
        else:
            return timeout(timeout_seconds)(func)
    
    return decorator


# 
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