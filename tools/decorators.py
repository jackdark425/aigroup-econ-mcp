"""
工具装饰器模块
"""

from functools import wraps
from typing import Callable, Any


def with_file_support_decorator(tool_name: str):
    """
    支持文件输入的装饰器
    
    Args:
        tool_name: 工具名称
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 简化实现 - 直接调用原函数
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_input(data_type: str = "econometric"):
    """
    输入验证装饰器
    
    Args:
        data_type: 数据类型
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 简化实现 - 直接调用原函数
            return func(*args, **kwargs)
        return wrapper
    return decorator