"""
稳健标准误 (Robust Errors) 模块

处理异方差/自相关的稳健推断方法
"""

from .robust_errors_model import (
    RobustErrorsResult,
    robust_errors_regression
)

__all__ = [
    "RobustErrorsResult",
    "robust_errors_regression"
]