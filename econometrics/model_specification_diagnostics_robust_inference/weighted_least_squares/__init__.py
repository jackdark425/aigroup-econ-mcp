"""
加权最小二乘法 (Weighted Least Squares) 模块

WLS方法实现
"""

from .wls_model import (
    WLSResult,
    wls_regression
)

__all__ = [
    "WLSResult",
    "wls_regression"
]