"""
普通最小二乘法 (OLS) 模块
"""

from .ols_model import (
    OLSResult,
    ols_regression
)

__all__ = [
    "OLSResult",
    "ols_regression"
]