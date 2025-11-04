"""
广义最小二乘法 (Generalized Least Squares) 模块

GLS方法实现
"""

from .gls_model import (
    GLSResult,
    gls_regression
)

__all__ = [
    "GLSResult",
    "gls_regression"
]