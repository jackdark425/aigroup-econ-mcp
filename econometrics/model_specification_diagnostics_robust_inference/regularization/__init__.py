"""
正则化方法 (Regularization Methods) 模块

包括岭回归、LASSO和弹性网络等方法，用于处理多重共线性/高维数据
"""

from .regularization_model import (
    RegularizationResult,
    regularized_regression
)

__all__ = [
    "RegularizationResult",
    "regularized_regression"
]