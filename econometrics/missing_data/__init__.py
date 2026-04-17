"""
缺失数据处理模块
提供多种插补和处理缺失数据的方法
"""

from .imputation_methods import (
    MultipleImputationResult,
    SimpleImputationResult,
    multiple_imputation,
    simple_imputation,
)

__all__ = [
    "simple_imputation",
    "multiple_imputation",
    "SimpleImputationResult",
    "MultipleImputationResult",
]
