"""
非参数与半参数方法模块
放宽函数形式的线性或参数化假设
"""

from .gam_model import GAMResult, gam_model
from .kernel_regression import KernelRegressionResult, kernel_regression
from .quantile_regression import QuantileRegressionResult, quantile_regression
from .spline_regression import SplineRegressionResult, spline_regression

__all__ = [
    "kernel_regression",
    "KernelRegressionResult",
    "quantile_regression",
    "QuantileRegressionResult",
    "spline_regression",
    "SplineRegressionResult",
    "gam_model",
    "GAMResult",
]
