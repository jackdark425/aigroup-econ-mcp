"""
非参数与半参数方法模块
放宽函数形式的线性或参数化假设
"""

from .kernel_regression import (
    kernel_regression,
    KernelRegressionResult
)

from .quantile_regression import (
    quantile_regression,
    QuantileRegressionResult
)

from .spline_regression import (
    spline_regression,
    SplineRegressionResult
)

from .gam_model import (
    gam_model,
    GAMResult
)

__all__ = [
    'kernel_regression',
    'KernelRegressionResult',
    'quantile_regression',
    'QuantileRegressionResult',
    'spline_regression',
    'SplineRegressionResult',
    'gam_model',
    'GAMResult'
]