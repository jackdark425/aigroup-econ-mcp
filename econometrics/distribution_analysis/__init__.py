"""
分布分析与分解方法模块
分析因变量的条件分布特征并进行各种分解分析
"""

from .oaxaca_blinder import (
    oaxaca_blinder_decomposition,
    OaxacaResult
)

from .variance_decomposition import (
    variance_decomposition,
    VarianceDecompositionResult
)

from .time_series_decomposition import (
    time_series_decomposition,
    TimeSeriesDecompositionResult
)

__all__ = [
    'oaxaca_blinder_decomposition',
    'OaxacaResult',
    'variance_decomposition',
    'VarianceDecompositionResult',
    'time_series_decomposition',
    'TimeSeriesDecompositionResult'
]