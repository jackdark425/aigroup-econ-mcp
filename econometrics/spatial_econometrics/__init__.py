"""
空间计量经济学模块
处理空间依赖性和空间异质性

主要功能：
1. 空间权重矩阵构建
2. 空间自相关检验（Moran's I, Geary's C, Local LISA）
3. 空间回归模型（SAR, SEM, SDM）
4. 地理加权回归（GWR）
"""

# 空间权重矩阵
from .spatial_weights import (
    create_spatial_weights,
    SpatialWeightsResult
)

# 空间自相关检验
from .spatial_autocorrelation import (
    morans_i_test,
    gearys_c_test,
    local_morans_i,
    MoranIResult,
    GearysCResult,
    LocalMoranResult
)

# 空间回归模型
from .spatial_regression import (
    spatial_lag_model,
    spatial_error_model,
    SpatialRegressionResult
)

# 空间杜宾模型
from .spatial_durbin_model import (
    spatial_durbin_model,
    SpatialDurbinResult
)

# 地理加权回归
from .geographically_weighted_regression import (
    geographically_weighted_regression,
    GWRResult
)

__all__ = [
    # 空间权重
    'create_spatial_weights',
    'SpatialWeightsResult',
    # 空间自相关
    'morans_i_test',
    'gearys_c_test',
    'local_morans_i',
    'MoranIResult',
    'GearysCResult',
    'LocalMoranResult',
    # 空间回归
    'spatial_lag_model',
    'spatial_error_model',
    'SpatialRegressionResult',
    # 空间杜宾模型
    'spatial_durbin_model',
    'SpatialDurbinResult',
    # 地理加权回归
    'geographically_weighted_regression',
    'GWRResult'
]