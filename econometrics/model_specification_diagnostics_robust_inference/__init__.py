"""
模型设定、诊断与稳健推断 (Model Specification, Diagnostics and Robust Inference)

当基础模型的理想假设不成立时，修正模型或调整推断；对模型进行诊断和选择。

主要方法包括：
- 稳健标准误（处理异方差/自相关）
- 广义最小二乘法 (GLS)
- 加权最小二乘法 (WLS)
- 岭回归/LASSO/弹性网络（处理多重共线性/高维数据）
- 联立方程模型（处理双向因果关系）

模型诊断：
- 异方差检验（White、Breusch-Pagan）
- 自相关检验（Durbin-Watson、Ljung-Box）
- 正态性检验（Jarque-Bera）
- 多重共线性诊断（VIF）
- 内生性检验（Durbin-Wu-Hausman）
- 残差诊断、影响点分析

模型选择：
- 信息准则（AIC/BIC/HQIC）
- 交叉验证（K折、留一法）
- 格兰杰因果检验
"""

# 导入子模块
from .robust_errors import (
    RobustErrorsResult,
    robust_errors_regression
)

from .diagnostic_tests import (
    DiagnosticTestsResult,
    diagnostic_tests
)

from .model_selection import (
    ModelSelectionResult,
    model_selection_criteria
)

from .generalized_least_squares import (
    GLSResult,
    gls_regression
)

from .weighted_least_squares import (
    WLSResult,
    wls_regression
)

from .regularization import (
    RegularizationResult,
    regularized_regression
)

from .simultaneous_equations import (
    SimultaneousEquationsResult,
    two_stage_least_squares
)

__all__ = [
    "RobustErrorsResult",
    "robust_errors_regression",
    "DiagnosticTestsResult", 
    "diagnostic_tests",
    "ModelSelectionResult",
    "model_selection_criteria",
    "GLSResult",
    "gls_regression",
    "WLSResult",
    "wls_regression",
    "RegularizationResult",
    "regularized_regression",
    "SimultaneousEquationsResult",
    "two_stage_least_squares"
]