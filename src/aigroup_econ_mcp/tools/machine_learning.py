"""
机器学习集成模块 - 统一导出接口
提供基于scikit-learn的机器学习算法，用于经济数据分析

此模块作为统一入口，导出所有机器学习相关功能：
- ml_models: 数据模型定义
- ml_ensemble: 集成学习方法（随机森林、梯度提升树）
- ml_regularization: 正则化回归（Lasso、Ridge）
- ml_evaluation: 评估和比较功能
"""

# 导入数据模型
from .ml_models import (
    MLModelResult,
    RandomForestResult,
    GradientBoostingResult,
    RegularizedRegressionResult,
    CrossValidationResult,
    FeatureImportanceResult
)

# 导入集成学习方法
from .ml_ensemble import (
    random_forest_regression,
    gradient_boosting_regression
)

# 导入正则化回归
from .ml_regularization import (
    lasso_regression,
    ridge_regression
)

# 导入评估和比较功能
from .ml_evaluation import (
    cross_validation,
    feature_importance_analysis,
    compare_ml_models
)

# 导出所有公共接口
__all__ = [
    # 数据模型
    "MLModelResult",
    "RandomForestResult",
    "GradientBoostingResult",
    "RegularizedRegressionResult",
    "CrossValidationResult",
    "FeatureImportanceResult",
    # 集成学习
    "random_forest_regression",
    "gradient_boosting_regression",
    # 正则化回归
    "lasso_regression",
    "ridge_regression",
    # 评估和比较
    "cross_validation",
    "feature_importance_analysis",
    "compare_ml_models"
]