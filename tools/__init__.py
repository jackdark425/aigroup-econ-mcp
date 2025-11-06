"""
工具模块初始化文件
"""

from .data_loader import DataLoader
from .output_formatter import OutputFormatter
from .econometrics_adapter import EconometricsAdapter

# 时间序列和面板数据工具
from .time_series_panel_data_adapter import TimeSeriesPanelDataAdapter
from .time_series_panel_data_tools import (
    arima_model,
    exponential_smoothing_model,
    garch_model,
    unit_root_tests,
    var_svar_model,
    cointegration_analysis,
    dynamic_panel_model
)

# 因果推断工具适配器
from .causal_inference_adapter import (
    did_adapter,
    iv_adapter,
    psm_adapter,
    fixed_effects_adapter,
    random_effects_adapter,
    rdd_adapter,
    synthetic_control_adapter,
    event_study_adapter,
    triple_difference_adapter,
    mediation_adapter,
    moderation_adapter,
    control_function_adapter,
    first_difference_adapter
)

# 机器学习工具适配器
from .machine_learning_adapter import (
    random_forest_adapter,
    gradient_boosting_adapter,
    svm_adapter,
    neural_network_adapter,
    kmeans_clustering_adapter,
    hierarchical_clustering_adapter,
    double_ml_adapter,
    causal_forest_adapter
)

# 微观计量模型工具适配器
from .microecon_adapter import (
    logit_adapter,
    probit_adapter,
    multinomial_logit_adapter,
    poisson_adapter,
    negative_binomial_adapter,
    tobit_adapter,
    heckman_adapter
)

# 保持向后兼容性
ols_adapter = EconometricsAdapter.ols_regression
mle_adapter = EconometricsAdapter.mle_estimation
gmm_adapter = EconometricsAdapter.gmm_estimation

__all__ = [
    "DataLoader",
    "OutputFormatter",
    "EconometricsAdapter",
    "TimeSeriesPanelDataAdapter",
    
    # 基础工具
    "ols_adapter",
    "mle_adapter",
    "gmm_adapter",
    
    # 时间序列和面板数据工具
    "arima_model",
    "exponential_smoothing_model",
    "garch_model",
    "unit_root_tests",
    "var_svar_model",
    "cointegration_analysis",
    "dynamic_panel_model",
    
    # 因果推断工具
    "did_adapter",
    "iv_adapter",
    "psm_adapter",
    "fixed_effects_adapter",
    "random_effects_adapter",
    "rdd_adapter",
    "synthetic_control_adapter",
    "event_study_adapter",
    "triple_difference_adapter",
    "mediation_adapter",
    "moderation_adapter",
    "control_function_adapter",
    "first_difference_adapter",
    
    # 机器学习工具
    "random_forest_adapter",
    "gradient_boosting_adapter",
    "svm_adapter",
    "neural_network_adapter",
    "kmeans_clustering_adapter",
    "hierarchical_clustering_adapter",
    "double_ml_adapter",
    "causal_forest_adapter",
    
    # 微观计量模型工具
    "logit_adapter",
    "probit_adapter",
    "multinomial_logit_adapter",
    "poisson_adapter",
    "negative_binomial_adapter",
    "tobit_adapter",
    "heckman_adapter"
]