"""
中介效应分析实现
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MediationResult(BaseModel):
    """中介效应分析结果"""
    method: str = Field(default="Mediation Analysis", description="使用的因果识别方法")
    direct_effect: float = Field(..., description="直接效应")
    indirect_effect: float = Field(..., description="间接效应（中介效应）")
    total_effect: float = Field(..., description="总效应")
    indirect_effect_std_error: float = Field(..., description="中介效应标准误")
    indirect_effect_p_value: float = Field(..., description="中介效应p值")
    n_observations: int = Field(..., description="观测数量")
    sobel_test_statistic: Optional[float] = Field(None, description="Sobel检验统计量")


def mediation_analysis(
    outcome: List[float],
    treatment: List[float],
    mediator: List[float],
    covariates: Optional[List[List[float]]] = None
) -> MediationResult:
    """
    中介效应分析（Baron-Kenny方法）
    
    中介效应分析用于识别和量化变量间因果路径中的中介机制。
    
    Args:
        outcome: 结果变量
        treatment: 处理变量
        mediator: 中介变量
        covariates: 协变量（可选）
        
    Returns:
        MediationResult: 中介效应分析结果
    """
    # 构建数据
    df = pd.DataFrame({
        'outcome': outcome,
        'treatment': treatment,
        'mediator': mediator
    })
    
    # 添加协变量
    if covariates:
        covariates_array = np.array(covariates)
        if covariates_array.ndim == 1:
            covariates_array = covariates_array.reshape(-1, 1)
        
        n_covariates = covariates_array.shape[1]
        for i in range(n_covariates):
            df[f'covariate_{i+1}'] = covariates_array[:, i]
    
    # 第一步：回归 mediator ~ treatment + covariates
    mediator_vars = ['treatment']
    if covariates:
        mediator_vars.extend([f'covariate_{i+1}' for i in range(n_covariates)])
    
    X_mediator = df[mediator_vars]
    X_mediator = sm.add_constant(X_mediator)
    y_mediator = df['mediator']
    
    mediator_model = sm.OLS(y_mediator, X_mediator)
    mediator_results = mediator_model.fit()
    
    # 提取处理变量对中介变量的效应 (alpha)
    alpha = mediator_results.params['treatment']
    alpha_se = mediator_results.bse['treatment']
    
    # 第二步：回归 outcome ~ treatment + mediator + covariates
    outcome_vars = ['treatment', 'mediator']
    if covariates:
        outcome_vars.extend([f'covariate_{i+1}' for i in range(n_covariates)])
    
    X_outcome = df[outcome_vars]
    X_outcome = sm.add_constant(X_outcome)
    y_outcome = df['outcome']
    
    outcome_model = sm.OLS(y_outcome, X_outcome)
    outcome_results = outcome_model.fit()
    
    # 提取直接效应 (beta2) 和中介变量效应 (beta1)
    direct_effect = outcome_results.params['treatment']  # 直接效应
    beta1 = outcome_results.params['mediator']  # 中介变量效应
    beta1_se = outcome_results.bse['mediator']
    
    # 计算间接效应（中介效应）
    indirect_effect = alpha * beta1
    
    # 计算总效应
    total_effect = direct_effect + indirect_effect
    
    # Sobel检验标准误
    indirect_effect_se = np.sqrt((alpha**2) * (beta1_se**2) + 
                                (beta1**2) * (alpha_se**2))
    
    # Sobel检验统计量
    sobel_stat = indirect_effect / indirect_effect_se if indirect_effect_se != 0 else 0
    
    # 中介效应的p值
    indirect_p_value = 2 * (1 - stats.norm.cdf(np.abs(sobel_stat)))
    
    return MediationResult(
        direct_effect=float(direct_effect),
        indirect_effect=float(indirect_effect),
        total_effect=float(total_effect),
        indirect_effect_std_error=float(indirect_effect_se),
        indirect_effect_p_value=float(indirect_p_value),
        n_observations=len(df),
        sobel_test_statistic=float(sobel_stat)
    )