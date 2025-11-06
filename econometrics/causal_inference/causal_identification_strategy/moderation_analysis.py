"""
调节效应分析实现
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import statsmodels.api as sm
from scipy import stats


class ModerationResult(BaseModel):
    """调节效应分析结果"""
    method: str = Field(default="Moderation Analysis", description="使用的因果识别方法")
    main_effect: float = Field(..., description="主要效应")
    moderator_effect: float = Field(..., description="调节变量效应")
    interaction_effect: float = Field(..., description="交互效应（调节效应）")
    main_effect_std_error: float = Field(..., description="主要效应标准误")
    moderator_effect_std_error: float = Field(..., description="调节变量效应标准误")
    interaction_effect_std_error: float = Field(..., description="交互效应标准误")
    main_effect_p_value: float = Field(..., description="主要效应p值")
    moderator_effect_p_value: float = Field(..., description="调节变量效应p值")
    interaction_effect_p_value: float = Field(..., description="交互效应p值")
    n_observations: int = Field(..., description="观测数量")
    r_squared: float = Field(..., description="模型R方")


def moderation_analysis(
    outcome: List[float],
    predictor: List[float],
    moderator: List[float],
    covariates: Optional[List[List[float]]] = None
) -> ModerationResult:
    """
    调节效应分析（交互项回归）
    
    调节效应分析用于检验一个变量是否影响另一个变量对结果的影响强度。
    
    Args:
        outcome: 结果变量
        predictor: 预测变量
        moderator: 调节变量
        covariates: 协变量（可选）
        
    Returns:
        ModerationResult: 调节效应分析结果
    """
    # 构建数据
    df = pd.DataFrame({
        'outcome': outcome,
        'predictor': predictor,
        'moderator': moderator
    })
    
    # 添加协变量
    if covariates:
        covariates_array = np.array(covariates)
        if covariates_array.ndim == 1:
            covariates_array = covariates_array.reshape(-1, 1)
        
        n_covariates = covariates_array.shape[1]
        for i in range(n_covariates):
            df[f'covariate_{i+1}'] = covariates_array[:, i]
    
    # 构造交互项
    df['interaction'] = df['predictor'] * df['moderator']
    
    # 构建回归模型
    vars_list = ['predictor', 'moderator', 'interaction']
    if covariates:
        vars_list.extend([f'covariate_{i+1}' for i in range(n_covariates)])
    
    X = df[vars_list]
    X = sm.add_constant(X)
    y = df['outcome']
    
    # OLS回归
    model = sm.OLS(y, X)
    results = model.fit()
    
    # 提取结果
    main_effect = results.params['predictor']
    moderator_effect = results.params['moderator']
    interaction_effect = results.params['interaction']
    
    main_effect_se = results.bse['predictor']
    moderator_effect_se = results.bse['moderator']
    interaction_effect_se = results.bse['interaction']
    
    main_effect_p = results.pvalues['predictor']
    moderator_effect_p = results.pvalues['moderator']
    interaction_effect_p = results.pvalues['interaction']
    
    r_squared = results.rsquared
    
    return ModerationResult(
        main_effect=float(main_effect),
        moderator_effect=float(moderator_effect),
        interaction_effect=float(interaction_effect),
        main_effect_std_error=float(main_effect_se),
        moderator_effect_std_error=float(moderator_effect_se),
        interaction_effect_std_error=float(interaction_effect_se),
        main_effect_p_value=float(main_effect_p),
        moderator_effect_p_value=float(moderator_effect_p),
        interaction_effect_p_value=float(interaction_effect_p),
        n_observations=len(df),
        r_squared=float(r_squared)
    )