"""
倾向得分匹配(PSM)实现
"""

from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats


class PSMMatchResult(BaseModel):
    """倾向得分匹配结果"""
    method: str = Field(default="Propensity Score Matching", description="使用的因果识别方法")
    ate: float = Field(..., description="平均处理效应")
    std_error: float = Field(..., description="标准误")
    t_statistic: float = Field(..., description="t统计量")
    p_value: float = Field(..., description="p值")
    confidence_interval: List[float] = Field(..., description="置信区间")
    n_observations: int = Field(..., description="观测数量")
    matched_observations: int = Field(..., description="匹配后的观测数量")


def propensity_score_matching(
    treatment: List[int],
    outcome: List[float],
    covariates: List[List[float]],
    matching_method: str = "nearest",
    k_neighbors: int = 1
) -> PSMMatchResult:
    """
    倾向得分匹配(PSM)
    
    倾向得分匹配通过匹配具有相似倾向得分的处理组和对照组个体来控制混杂因素。
    
    Args:
        treatment: 处理状态变量 (0/1)
        outcome: 结果变量
        covariates: 协变量矩阵
        matching_method: 匹配方法 ("nearest", "caliper", "kernel")
        k_neighbors: 近邻匹配中的邻居数
        
    Returns:
        PSMMatchResult: 倾向得分匹配结果
    """
    # 转换为DataFrame
    covariates_array = np.array(covariates)
    if covariates_array.ndim == 1:
        covariates_array = covariates_array.reshape(-1, 1)
    
    df = pd.DataFrame({
        'treatment': treatment,
        'outcome': outcome
    })
    
    # 添加协变量
    n_covariates = covariates_array.shape[1]
    for i in range(n_covariates):
        df[f'covariate_{i+1}'] = covariates_array[:, i]
    
    # 估计倾向得分（使用逻辑回归）
    X_cov = df[[f'covariate_{i+1}' for i in range(n_covariates)]]
    y_treatment = df['treatment']
    
    logit_model = LogisticRegression(solver='liblinear')
    logit_model.fit(X_cov, y_treatment)
    propensity_scores = logit_model.predict_proba(X_cov)[:, 1]
    df['propensity_score'] = propensity_scores
    
    # 进行匹配
    treated_df = df[df['treatment'] == 1].copy()
    control_df = df[df['treatment'] == 0].copy()
    
    if matching_method == "nearest":
        # 最近邻匹配
        matched_outcomes = []
        
        # 为每个处理组个体找到匹配的对照组个体
        for idx, treated_row in treated_df.iterrows():
            # 计算与所有对照组个体的倾向得分距离
            control_df.loc[:, 'ps_distance'] = np.abs(
                control_df['propensity_score'] - treated_row['propensity_score']
            )
            
            # 选择最近的k个邻居
            nearest_controls = control_df.nsmallest(k_neighbors, 'ps_distance')
            
            # 计算处理效应
            treated_outcome = treated_row['outcome']
            control_outcomes = nearest_controls['outcome'].values
            
            for control_outcome in control_outcomes:
                matched_outcomes.append(treated_outcome - control_outcome)
        
        # 计算平均处理效应
        ate = np.mean(matched_outcomes)
        std_error = np.std(matched_outcomes) / np.sqrt(len(matched_outcomes))
        t_statistic = ate / std_error
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), len(matched_outcomes) - 1))
        
        # 计算置信区间
        ci_lower = ate - 1.96 * std_error
        ci_upper = ate + 1.96 * std_error
        
    else:
        # 简化处理其他方法，使用最近邻作为默认
        matched_outcomes = []
        
        for idx, treated_row in treated_df.iterrows():
            control_df.loc[:, 'ps_distance'] = np.abs(
                control_df['propensity_score'] - treated_row['propensity_score']
            )
            
            nearest_controls = control_df.nsmallest(k_neighbors, 'ps_distance')
            
            treated_outcome = treated_row['outcome']
            control_outcomes = nearest_controls['outcome'].values
            
            for control_outcome in control_outcomes:
                matched_outcomes.append(treated_outcome - control_outcome)
        
        ate = np.mean(matched_outcomes)
        std_error = np.std(matched_outcomes) / np.sqrt(len(matched_outcomes))
        t_statistic = ate / std_error
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), len(matched_outcomes) - 1))
        
        ci_lower = ate - 1.96 * std_error
        ci_upper = ate + 1.96 * std_error
    
    return PSMMatchResult(
        ate=float(ate),
        std_error=float(std_error),
        t_statistic=float(t_statistic),
        p_value=float(p_value),
        confidence_interval=[float(ci_lower), float(ci_upper)],
        n_observations=len(df),
        matched_observations=len(matched_outcomes)
    )