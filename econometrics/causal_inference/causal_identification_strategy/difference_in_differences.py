"""
双重差分法 (DID) 实现
"""

from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
import statsmodels.api as sm


class DIDResult(BaseModel):
    """双重差分法结果"""
    method: str = Field(default="Difference-in-Differences", description="使用的因果识别方法")
    estimate: float = Field(..., description="因果效应估计值")
    std_error: float = Field(..., description="标准误")
    t_statistic: float = Field(..., description="t统计量")
    p_value: float = Field(..., description="p值")
    confidence_interval: List[float] = Field(..., description="置信区间")
    n_observations: int = Field(..., description="观测数量")
    parallel_trend_test: Optional[Dict[str, Any]] = Field(None, description="平行趋势检验")


def difference_in_differences(
    treatment: List[int],
    time_period: List[int],
    outcome: List[float],
    covariates: Optional[List[List[float]]] = None
) -> DIDResult:
    """
    双重差分法 (DID)
    
    使用statsmodels实现双重差分法，评估处理效应。
    
    Args:
        treatment: 处理组虚拟变量 (0/1)
        time_period: 时间虚拟变量 (0/1)
        outcome: 结果变量
        covariates: 协变量
        
    Returns:
        DIDResult: 双重差分法结果
    """
    # 构建数据
    data = {
        'treatment': treatment,
        'time': time_period,
        'outcome': outcome
    }
    
    # 添加协变量
    if covariates:
        covariates_array = np.array(covariates)
        if covariates_array.ndim == 1:
            covariates_array = covariates_array.reshape(-1, 1)
        
        k_cov = covariates_array.shape[1]
        for i in range(k_cov):
            data[f"covariate_{i+1}"] = covariates_array[:, i]
    
    df = pd.DataFrame(data)
    
    # 构建交互项
    df['treatment_time'] = df['treatment'] * df['time']
    
    # 构建回归公式
    independent_vars = ['treatment', 'time', 'treatment_time']
    if covariates:
        independent_vars.extend([f"covariate_{i+1}" for i in range(k_cov)])
    
    # 添加常数项
    df['const'] = 1
    independent_vars = ['const'] + independent_vars
    
    # 使用statsmodels进行OLS回归
    X = df[independent_vars]
    y = df['outcome']
    
    model = sm.OLS(y, X)
    results = model.fit()
    
    # 提取DID估计结果（交互项系数）
    coef = results.params['treatment_time']
    stderr = results.bse['treatment_time']
    tstat = results.tvalues['treatment_time']
    pval = results.pvalues['treatment_time']
    
    # 计算置信区间
    ci_lower = coef - 1.96 * stderr
    ci_upper = coef + 1.96 * stderr
    
    # 平行趋势检验（简化处理）
    # 这里只是一个示例，实际的平行趋势检验需要更多的前期数据
    parallel_trend = {
        "description": "Simplified parallel trend test - full test requires pre-treatment periods"
    }
    
    return DIDResult(
        estimate=float(coef),
        std_error=float(stderr),
        t_statistic=float(tstat),
        p_value=float(pval),
        confidence_interval=[float(ci_lower), float(ci_upper)],
        n_observations=len(df),
        parallel_trend_test=parallel_trend
    )