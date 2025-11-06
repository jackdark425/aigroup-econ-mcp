"""
三重差分法 (DDD) 实现
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import statsmodels.api as sm
from scipy import stats


class TripeDifferenceResult(BaseModel):
    """三重差分法结果"""
    method: str = Field(default="Triple Difference", description="使用的因果识别方法")
    estimate: float = Field(..., description="因果效应估计值")
    std_error: float = Field(..., description="标准误")
    t_statistic: float = Field(..., description="t统计量")
    p_value: float = Field(..., description="p值")
    confidence_interval: List[float] = Field(..., description="置信区间")
    n_observations: int = Field(..., description="观测数量")


def triple_difference(
    outcome: List[float],
    treatment_group: List[int],
    time_period: List[int],
    cohort_group: List[int]
) -> TripeDifferenceResult:
    """
    三重差分法 (DDD)
    
    三重差分法通过引入第三个维度（如不同的队列组）来进一步控制混杂因素，
    提供比双重差分法更强的因果识别能力。
    
    Args:
        outcome: 结果变量
        treatment_group: 处理组虚拟变量 (0/1)
        time_period: 时间虚拟变量 (0/1)
        cohort_group: 队列组虚拟变量 (0/1)
        
    Returns:
        TripeDifferenceResult: 三重差分法结果
    """
    # 构建数据
    df = pd.DataFrame({
        'outcome': outcome,
        'treatment': treatment_group,
        'time': time_period,
        'cohort': cohort_group
    })
    
    # 构建交互项
    df['treatment_time'] = df['treatment'] * df['time']
    df['treatment_cohort'] = df['treatment'] * df['cohort']
    df['time_cohort'] = df['time'] * df['cohort']
    df['treatment_time_cohort'] = df['treatment'] * df['time'] * df['cohort']
    
    # 构建回归设计矩阵
    X_vars = ['treatment', 'time', 'cohort', 'treatment_time', 'treatment_cohort', 'time_cohort', 'treatment_time_cohort']
    X = df[X_vars]
    X = sm.add_constant(X)  # 添加常数项
    y = df['outcome']
    
    # OLS回归
    model = sm.OLS(y, X)
    results = model.fit()
    
    # 提取三重差分估计结果（三重交互项系数）
    coef = results.params['treatment_time_cohort']
    stderr = results.bse['treatment_time_cohort']
    tstat = results.tvalues['treatment_time_cohort']
    pval = results.pvalues['treatment_time_cohort']
    
    # 计算置信区间
    ci_lower = coef - 1.96 * stderr
    ci_upper = coef + 1.96 * stderr
    
    return TripeDifferenceResult(
        estimate=float(coef),
        std_error=float(stderr),
        t_statistic=float(tstat),
        p_value=float(pval),
        confidence_interval=[float(ci_lower), float(ci_upper)],
        n_observations=len(df)
    )