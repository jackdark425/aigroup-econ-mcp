"""
断点回归设计 (RDD) 实现
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures


class RDDResult(BaseModel):
    """断点回归设计结果"""
    method: str = Field(default="Regression Discontinuity Design", description="使用的因果识别方法")
    estimate: float = Field(..., description="因果效应估计值")
    std_error: float = Field(..., description="标准误")
    t_statistic: float = Field(..., description="t统计量")
    p_value: float = Field(..., description="p值")
    confidence_interval: List[float] = Field(..., description="置信区间")
    n_observations: int = Field(..., description="观测数量")
    bandwidth: Optional[float] = Field(None, description="使用的带宽")
    polynomial_order: Optional[int] = Field(None, description="多项式阶数")
    discontinuity_location: Optional[float] = Field(None, description="断点位置")


def regression_discontinuity(
    running_variable: List[float],
    outcome: List[float],
    cutoff: float,
    bandwidth: Optional[float] = None,
    polynomial_order: int = 1
) -> RDDResult:
    """
    断点回归设计 (RDD)
    
    使用statsmodels实现断点回归设计，评估在断点处的处理效应。
    
    Args:
        running_variable: 运行变量
        outcome: 结果变量
        cutoff: 断点值
        bandwidth: 带宽
        polynomial_order: 多项式阶数
        
    Returns:
        RDDResult: 断点回归设计结果
    """
    # 转换为numpy数组
    running_array = np.array(running_variable)
    outcome_array = np.array(outcome)
    
    # 如果未指定带宽，使用默认方法计算
    if bandwidth is None:
        # 使用数据标准差的四分之一作为默认带宽
        bandwidth = 0.25 * np.std(running_array)
    
    # 筛选带宽内的观测
    mask = np.abs(running_array - cutoff) <= bandwidth
    running_local = running_array[mask]
    outcome_local = outcome_array[mask]
    
    # 构造处理变量（运行变量是否大于cutoff）
    treatment = (running_local >= cutoff).astype(int)
    
    # 构造多项式项
    poly = PolynomialFeatures(degree=polynomial_order, include_bias=False)
    running_poly = poly.fit_transform(running_local.reshape(-1, 1))
    
    # 构建设计矩阵
    X = np.column_stack([np.ones(len(running_local)), treatment, running_poly])
    
    # 使用statsmodels进行OLS回归
    model = sm.OLS(outcome_local, X)
    results = model.fit()
    
    # 提取处理效应（处理变量系数）
    coef = results.params[1]  # treatment变量是第2列（索引为1）
    stderr = results.bse[1]
    tstat = results.tvalues[1]
    pval = results.pvalues[1]
    
    # 计算置信区间
    ci_lower = coef - 1.96 * stderr
    ci_upper = coef + 1.96 * stderr
    
    return RDDResult(
        estimate=float(coef),
        std_error=float(stderr),
        t_statistic=float(tstat),
        p_value=float(pval),
        confidence_interval=[float(ci_lower), float(ci_upper)],
        n_observations=len(running_local),
        bandwidth=bandwidth,
        polynomial_order=polynomial_order,
        discontinuity_location=cutoff
    )