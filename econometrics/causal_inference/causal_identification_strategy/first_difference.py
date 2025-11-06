"""
一阶差分模型实现
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import statsmodels.api as sm
from scipy import stats


class FirstDifferenceResult(BaseModel):
    """一阶差分模型结果"""
    method: str = Field(default="First Difference Model", description="使用的因果识别方法")
    estimate: float = Field(..., description="因果效应估计值")
    std_error: float = Field(..., description="标准误")
    t_statistic: float = Field(..., description="t统计量")
    p_value: float = Field(..., description="p值")
    confidence_interval: List[float] = Field(..., description="置信区间")
    n_observations: int = Field(..., description="观测数量")


def first_difference_model(
    y: List[float],
    x: List[float],
    entity_ids: List[str]
) -> FirstDifferenceResult:
    """
    一阶差分模型
    
    一阶差分法通过差分操作消除不随时间变化的个体固定效应，常用于面板数据分析。
    
    Args:
        y: 因变量（时间序列）
        x: 自变量（时间序列）
        entity_ids: 个体标识符
        
    Returns:
        FirstDifferenceResult: 一阶差分模型结果
    """
    # 转换为DataFrame便于处理
    df = pd.DataFrame({
        'y': y,
        'x': x,
        'entity': entity_ids
    })
    
    # 按个体排序
    df = df.sort_values(['entity'])
    
    # 计算一阶差分
    df['y_diff'] = df.groupby('entity')['y'].diff()
    df['x_diff'] = df.groupby('entity')['x'].diff()
    
    # 删除NaN值（每组的第一行）
    df_diff = df.dropna()
    
    # 提取差分后的数据
    y_diff = df_diff['y_diff'].values
    x_diff = df_diff['x_diff'].values
    
    n = len(y_diff)
    
    # 添加常数项
    X = np.column_stack([np.ones(n), x_diff])
    
    # OLS回归
    model = sm.OLS(y_diff, X)
    results = model.fit()
    
    # 提取x_diff的系数作为因果效应估计
    coef = results.params[1]
    stderr = results.bse[1]
    tstat = results.tvalues[1]
    pval = results.pvalues[1]
    
    # 计算置信区间
    ci_lower = coef - 1.96 * stderr
    ci_upper = coef + 1.96 * stderr
    
    return FirstDifferenceResult(
        estimate=float(coef),
        std_error=float(stderr),
        t_statistic=float(tstat),
        p_value=float(pval),
        confidence_interval=[float(ci_lower), float(ci_upper)],
        n_observations=n
    )