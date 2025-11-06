"""
事件研究法 (Event Study) 实现
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import statsmodels.api as sm
from scipy import stats


class EventStudyResult(BaseModel):
    """事件研究法结果"""
    method: str = Field(default="Event Study", description="使用的因果识别方法")
    estimates: List[float] = Field(..., description="各期效应估计值")
    std_errors: List[float] = Field(..., description="各期效应标准误")
    t_statistics: List[float] = Field(..., description="各期效应t统计量")
    p_values: List[float] = Field(..., description="各期效应p值")
    confidence_intervals: List[List[float]] = Field(..., description="各期效应置信区间")
    n_observations: int = Field(..., description="观测数量")
    event_time_periods: List[int] = Field(..., description="事件时间期列表")


def event_study(
    outcome: List[float],
    treatment: List[int],
    entity_ids: List[str],
    time_periods: List[str],
    event_time: List[int]
) -> EventStudyResult:
    """
    事件研究法 (Event Study)
    
    事件研究法通过分析处理前后多个时间点的效应，验证处理效应的动态变化模式。
    
    Args:
        outcome: 结果变量
        treatment: 处理状态变量
        entity_ids: 个体标识符
        time_periods: 时间标识符
        event_time: 相对于事件发生时间的时间标识（如-2, -1, 0, 1, 2）
        
    Returns:
        EventStudyResult: 事件研究法结果
    """
    # 构建数据
    df = pd.DataFrame({
        'outcome': outcome,
        'treatment': treatment,
        'entity': entity_ids,
        'time': time_periods,
        'event_time': event_time
    })
    
    # 创建时间虚拟变量
    time_dummies = pd.get_dummies(df['event_time'], prefix='time')
    df = pd.concat([df, time_dummies], axis=1)
    
    # 与处理状态交互
    for col in time_dummies.columns:
        df[f'{col}_treated'] = df[col] * df['treatment']
    
    # 构建回归设计矩阵
    interaction_vars = [col for col in df.columns if col.endswith('_treated')]
    X = df[interaction_vars]
    X = sm.add_constant(X)  # 添加常数项
    y = df['outcome']
    
    # OLS回归
    model = sm.OLS(y, X)
    results = model.fit()
    
    # 提取各期效应估计结果
    estimates = []
    std_errors = []
    t_statistics = []
    p_values = []
    confidence_intervals = []
    event_time_periods = []
    
    for col in interaction_vars:
        # 从列名中提取时间期数
        time_period = int(col.replace('time_', '').replace('_treated', ''))
        event_time_periods.append(time_period)
        
        coef = results.params[col]
        stderr = results.bse[col]
        tstat = results.tvalues[col]
        pval = results.pvalues[col]
        
        # 计算置信区间
        ci_lower = coef - 1.96 * stderr
        ci_upper = coef + 1.96 * stderr
        
        estimates.append(float(coef))
        std_errors.append(float(stderr))
        t_statistics.append(float(tstat))
        p_values.append(float(pval))
        confidence_intervals.append([float(ci_lower), float(ci_upper)])
    
    # 按时间期排序
    sorted_indices = np.argsort(event_time_periods)
    event_time_periods = [event_time_periods[i] for i in sorted_indices]
    estimates = [estimates[i] for i in sorted_indices]
    std_errors = [std_errors[i] for i in sorted_indices]
    t_statistics = [t_statistics[i] for i in sorted_indices]
    p_values = [p_values[i] for i in sorted_indices]
    confidence_intervals = [confidence_intervals[i] for i in sorted_indices]
    
    return EventStudyResult(
        estimates=estimates,
        std_errors=std_errors,
        t_statistics=t_statistics,
        p_values=p_values,
        confidence_intervals=confidence_intervals,
        n_observations=len(df),
        event_time_periods=event_time_periods
    )