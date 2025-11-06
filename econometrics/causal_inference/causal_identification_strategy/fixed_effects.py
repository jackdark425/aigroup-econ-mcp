"""
面板数据固定效应模型实现
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
import statsmodels.api as sm
from linearmodels.panel import PanelOLS


class FixedEffectsResult(BaseModel):
    """固定效应模型结果"""
    method: str = Field(default="Fixed Effects Model", description="使用的因果识别方法")
    estimate: float = Field(..., description="因果效应估计值")
    std_error: float = Field(..., description="标准误")
    t_statistic: float = Field(..., description="t统计量")
    p_value: float = Field(..., description="p值")
    confidence_interval: List[float] = Field(..., description="置信区间")
    n_observations: int = Field(..., description="观测数量")
    n_entities: int = Field(..., description="个体数量")
    n_time_periods: int = Field(..., description="时间期数")


def fixed_effects_model(
    y: List[float],
    x: List[List[float]],
    entity_ids: List[str],
    time_periods: List[str],
    constant: bool = True
) -> FixedEffectsResult:
    """
    固定效应模型
    
    使用linearmodels.panel.PanelOLS实现固定效应模型。
    
    Args:
        y: 因变量
        x: 自变量
        entity_ids: 个体标识符
        time_periods: 时间标识符
        constant: 是否包含常数项
        
    Returns:
        FixedEffectsResult: 固定效应模型结果
    """
    # 转换为DataFrame
    x_array = np.array(x)
    if x_array.ndim == 1:
        x_array = x_array.reshape(-1, 1)
    
    # 创建多重索引面板数据
    df = pd.DataFrame({
        'y': y,
        'entity': entity_ids,
        'time': [int(t.split('_')[1]) if isinstance(t, str) and '_' in t else i 
                for i, t in enumerate(time_periods)]  # 处理字符串格式的时间
    })
    
    # 添加自变量
    k_x = x_array.shape[1]
    for i in range(k_x):
        df[f'x{i+1}'] = x_array[:, i]
    
    # 设置多重索引
    df = df.set_index(['entity', 'time'])
    
    # 定义因变量和自变量
    dependent = df['y']
    explanatory_vars = [f'x{i+1}' for i in range(k_x)]
    explanatory = df[explanatory_vars]
    
    # 使用linearmodels进行固定效应估计
    model = PanelOLS(dependent, explanatory, entity_effects=True)
    results = model.fit()
    
    # 提取主要变量的估计结果（假设关注最后一个变量）
    target_var = f'x{k_x}'
    coef = results.params[target_var]
    stderr = results.std_errors[target_var]
    tstat = results.tstats[target_var]
    pval = results.pvalues[target_var]
    
    # 计算置信区间
    ci_lower = coef - 1.96 * stderr
    ci_upper = coef + 1.96 * stderr
    
    # 计算实体和时间期数
    n_entities = len(df.index.get_level_values('entity').unique())
    n_time_periods = len(df.index.get_level_values('time').unique())
    
    return FixedEffectsResult(
        estimate=float(coef),
        std_error=float(stderr),
        t_statistic=float(tstat),
        p_value=float(pval),
        confidence_interval=[float(ci_lower), float(ci_upper)],
        n_observations=len(df),
        n_entities=n_entities,
        n_time_periods=n_time_periods
    )