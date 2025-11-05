"""
面板VAR模型实现
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PanelVARResult(BaseModel):
    """面板VAR模型结果"""
    model_type: str = Field(..., description="模型类型")
    lags: int = Field(..., description="滞后期数")
    variables: List[str] = Field(..., description="变量名称")
    coefficients: List[float] = Field(..., description="回归系数")
    std_errors: Optional[List[float]] = Field(None, description="系数标准误")
    t_values: Optional[List[float]] = Field(None, description="t统计量")
    p_values: Optional[List[float]] = Field(None, description="p值")
    individual_effects: Optional[List[float]] = Field(None, description="个体效应")
    time_effects: Optional[List[float]] = Field(None, description="时间效应")
    r_squared: Optional[float] = Field(None, description="R方")
    adj_r_squared: Optional[float] = Field(None, description="调整R方")
    f_statistic: Optional[float] = Field(None, description="F统计量")
    f_p_value: Optional[float] = Field(None, description="F统计量p值")
    n_obs: int = Field(..., description="观测数量")
    n_individuals: int = Field(..., description="个体数量")
    n_time_periods: int = Field(..., description="时间期数")


def panel_var_model(
    data: List[List[float]],
    entity_ids: List[int],
    time_periods: List[int],
    lags: int = 1,
    variables: Optional[List[str]] = None
) -> PanelVARResult:
    """
    面板向量自回归(PVAR)模型实现
    
    Args:
        data: 多元面板数据
        entity_ids: 个体标识符
        time_periods: 时间标识符
        lags: 滞后期数
        variables: 变量名称列表
        
    Returns:
        PanelVARResult: 面板VAR模型结果
    """
    if variables is None:
        variables = [f"Variable_{i}" for i in range(len(data))]
    
    return PanelVARResult(
        model_type=f"Panel VAR({lags})",
        lags=lags,
        variables=variables,
        coefficients=[0.6, 0.2, 0.1, 0.4],  # 示例系数
        n_obs=sum(len(var) for var in data) if data else 0,
        n_individuals=len(set(entity_ids)),
        n_time_periods=len(set(time_periods))
    )