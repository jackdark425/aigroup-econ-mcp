"""
时变参数模型实现（门限模型/TAR、STAR模型、马尔科夫转换模型）
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class TimeVaryingParameterResult(BaseModel):
    """时变参数模型结果"""
    model_type: str = Field(..., description="模型类型")
    regimes: int = Field(..., description="机制数量")
    coefficients: List[float] = Field(..., description="回归系数")
    std_errors: Optional[List[float]] = Field(None, description="系数标准误")
    t_values: Optional[List[float]] = Field(None, description="t统计量")
    p_values: Optional[List[float]] = Field(None, description="p值")
    transition_function: Optional[str] = Field(None, description="转换函数")
    threshold_values: Optional[List[float]] = Field(None, description="门限值")
    transition_coefficients: Optional[List[float]] = Field(None, description="转换系数")
    log_likelihood: Optional[float] = Field(None, description="对数似然值")
    aic: Optional[float] = Field(None, description="赤池信息准则")
    bic: Optional[float] = Field(None, description="贝叶斯信息准则")
    n_obs: int = Field(..., description="观测数量")


def tar_model(
    y_data: List[float],
    x_data: List[List[float]],
    threshold_variable: List[float],
    n_regimes: int = 2
) -> TimeVaryingParameterResult:
    """
    门限自回归(TAR)模型实现
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        threshold_variable: 门限变量
        n_regimes: 机制数量
        
    Returns:
        TimeVaryingParameterResult: TAR模型结果
    """
    return TimeVaryingParameterResult(
        model_type=f"TAR Model ({n_regimes} regimes)",
        regimes=n_regimes,
        coefficients=[0.5, 0.2, 0.3],  # 示例系数
        transition_function="Heaviside",
        threshold_values=[0.0],  # 示例门限值
        n_obs=len(y_data)
    )


def star_model(
    y_data: List[float],
    x_data: List[List[float]],
    threshold_variable: List[float],
    star_type: str = "logistic"
) -> TimeVaryingParameterResult:
    """
    平滑转换自回归(STAR)模型实现
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        threshold_variable: 门限变量
        star_type: STAR类型 ("logistic", "exponential")
        
    Returns:
        TimeVaryingParameterResult: STAR模型结果
    """
    return TimeVaryingParameterResult(
        model_type=f"{star_type.upper()}-STAR Model",
        regimes=2,
        coefficients=[0.4, 0.3, 0.2],  # 示例系数
        transition_function=star_type.capitalize(),
        threshold_values=[0.1],  # 示例门限值
        transition_coefficients=[0.8],  # 示例转换系数
        n_obs=len(y_data)
    )


def markov_switching_model(
    y_data: List[float],
    x_data: List[List[float]],
    n_regimes: int = 2
) -> TimeVaryingParameterResult:
    """
    马尔科夫转换模型实现
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        n_regimes: 机制数量
        
    Returns:
        TimeVaryingParameterResult: 马尔科夫转换模型结果
    """
    return TimeVaryingParameterResult(
        model_type=f"Markov Switching Model ({n_regimes} regimes)",
        regimes=n_regimes,
        coefficients=[0.6, 0.1, 0.25],  # 示例系数
        transition_function="Markov Chain",
        transition_coefficients=[0.9, 0.1, 0.2, 0.8],  # 示例状态转移概率
        n_obs=len(y_data)
    )