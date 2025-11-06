"""
控制函数法实现
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import statsmodels.api as sm
from scipy import stats


class ControlFunctionResult(BaseModel):
    """控制函数法结果"""
    method: str = Field(default="Control Function Approach", description="使用的因果识别方法")
    estimate: float = Field(..., description="因果效应估计值")
    std_error: float = Field(..., description="标准误")
    t_statistic: float = Field(..., description="t统计量")
    p_value: float = Field(..., description="p值")
    confidence_interval: List[float] = Field(..., description="置信区间")
    n_observations: int = Field(..., description="观测数量")
    endogeneity_test: Optional[dict] = Field(None, description="内生性检验结果")


def control_function_approach(
    y: List[float],
    x: List[float],
    z: List[List[float]],
    constant: bool = True
) -> ControlFunctionResult:
    """
    控制函数法
    
    控制函数法是一种解决内生性问题的方法，通过在第二阶段回归中加入第一阶段回归的残差来控制内生性。
    
    Args:
        y: 因变量
        x: 内生自变量
        z: 外生变量（包括工具变量和外生控制变量）
        constant: 是否包含常数项
        
    Returns:
        ControlFunctionResult: 控制函数法结果
    """
    # 转换为numpy数组
    y_array = np.array(y)
    x_array = np.array(x)
    z_array = np.array(z)
    
    if z_array.ndim == 1:
        z_array = z_array.reshape(-1, 1)
    
    n = len(y)
    
    # 第一阶段：将内生变量x对所有外生变量z回归
    if constant:
        Z = np.column_stack([np.ones(n), z_array])
    else:
        Z = z_array
    
    # 第一阶段回归
    first_stage_model = sm.OLS(x_array, Z)
    first_stage_results = first_stage_model.fit()
    
    # 获取第一阶段残差
    x_residuals = first_stage_results.resid
    
    # 第二阶段：将y对x和第一阶段残差回归
    if constant:
        X_second = np.column_stack([np.ones(n), x_array, x_residuals])
    else:
        X_second = np.column_stack([x_array, x_residuals])
    
    second_stage_model = sm.OLS(y_array, X_second)
    second_stage_results = second_stage_model.fit()
    
    # 提取x的系数作为因果效应估计
    # 如果有常数项，x是第2列；否则是第1列
    x_coef_idx = 1 if constant else 0
    coef = second_stage_results.params[x_coef_idx]
    stderr = second_stage_results.bse[x_coef_idx]
    tstat = second_stage_results.tvalues[x_coef_idx]
    pval = second_stage_results.pvalues[x_coef_idx]
    
    # 计算置信区间
    ci_lower = coef - 1.96 * stderr
    ci_upper = coef + 1.96 * stderr
    
    # 内生性检验（检验控制函数/残差项的系数是否显著）
    residual_coef_idx = 2 if constant else 1
    residual_coef = second_stage_results.params[residual_coef_idx]
    residual_stderr = second_stage_results.bse[residual_coef_idx]
    residual_tstat = second_stage_results.tvalues[residual_coef_idx]
    residual_pval = second_stage_results.pvalues[residual_coef_idx]
    
    endogeneity_test = {
        "residual_coefficient": float(residual_coef),
        "residual_std_error": float(residual_stderr),
        "t_statistic": float(residual_tstat),
        "p_value": float(residual_pval),
        "interpretation": "如果残差项系数显著，表明存在内生性问题"
    }
    
    return ControlFunctionResult(
        estimate=float(coef),
        std_error=float(stderr),
        t_statistic=float(tstat),
        p_value=float(pval),
        confidence_interval=[float(ci_lower), float(ci_upper)],
        n_observations=n,
        endogeneity_test=endogeneity_test
    )