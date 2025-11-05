"""
稳健标准误 (Robust Errors) 模型实现
处理异方差/自相关的稳健推断方法
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from tools.decorators import with_file_support_decorator as econometric_tool, validate_input


class RobustErrorsResult(BaseModel):
    """稳健标准误回归结果"""
    coefficients: List[float] = Field(..., description="回归系数")
    robust_std_errors: List[float] = Field(..., description="稳健标准误")
    t_values: List[float] = Field(..., description="t统计量 (基于稳健标准误)")
    p_values: List[float] = Field(..., description="p值 (基于稳健标准误)")
    conf_int_lower: List[float] = Field(..., description="置信区间下界 (基于稳健标准误)")
    conf_int_upper: List[float] = Field(..., description="置信区间上界 (基于稳健标准误)")
    r_squared: float = Field(..., description="R方")
    adj_r_squared: float = Field(..., description="调整R方")
    f_statistic: float = Field(..., description="F统计量")
    f_p_value: float = Field(..., description="F统计量p值")
    n_obs: int = Field(..., description="观测数量")
    feature_names: List[str] = Field(..., description="特征名称")


@econometric_tool("robust_errors_regression")
@validate_input(data_type="econometric")
def robust_errors_regression(
    y_data: List[float],
    x_data: List[List[float]], 
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    confidence_level: float = 0.95,
    cov_type: str = "HC1"
) -> RobustErrorsResult:
    """
    使用稳健标准误的回归分析（处理异方差性）
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        feature_names: 特征名称
        constant: 是否包含常数项
        confidence_level: 置信水平
        cov_type: 协方差矩阵类型 ('HC0', 'HC1', 'HC2', 'HC3')
        
    Returns:
        RobustErrorsResult: 稳健标准误回归结果
    """
    # 转换为numpy数组
    y = np.asarray(y_data, dtype=np.float64)
    X = np.asarray(x_data, dtype=np.float64)
    
    # 添加常数项
    if constant:
        X = sm.add_constant(X)
        if feature_names:
            feature_names = ["const"] + feature_names
        else:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    else:
        if not feature_names:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    
    # 检查数据维度
    n, k = X.shape
    if n <= k:
        raise ValueError(f"观测数量({n})必须大于变量数量({k})")
    
    # 使用statsmodels执行OLS回归
    try:
        model = sm.OLS(y, X)
        results = model.fit(cov_type=cov_type)
    except Exception as e:
        # 如果出现问题，使用更稳健的方法
        try:
            model = sm.OLS(y, X)
            results = model.fit(cov_type='HC1')
        except Exception:
            raise ValueError(f"无法拟合模型: {str(e)}")
    
    # 提取结果
    coefficients = results.params.tolist()
    robust_std_errors = results.bse.tolist()
    t_values = results.tvalues.tolist()
    p_values = results.pvalues.tolist()
    
    # 计算置信区间
    alpha = 1 - confidence_level
    conf_int = results.conf_int(alpha=alpha)
    conf_int_lower = conf_int[:, 0].tolist()
    conf_int_upper = conf_int[:, 1].tolist()
    
    # 其他统计量
    r_squared = float(results.rsquared)
    adj_r_squared = float(results.rsquared_adj)
    
    # F统计量
    f_statistic = float(results.fvalue) if not np.isnan(results.fvalue) else 0.0
    f_p_value = float(results.f_pvalue) if not np.isnan(results.f_pvalue) else 1.0
    
    return RobustErrorsResult(
        coefficients=coefficients,
        robust_std_errors=robust_std_errors,
        t_values=t_values,
        p_values=p_values,
        conf_int_lower=conf_int_lower,
        conf_int_upper=conf_int_upper,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        f_statistic=f_statistic,
        f_p_value=f_p_value,
        n_obs=int(results.nobs),
        feature_names=feature_names
    )