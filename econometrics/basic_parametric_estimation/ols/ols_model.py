"""
普通最小二乘法 (OLS) 模型实现
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from scipy import stats

from tools.decorators import with_file_support_decorator as econometric_tool, validate_input


class OLSResult(BaseModel):
    """OLS回归结果"""
    coefficients: List[float] = Field(..., description="回归系数")
    std_errors: List[float] = Field(..., description="系数标准误")
    t_values: List[float] = Field(..., description="t统计量")
    p_values: List[float] = Field(..., description="p值")
    conf_int_lower: List[float] = Field(..., description="置信区间下界")
    conf_int_upper: List[float] = Field(..., description="置信区间上界")
    r_squared: float = Field(..., description="R方")
    adj_r_squared: float = Field(..., description="调整R方")
    f_statistic: float = Field(..., description="F统计量")
    f_p_value: float = Field(..., description="F统计量p值")
    n_obs: int = Field(..., description="观测数量")
    feature_names: List[str] = Field(..., description="特征名称")


@econometric_tool("ols_regression")
@validate_input(data_type="econometric")
def ols_regression(
    y_data: List[float],
    x_data: List[List[float]], 
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    confidence_level: float = 0.95
) -> OLSResult:
    """
    普通最小二乘法回归
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        feature_names: 特征名称
        constant: 是否包含常数项
        confidence_level: 置信水平
        
    Returns:
        OLSResult: OLS回归结果
    """
    # 转换为numpy数组
    y = np.array(y_data)
    X = np.array(x_data)
    
    # 添加常数项
    if constant:
        X = np.column_stack([np.ones(len(X)), X])
        if feature_names:
            feature_names = ["const"] + feature_names
        else:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    else:
        if not feature_names:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    
    # 执行OLS回归: β = (X'X)^(-1)X'y
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    
    # 计算预测值和残差
    y_pred = X @ beta
    residuals = y - y_pred
    
    # 计算各种统计量
    n = len(y)
    k = len(beta)
    df_resid = n - k
    
    # 残差平方和
    ssr = np.sum(residuals ** 2)
    
    # 总平方和
    sst = np.sum((y - np.mean(y)) ** 2)
    
    # 回归平方和
    ssr_reg = sst - ssr
    
    # 均方误差
    mse = ssr / df_resid
    
    # R方和调整R方
    r_squared = 1 - (ssr / sst)
    adj_r_squared = 1 - ((ssr / df_resid) / (sst / (n - 1)))
    
    # 系数标准误
    var_beta = mse * XtX_inv
    std_errors = np.sqrt(np.diag(var_beta))
    
    # t统计量和p值
    t_values = beta / std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df_resid))
    
    # 置信区间
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, df_resid)
    conf_int_lower = beta - t_critical * std_errors
    conf_int_upper = beta + t_critical * std_errors
    
    # F统计量
    f_statistic = (ssr_reg / (k - 1)) / mse
    f_p_value = 1 - stats.f.cdf(f_statistic, k - 1, df_resid)
    
    return OLSResult(
        coefficients=beta.tolist(),
        std_errors=std_errors.tolist(),
        t_values=t_values.tolist(),
        p_values=p_values.tolist(),
        conf_int_lower=conf_int_lower.tolist(),
        conf_int_upper=conf_int_upper.tolist(),
        r_squared=float(r_squared),
        adj_r_squared=float(adj_r_squared),
        f_statistic=float(f_statistic),
        f_p_value=float(f_p_value),
        n_obs=n,
        feature_names=feature_names
    )