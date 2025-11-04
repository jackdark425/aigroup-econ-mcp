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
        X = np.column_stack([np.ones(len(X)), X])
        if feature_names:
            feature_names = ["const"] + feature_names
        else:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    else:
        if not feature_names:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    
    # 执行OLS回归: β = (X'X)^(-1)X'y
    try:
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        beta = XtX_inv @ X.T @ y
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，添加正则化项
        XtX = X.T @ X
        k_reg = X.shape[1]
        XtX_reg = XtX + np.eye(k_reg) * 1e-10 * np.trace(XtX) / k_reg
        XtX_inv = np.linalg.inv(XtX_reg)
        beta = XtX_inv @ X.T @ y
    
    # 计算预测值和残差
    y_pred = X @ beta
    residuals = y - y_pred
    
    # 计算各种统计量
    n = len(y)
    k = len(beta)
    df_resid = max(n - k, 1)  # 避免自由度为0或负数
    
    # 残差平方和
    ssr = np.sum(residuals ** 2)
    
    # 总平方和
    y_mean = np.mean(y)
    sst = np.sum((y - y_mean) ** 2)
    
    # 回归平方和
    ssr_reg = max(sst - ssr, 0)  # 避免负数
    
    # 均方误差
    mse = ssr / df_resid if df_resid > 0 else ssr
    
    # R方和调整R方
    r_squared = 1 - (ssr / sst) if sst > 1e-10 else 0  # 避免除以接近零的数
    adj_r_squared = 1 - ((ssr / df_resid) / (sst / max(n - 1, 1))) if sst > 1e-10 and n > k else 0
    
    # 计算稳健标准误
    # White异方差一致协方差估计
    if cov_type == "HC0":
        # HC0: 原始White估计
        squared_residuals = residuals ** 2
    elif cov_type == "HC1":
        # HC1: 自由度调整的White估计
        squared_residuals = residuals ** 2 * n / max(df_resid, 1)
    elif cov_type == "HC2":
        # HC2: 杠杆值调整的White估计
        try:
            h = np.diag(X @ XtX_inv @ X.T)
            h = np.clip(h, 0, 0.999)  # 避免杠杆值超过1
            squared_residuals = (residuals ** 2) / (1 - h)
        except:
            # 如果计算失败，回退到HC1
            squared_residuals = residuals ** 2 * n / max(df_resid, 1)
    elif cov_type == "HC3":
        # HC3: 杠杆值调整的White估计（更稳健）
        try:
            h = np.diag(X @ XtX_inv @ X.T)
            h = np.clip(h, 0, 0.999)  # 避免杠杆值超过1
            squared_residuals = (residuals ** 2) / ((1 - h) ** 2)
        except:
            # 如果计算失败，回退到HC1
            squared_residuals = residuals ** 2 * n / max(df_resid, 1)
    else:
        # 默认使用HC1
        squared_residuals = residuals ** 2 * n / max(df_resid, 1)
    
    # 构建稳健协方差矩阵 - 使用更高效的计算方法
    try:
        # 使用外积形式计算，避免循环
        weighted_residuals = squared_residuals[:, np.newaxis, np.newaxis]
        X_expanded = X[:, :, np.newaxis]
        X_transposed_expanded = X[:, np.newaxis, :]
        XeX = np.sum(weighted_residuals * X_expanded * X_transposed_expanded, axis=0)
        
        robust_cov = XtX_inv @ XeX @ XtX_inv
        robust_std_errors = np.sqrt(np.maximum(np.diag(robust_cov), 0))  # 避免负方差
    except:
        # 如果高效方法失败，使用循环方法
        XeX = np.zeros((k, k))
        for i in range(n):
            Xi = X[i, :].reshape(-1, 1)
            ei = squared_residuals[i]
            XeX += ei * (Xi @ Xi.T)
        
        robust_cov = XtX_inv @ XeX @ XtX_inv
        robust_std_errors = np.sqrt(np.maximum(np.diag(robust_cov), 0))  # 避免负方差
    
    # 避免标准误为零
    robust_std_errors = np.maximum(robust_std_errors, 1e-12)
    
    # t统计量和p值 (基于稳健标准误)
    t_values = np.divide(beta, robust_std_errors, out=np.zeros_like(beta), where=robust_std_errors!=0)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df_resid))
    
    # 置信区间 (基于稳健标准误)
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, df_resid)
    conf_int_lower = beta - t_critical * robust_std_errors
    conf_int_upper = beta + t_critical * robust_std_errors
    
    # F统计量 (使用经典标准误)
    f_statistic = (ssr_reg / max(k - 1, 1)) / max(mse, 1e-10) if k > 1 else 0
    f_p_value = 1 - stats.f.cdf(f_statistic, max(k - 1, 1), df_resid) if k > 1 else 1
    
    return RobustErrorsResult(
        coefficients=beta.tolist(),
        robust_std_errors=robust_std_errors.tolist(),
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