"""
加权最小二乘法 (Weighted Least Squares, WLS) 模型实现
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from scipy import stats

from tools.decorators import with_file_support_decorator as econometric_tool, validate_input


class WLSResult(BaseModel):
    """WLS回归结果"""
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
    weights: List[float] = Field(..., description="使用的权重")


@econometric_tool("wls_regression")
@validate_input(data_type="econometric")
def wls_regression(
    y_data: List[float],
    x_data: List[List[float]], 
    weights: List[float],
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    confidence_level: float = 0.95
) -> WLSResult:
    """
    加权最小二乘法回归
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        weights: 权重列表（与观测值一一对应）
        feature_names: 特征名称
        constant: 是否包含常数项
        confidence_level: 置信水平
        
    Returns:
        WLSResult: WLS回归结果
    """
    # 转换为numpy数组
    y = np.asarray(y_data, dtype=np.float64)
    X = np.asarray(x_data, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    # 检查数据维度
    if len(w) != len(y):
        raise ValueError("权重数量必须与观测值数量相同")
    
    # 检查权重是否为正数
    if np.any(w <= 0):
        raise ValueError("所有权重必须为正数")
    
    # 处理极小权重，避免数值问题
    w_min = np.min(w)
    if w_min < 1e-10:
        # 将极小权重设置为最小权重的某个比例
        w = np.maximum(w, w_min * 1e-3)
    
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
    
    n, k = X.shape
    
    # 检查是否有足够的自由度
    if n <= k:
        raise ValueError(f"观测数量({n})必须大于变量数量({k})")
    
    # 构造权重矩阵
    W = np.diag(w)
    
    # 执行WLS回归: β = (X'WX)^(-1)X'Wy
    try:
        XtWX = X.T @ W @ X
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，添加一个小的正数进行正则化
        XtWX = X.T @ W @ X
        # 使用迹来确定正则化强度
        reg_strength = 1e-10 * np.trace(XtWX) / k if k > 0 and np.trace(XtWX) > 0 else 1e-10
        XtWX_reg = XtWX + reg_strength * np.eye(k)
        XtWX_inv = np.linalg.inv(XtWX_reg)
    
    # 计算系数
    beta = XtWX_inv @ X.T @ W @ y
    
    # 计算预测值和残差
    y_pred = X @ beta
    residuals = y - y_pred
    
    # 计算各种统计量
    df_resid = max(n - k, 1)  # 至少为1，避免除零错误
    
    # 加权残差平方和
    weighted_ssr = np.sum(w * (residuals ** 2))
    
    # 加权总平方和
    weighted_y = w * y
    weighted_mean_y = np.sum(weighted_y) / np.sum(w) if np.sum(w) > 1e-10 else 0
    weighted_sst = np.sum(w * (y - weighted_mean_y) ** 2)
    
    # 加权回归平方和
    weighted_ssr_reg = max(weighted_sst - weighted_ssr, 0)  # 避免负数
    
    # 均方误差
    mse = weighted_ssr / df_resid if df_resid > 0 else weighted_ssr
    
    # R方和调整R方（加权版本）
    r_squared = 1 - (weighted_ssr / weighted_sst) if weighted_sst > 1e-10 else 0
    adj_r_squared = 1 - ((weighted_ssr / df_resid) / (weighted_sst / max(n - 1, 1))) if weighted_sst > 1e-10 and n > k else 0
    
    # 系数标准误
    var_beta = mse * XtWX_inv
    # 避免负方差
    diag_var_beta = np.diag(var_beta)
    diag_var_beta = np.maximum(diag_var_beta, 0)
    std_errors = np.sqrt(diag_var_beta)
    # 避免标准误为零
    std_errors = np.maximum(std_errors, 1e-12)
    
    # t统计量和p值
    t_values = np.divide(beta, std_errors, out=np.zeros_like(beta), where=std_errors!=0)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df_resid))
    
    # 置信区间
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, df_resid)
    conf_int_lower = beta - t_critical * std_errors
    conf_int_upper = beta + t_critical * std_errors
    
    # F统计量
    if k > 1:
        f_statistic = (weighted_ssr_reg / max(k - 1, 1)) / max(mse, 1e-10)
        f_p_value = 1 - stats.f.cdf(f_statistic, max(k - 1, 1), df_resid)
    else:
        f_statistic = 0
        f_p_value = 1
    
    return WLSResult(
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
        feature_names=feature_names,
        weights=weights
    )