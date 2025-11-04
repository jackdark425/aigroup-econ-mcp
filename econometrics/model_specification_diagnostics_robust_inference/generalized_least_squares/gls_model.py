"""
广义最小二乘法 (Generalized Least Squares, GLS) 模型实现
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from scipy import stats

from tools.decorators import with_file_support_decorator as econometric_tool, validate_input


class GLSResult(BaseModel):
    """GLS回归结果"""
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
    log_likelihood: float = Field(..., description="对数似然值")


@econometric_tool("gls_regression")
@validate_input(data_type="econometric")
def gls_regression(
    y_data: List[float],
    x_data: List[List[float]], 
    sigma: Optional[List[List[float]]] = None,
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    confidence_level: float = 0.95
) -> GLSResult:
    """
    广义最小二乘法回归
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        sigma: 误差项协方差矩阵（可选，如未提供则使用单位矩阵）
        feature_names: 特征名称
        constant: 是否包含常数项
        confidence_level: 置信水平
        
    Returns:
        GLSResult: GLS回归结果
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
    
    n, k = X.shape
    
    # 检查数据维度
    if n <= k:
        raise ValueError(f"观测数量({n})必须大于变量数量({k})")
    
    # 如果未提供协方差矩阵，则使用单位矩阵（等价于OLS）
    if sigma is None:
        sigma = np.eye(n, dtype=np.float64)
    else:
        sigma = np.asarray(sigma, dtype=np.float64)
    
    # 检查协方差矩阵维度
    if sigma.shape != (n, n):
        raise ValueError(f"协方差矩阵sigma的维度必须是({n}, {n})，当前是{sigma.shape}")
    
    # 检查协方差矩阵是否包含无效值
    if not np.all(np.isfinite(sigma)):
        raise ValueError("协方差矩阵包含无效值（inf或NaN）")
    
    # 计算协方差矩阵的平方根逆矩阵
    try:
        # 尝试Cholesky分解
        L = np.linalg.cholesky(sigma)
        sigma_inv = np.linalg.inv(sigma)
        # 计算sigma的平方根逆矩阵
        L_inv = np.linalg.inv(L)
    except np.linalg.LinAlgError:
        # 如果Cholesky分解失败，使用特征值分解
        try:
            eigenvals, eigenvecs = np.linalg.eigh(sigma)
            # 处理接近零或负的特征值
            eigenvals = np.maximum(eigenvals, 1e-10)
            # 计算平方根逆矩阵
            L_inv = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
            sigma_inv = L_inv.T @ L_inv
        except np.linalg.LinAlgError:
            # 最后的备选方案：添加对角扰动
            sigma_reg = sigma + 1e-6 * np.eye(n)
            L = np.linalg.cholesky(sigma_reg)
            L_inv = np.linalg.inv(L)
            sigma_inv = np.linalg.inv(sigma_reg)
    
    # 变换数据
    X_transformed = L_inv @ X
    y_transformed = L_inv @ y
    
    # 执行GLS回归: β = (X'*X)^(-1)X'*y
    try:
        XtX = X_transformed.T @ X_transformed
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，添加正则化项
        XtX = X_transformed.T @ X_transformed
        # 使用迹来确定正则化强度
        reg_strength = 1e-10 * np.trace(XtX) / k if k > 0 and np.trace(XtX) > 0 else 1e-10
        XtX_reg = XtX + reg_strength * np.eye(k)
        XtX_inv = np.linalg.inv(XtX_reg)
    
    beta = XtX_inv @ X_transformed.T @ y_transformed
    
    # 计算预测值和残差
    y_pred = X @ beta
    residuals = y - y_pred
    
    # 计算各种统计量
    df_resid = max(n - k, 1)  # 至少为1，避免除零错误
    
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
    
    # 系数标准误 (基于GLS估计的协方差矩阵)
    # Var(β) = (X'Σ^(-1)X)^(-1)
    var_beta = XtX_inv
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
        f_statistic = (ssr_reg / max(k - 1, 1)) / max(mse, 1e-10)
        f_p_value = 1 - stats.f.cdf(f_statistic, max(k - 1, 1), df_resid)
    else:
        f_statistic = 0
        f_p_value = 1
    
    # 对数似然 (假设误差服从正态分布)
    # 基于GLS残差计算
    try:
        sigma_det = np.linalg.det(sigma)
        sigma_det = max(sigma_det, 1e-20)  # 避免log(0)
        log_likelihood = -0.5 * (n * np.log(2 * np.pi) + 
                                np.log(sigma_det) + 
                                n)
    except:
        # 当无法计算行列式时的备选方案
        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + 
                                    np.log(ssr / n + 1e-10) + 
                                    1)

    return GLSResult(
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
        log_likelihood=float(log_likelihood)
    )