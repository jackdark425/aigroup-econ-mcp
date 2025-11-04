"""
广义矩估计 (GMM) 模型实现
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

from econometrics.decorators import econometric_tool, validate_input


class GMMResult(BaseModel):
    """广义矩估计结果"""
    coefficients: List[float] = Field(..., description="估计系数")
    std_errors: List[float] = Field(..., description="系数标准误")
    t_values: List[float] = Field(..., description="t统计量")
    p_values: List[float] = Field(..., description="p值")
    conf_int_lower: List[float] = Field(..., description="置信区间下界")
    conf_int_upper: List[float] = Field(..., description="置信区间上界")
    j_statistic: float = Field(..., description="J统计量")
    j_p_value: float = Field(..., description="J统计量p值")
    weight_matrix: List[List[float]] = Field(..., description="权重矩阵")
    n_obs: int = Field(..., description="观测数量")
    n_moments: int = Field(..., description="矩条件数量")
    feature_names: List[str] = Field(..., description="特征名称")


@econometric_tool("gmm_estimation")
@validate_input(data_type="econometric")
def gmm_estimation(
    y_data: List[float],
    x_data: List[List[float]], 
    instruments: Optional[List[List[float]]] = None,
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    confidence_level: float = 0.95
) -> GMMResult:
    """
    广义矩估计
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        instruments: 工具变量数据 (如果为None，则使用x_data作为工具变量，退化为OLS)
        feature_names: 特征名称
        constant: 是否包含常数项
        confidence_level: 置信水平
        
    Returns:
        GMMResult: 广义矩估计结果
    """
    # 转换为numpy数组
    y = np.array(y_data)
    X = np.array(x_data)
    
    # 如果没有提供工具变量，则使用x_data作为工具变量 (两阶段最小二乘法)
    if instruments is None:
        Z = X.copy()
    else:
        Z = np.array(instruments)
    
    # 添加常数项
    if constant:
        X = np.column_stack([np.ones(len(X)), X])
        Z = np.column_stack([np.ones(len(Z)), Z])
        if feature_names:
            feature_names = ["const"] + feature_names
        else:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    else:
        if not feature_names:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    
    n = len(y)
    k = X.shape[1]  # 参数数量
    l = Z.shape[1]  # 矩条件数量
    
    # 第一步：使用单位矩阵作为权重矩阵进行估计
    W1 = np.eye(l)
    beta_1 = _gmm_objective_estep(X, Z, y, W1)
    
    # 计算残差和矩条件
    residuals_1 = y - X @ beta_1
    moments_1 = Z * residuals_1[:, np.newaxis]
    
    # 估计最优权重矩阵 (协方差矩阵的逆)
    S = moments_1.T @ moments_1 / n
    W2 = np.linalg.inv(S)
    
    # 第二步：使用最优权重矩阵进行估计
    beta_2 = _gmm_objective_estep(X, Z, y, W2)
    
    # 计算最终的统计量
    residuals_2 = y - X @ beta_2
    moments_2 = Z * residuals_2[:, np.newaxis]
    
    # J统计量 (过度识别限制检验)
    g_bar = np.mean(moments_2, axis=0)
    J_stat = n * g_bar.T @ W2 @ g_bar
    J_p_value = 1 - stats.chi2.cdf(J_stat, df=l-k)  # 自由度为矩条件数减去参数数量
    
    # 计算协方差矩阵和标准误
    # GMM估计量的渐近协方差矩阵: (G'WG)^(-1)G'WSW'G(G'WG)^(-1)
    G = -Z.T @ X / n  # 矩条件对参数的导数
    cov_matrix = np.linalg.inv(G.T @ W2 @ G) @ (G.T @ W2 @ S @ W2 @ G) @ np.linalg.inv(G.T @ W2 @ G) / n
    std_errors = np.sqrt(np.diag(cov_matrix))
    
    # t统计量和p值
    t_values = beta_2 / std_errors
    p_values = 2 * (1 - stats.norm.cdf(np.abs(t_values)))  # 使用标准正态分布
    
    # 置信区间
    alpha = 1 - confidence_level
    z_critical = stats.norm.ppf(1 - alpha/2)
    conf_int_lower = beta_2 - z_critical * std_errors
    conf_int_upper = beta_2 + z_critical * std_errors
    
    return GMMResult(
        coefficients=beta_2.tolist(),
        std_errors=std_errors.tolist(),
        t_values=t_values.tolist(),
        p_values=p_values.tolist(),
        conf_int_lower=conf_int_lower.tolist(),
        conf_int_upper=conf_int_upper.tolist(),
        j_statistic=float(J_stat),
        j_p_value=float(J_p_value),
        weight_matrix=W2.tolist(),
        n_obs=n,
        n_moments=l,
        feature_names=feature_names
    )


def _gmm_objective_estep(X: np.ndarray, Z: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    GMM目标函数最小化步骤
    
    Args:
        X: 自变量矩阵
        Z: 工具变量矩阵
        y: 因变量向量
        W: 权重矩阵
        
    Returns:
        np.ndarray: 参数估计值
    """
    # 定义目标函数
    def objective(beta):
        residuals = y - X @ beta
        moments = Z * residuals[:, np.newaxis]
        g_bar = np.mean(moments, axis=0)
        return g_bar.T @ W @ g_bar
    
    # 初始值使用OLS估计
    beta_ols = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # 优化
    result = minimize(objective, beta_ols, method='BFGS')
    return result.x