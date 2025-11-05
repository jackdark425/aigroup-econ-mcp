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
        
    Raises:
        ValueError: 当输入数据无效时抛出异常
    """
    # 输入验证
    if not y_data or not x_data:
        raise ValueError("因变量和自变量数据不能为空")
    
    # 转换为numpy数组
    y = np.array(y_data)
    
    # 确保X是二维数组
    if isinstance(x_data[0], (int, float)):
        # 单个特征的情况，需要转置
        X = np.array(x_data).reshape(-1, 1)
    else:
        X = np.array(x_data)  # 移除转置，保持原始维度
    
    # 验证数据维度一致性
    if len(y) != X.shape[0]:
        raise ValueError(f"因变量长度({len(y)})与自变量长度({X.shape[0]})不一致")
    
    # 特殊情况处理：单个观测值
    if len(y) == 1:
        # 对于单个观测值，无法进行有效的GMM估计，退化为简单的OLS
        if constant:
            coefficients = [float(y[0]), 0.0]  # 截距为y值，斜率为0
            std_errors = [0.0, 0.0]
            t_values = [np.inf, np.inf]
            p_values = [0.0, 0.0]
            conf_int_lower = [float(y[0]), 0.0]
            conf_int_upper = [float(y[0]), 0.0]
            j_statistic = 0.0
            j_p_value = 1.0
            weight_matrix = [[1.0, 0.0], [0.0, 1.0]]
            n_moments = 2
            if feature_names:
                feature_names = ["const"] + feature_names
            else:
                feature_names = ["const", "x0"]
        else:
            coefficients = [float(y[0])]  # 系数为y值
            std_errors = [0.0]
            t_values = [np.inf]
            p_values = [0.0]
            conf_int_lower = [float(y[0])]
            conf_int_upper = [float(y[0])]
            j_statistic = 0.0
            j_p_value = 1.0
            weight_matrix = [[1.0]]
            n_moments = 1
            if not feature_names:
                feature_names = ["x0"]
        
        return GMMResult(
            coefficients=coefficients,
            std_errors=std_errors,
            t_values=t_values,
            p_values=p_values,
            conf_int_lower=conf_int_lower,
            conf_int_upper=conf_int_upper,
            j_statistic=j_statistic,
            j_p_value=j_p_value,
            weight_matrix=weight_matrix,
            n_obs=1,
            n_moments=n_moments,
            feature_names=feature_names
        )
    
    # 如果没有提供工具变量，则使用x_data作为工具变量 (两阶段最小二乘法)
    if instruments is None:
        Z = X.copy()
    else:
        if isinstance(instruments[0], (int, float)):
            # 单个工具变量的情况，需要转置
            Z = np.array(instruments).reshape(-1, 1)
        else:
            Z = np.array(instruments)  # 移除转置，保持原始维度
        
        # 验证工具变量维度一致性
        if len(y) != Z.shape[0]:
            raise ValueError(f"因变量长度({len(y)})与工具变量长度({Z.shape[0]})不一致")
    
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
    
    # 检查过度识别情况
    if l < k:
        raise ValueError(f"工具变量数量({l})少于参数数量({k})，无法进行GMM估计")
    
    try:
        # 使用statsmodels的GMM实现
        # 定义矩条件函数
        def moment_conditions(params, y, X, Z):
            residuals = y - X @ params
            moments = Z * residuals[:, np.newaxis]
            return moments
        
        # 创建GMM模型
        # 注意：statsmodels的GMM实现可能需要特定的数据格式
        # 这里我们继续使用自定义实现，但增强其稳健性
        
        # 第一步：使用单位矩阵作为权重矩阵进行估计
        W1 = np.eye(l)
        beta_1 = _gmm_objective_estep(X, Z, y, W1)
        
        # 计算残差和矩条件
        residuals_1 = y - X @ beta_1
        moments_1 = Z * residuals_1[:, np.newaxis]
        
        # 估计最优权重矩阵 (协方差矩阵的逆)
        S = moments_1.T @ moments_1 / n
        # 添加一个小的正数以确保矩阵可逆
        S = S + np.eye(S.shape[0]) * 1e-8
        W2 = np.linalg.inv(S)
        
        # 第二步：使用最优权重矩阵进行估计
        beta_2 = _gmm_objective_estep(X, Z, y, W2)
        
        # 计算最终的统计量
        residuals_2 = y - X @ beta_2
        moments_2 = Z * residuals_2[:, np.newaxis]
        
        # J统计量 (过度识别限制检验)
        g_bar = np.mean(moments_2, axis=0)
        J_stat = n * g_bar.T @ W2 @ g_bar
        
        # 计算p值，处理恰好识别（自由度=0）的情况
        df = l - k  # 自由度为矩条件数减去参数数量
        if df > 0:
            J_p_value = 1 - stats.chi2.cdf(J_stat, df=df)
        else:
            # 恰好识别的情况，J统计量应该接近0，p值设为1.0
            J_p_value = 1.0
        
        # 计算协方差矩阵和标准误
        # GMM估计量的渐近协方差矩阵: (G'WG)^(-1)G'WSW'G(G'WG)^(-1)
        G = -Z.T @ X / n  # 矩条件对参数的导数
        # 添加正则化项以提高数值稳定性
        GG = G.T @ W2 @ G + np.eye(G.shape[1]) * 1e-8
        cov_matrix = np.linalg.inv(GG) @ (G.T @ W2 @ S @ W2 @ G) @ np.linalg.inv(GG) / n
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
    except Exception as e:
        # 出现错误时使用原始实现
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
        
        # 计算p值，处理恰好识别（自由度=0）的情况
        df = l - k  # 自由度为矩条件数减去参数数量
        if df > 0:
            J_p_value = 1 - stats.chi2.cdf(J_stat, df=df)
        else:
            # 恰好识别的情况，J统计量应该接近0，p值设为1.0
            J_p_value = 1.0
        
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
    try:
        beta_ols = np.linalg.inv(X.T @ X) @ X.T @ y.reshape(-1, 1)
        beta_ols = beta_ols.flatten()
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，使用伪逆
        beta_ols = np.linalg.pinv(X.T @ X) @ X.T @ y.reshape(-1, 1)
        beta_ols = beta_ols.flatten()
    
    # 优化
    result = minimize(objective, beta_ols, method='BFGS')
    return result.x