"""
正则化方法 (Regularization Methods) 模块实现

包括岭回归、LASSO和弹性网络等方法，用于处理多重共线性/高维数据
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from scipy import stats

from tools.decorators import with_file_support_decorator as econometric_tool, validate_input


class RegularizationResult(BaseModel):
    """正则化回归结果"""
    coefficients: List[float] = Field(..., description="回归系数")
    intercept: float = Field(..., description="截距项")
    r_squared: float = Field(..., description="R方")
    adj_r_squared: float = Field(..., description="调整R方")
    n_obs: int = Field(..., description="观测数量")
    feature_names: List[str] = Field(..., description="特征名称")
    method: str = Field(..., description="使用的正则化方法")


@econometric_tool("regularized_regression")
@validate_input(data_type="econometric")
def regularized_regression(
    y_data: List[float],
    x_data: List[List[float]], 
    method: str = "ridge",
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    feature_names: Optional[List[str]] = None,
    fit_intercept: bool = True
) -> RegularizationResult:
    """
    正则化回归（岭回归、LASSO、弹性网络）
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        method: 正则化方法 ('ridge', 'lasso', 'elastic_net')
        alpha: 正则化强度
        l1_ratio: 弹性网络混合比例 (仅用于elastic_net，0为岭回归，1为LASSO)
        feature_names: 特征名称
        fit_intercept: 是否拟合截距项
        
    Returns:
        RegularizationResult: 正则化回归结果
    """
    # 转换为numpy数组
    y = np.asarray(y_data, dtype=np.float64)
    X = np.asarray(x_data, dtype=np.float64)
    
    # 检查数据维度
    if X.size == 0 or y.size == 0:
        raise ValueError("输入数据不能为空")
    
    n, p = X.shape
    
    if len(y) != n:
        raise ValueError("因变量和自变量的观测数量必须相同")
    
    if p == 0:
        # 没有特征，只拟合截距
        y_mean = np.mean(y)
        if fit_intercept:
            intercept = float(y_mean)
            beta = np.array([])
        else:
            intercept = 0.0
            beta = np.array([])
        
        # 计算R方（简单情况）
        y_pred = np.full_like(y, y_mean)
        ssr = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ssr / sst) if sst > 1e-10 else 0
        adj_r_squared = r_squared  # 无特征时调整R方等于R方
        
        if not feature_names and p > 0:
            feature_names = [f"x{i}" for i in range(p)]
        elif not feature_names:
            feature_names = []
        
        return RegularizationResult(
            coefficients=beta.tolist(),
            intercept=intercept,
            r_squared=float(r_squared),
            adj_r_squared=float(adj_r_squared),
            n_obs=n,
            feature_names=feature_names,
            method=method
        )
    
    # 标准化特征（均值为0，标准差为1）
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    # 更安全地处理标准差为零的情况
    X_std = np.where(X_std < 1e-10, 1.0, X_std)  # 避免除以接近零的数
    X_scaled = (X - X_mean) / X_std
    
    # 标准化目标变量（如果需要）
    y_mean = np.mean(y)
    y_scaled = y - y_mean if fit_intercept else y
    
    # 根据方法选择正则化
    if method == "ridge":
        # 岭回归: β = (X'X + αI)^(-1)X'y
        try:
            I = np.eye(p)
            XtX = X_scaled.T @ X_scaled
            # 添加正则化项
            XtX_reg = XtX + alpha * I * n  # 乘以n以匹配sklearn的实现
            beta_scaled = np.linalg.inv(XtX_reg) @ X_scaled.T @ y_scaled
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            XtX = X_scaled.T @ X_scaled
            I = np.eye(p)
            XtX_reg = XtX + alpha * I * n
            beta_scaled = np.linalg.pinv(XtX_reg) @ X_scaled.T @ y_scaled
    elif method == "lasso":
        # LASSO回归（简化实现，使用坐标下降法）
        beta_scaled = _lasso_coordinate_descent(X_scaled, y_scaled, alpha, max_iter=2000, tol=1e-6)
    elif method == "elastic_net":
        # 弹性网络（简化实现）
        beta_scaled = _elastic_net_coordinate_descent(X_scaled, y_scaled, alpha, l1_ratio, max_iter=2000, tol=1e-6)
    else:
        raise ValueError("方法必须是 'ridge', 'lasso' 或 'elastic_net'")
    
    # 变换回原始尺度
    if fit_intercept and len(beta_scaled) > 0:
        intercept = y_mean - np.sum(beta_scaled * X_mean / X_std)
        # 避免除以零
        beta = np.divide(beta_scaled, X_std, out=np.zeros_like(beta_scaled), where=X_std!=0)
    elif len(beta_scaled) > 0:
        intercept = 0.0
        beta = np.divide(beta_scaled, X_std, out=np.zeros_like(beta_scaled), where=X_std!=0)
    else:
        intercept = 0.0 if not fit_intercept else float(y_mean)
        beta = np.array([])
    
    # 计算预测值和R方
    if len(beta) > 0:
        y_pred = X @ beta + intercept
    else:
        y_pred = np.full_like(y, intercept)
    
    ssr = np.sum((y - y_pred) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2) if len(y) > 1 else 0
    r_squared = 1 - (ssr / sst) if sst > 1e-10 else 0
    
    # 调整R方
    if n > len(beta) + (1 if fit_intercept else 0) and sst > 1e-10:
        adj_r_squared = 1 - ((ssr / (n - len(beta) - (1 if fit_intercept else 0))) / 
                            (sst / (n - 1)))
    else:
        adj_r_squared = r_squared
    
    if not feature_names and p > 0:
        feature_names = [f"x{i}" for i in range(p)]
    elif not feature_names:
        feature_names = []
    
    return RegularizationResult(
        coefficients=beta.tolist(),
        intercept=float(intercept),
        r_squared=float(r_squared),
        adj_r_squared=float(adj_r_squared),
        n_obs=n,
        feature_names=feature_names,
        method=method
    )


def _lasso_coordinate_descent(X, y, alpha, max_iter=2000, tol=1e-6):
    """
    LASSO回归的坐标下降实现
    
    Args:
        X: 特征矩阵
        y: 目标变量
        alpha: 正则化参数
        max_iter: 最大迭代次数
        tol: 收敛容忍度
        
    Returns:
        numpy.ndarray: 回归系数
    """
    n, p = X.shape
    beta = np.zeros(p)
    
    # 预计算 X_j^T * X_j 以提高效率
    XtX_diag = np.sum(X**2, axis=0)
    
    # 避免除以零
    XtX_diag = np.maximum(XtX_diag, 1e-10)
    
    for iteration in range(max_iter):
        beta_old = beta.copy()
        max_change = 0.0
        
        for j in range(p):
            # 计算第j个特征的残差（不包括第j个特征的贡献）
            residual = y - X @ beta + beta[j] * X[:, j]
            
            # 计算软阈值
            rho = X[:, j] @ residual
            
            # 软阈值函数
            old_beta_j = beta[j]
            if rho > alpha * n:
                beta[j] = (rho - alpha * n) / XtX_diag[j]
            elif rho < -alpha * n:
                beta[j] = (rho + alpha * n) / XtX_diag[j]
            else:
                beta[j] = 0
            
            # 更新最大变化量
            max_change = max(max_change, abs(beta[j] - old_beta_j))
        
        # 检查收敛
        if max_change < tol:
            break
    
    return beta


def _elastic_net_coordinate_descent(X, y, alpha, l1_ratio, max_iter=2000, tol=1e-6):
    """
    弹性网络的坐标下降实现
    
    Args:
        X: 特征矩阵
        y: 目标变量
        alpha: 正则化参数
        l1_ratio: L1正则化比例
        max_iter: 最大迭代次数
        tol: 收敛容忍度
        
    Returns:
        numpy.ndarray: 回归系数
    """
    n, p = X.shape
    beta = np.zeros(p)
    alpha_l1 = alpha * l1_ratio * n
    alpha_l2 = alpha * (1 - l1_ratio) * n
    
    # 预计算 X_j^T * X_j 以提高效率
    XtX_diag = np.sum(X**2, axis=0)
    # 添加L2正则化项
    XtX_diag_reg = XtX_diag + alpha_l2
    
    # 避免除以零
    XtX_diag_reg = np.maximum(XtX_diag_reg, 1e-10)
    
    for iteration in range(max_iter):
        beta_old = beta.copy()
        max_change = 0.0
        
        for j in range(p):
            # 计算第j个特征的残差（不包括第j个特征的贡献）
            residual = y - X @ beta + beta[j] * X[:, j]
            
            # 计算梯度
            rho = X[:, j] @ residual
            
            # 弹性网络软阈值函数
            old_beta_j = beta[j]
            if rho > alpha_l1:
                beta[j] = (rho - alpha_l1) / XtX_diag_reg[j]
            elif rho < -alpha_l1:
                beta[j] = (rho + alpha_l1) / XtX_diag_reg[j]
            else:
                beta[j] = 0
            
            # 更新最大变化量
            max_change = max(max_change, abs(beta[j] - old_beta_j))
        
        # 检查收敛
        if max_change < tol:
            break
    
    return beta