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
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

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
    
    # 确保X是二维数组
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
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
    
    # 使用sklearn的StandardScaler进行标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # 标准化特征和目标变量
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # 根据方法选择模型
    if method == "ridge":
        model = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
    elif method == "lasso":
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=2000, tol=1e-6, random_state=42)
    elif method == "elastic_net":
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=2000, tol=1e-6, random_state=42)
    else:
        raise ValueError("方法必须是 'ridge', 'lasso' 或 'elastic_net'")
    
    # 训练模型
    try:
        model.fit(X_scaled, y_scaled)
    except Exception as e:
        raise ValueError(f"模型拟合失败: {str(e)}")
    
    # 获取系数并转换回原始尺度
    coef_scaled = model.coef_
    intercept_scaled = model.intercept_
    
    # 转换回原始尺度
    # 对于标准化的数据，系数变换为: beta = coef_scaled * std_y / std_X
    # 截距变换为: intercept = mean_y - beta * mean_X
    if fit_intercept and len(scaler_X.scale_) == len(coef_scaled):
        # 确保不会除以零
        scale_X = np.where(scaler_X.scale_ == 0, 1.0, scaler_X.scale_)
        beta = coef_scaled * (scaler_y.scale_ / scale_X)
        intercept = scaler_y.mean_ - np.sum(beta * scaler_X.mean_)
    else:
        beta = coef_scaled * scaler_y.scale_ if len(coef_scaled) > 0 else np.array([])
        intercept = scaler_y.mean_ if fit_intercept else 0.0
    
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


