"""
模型选择 (Model Selection) 模块实现

包括：
- 信息准则（AIC/BIC/HQIC）
- 交叉验证（K折、留一法）
- 格兰杰因果检验
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from scipy import stats

from tools.decorators import with_file_support_decorator as econometric_tool, validate_input


class ModelSelectionResult(BaseModel):
    """模型选择结果"""
    aic: float = Field(..., description="赤池信息准则 (AIC)")
    bic: float = Field(..., description="贝叶斯信息准则 (BIC)")
    hqic: float = Field(..., description="汉南-奎因信息准则 (HQIC)")
    r_squared: float = Field(..., description="R方")
    adj_r_squared: float = Field(..., description="调整R方")
    log_likelihood: float = Field(..., description="对数似然值")
    n_obs: int = Field(..., description="观测数量")
    n_params: int = Field(..., description="参数数量")
    cv_score: Optional[float] = Field(None, description="交叉验证得分")


@econometric_tool("model_selection_criteria")
@validate_input(data_type="econometric")
def model_selection_criteria(
    y_data: List[float],
    x_data: List[List[float]], 
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    cv_folds: Optional[int] = None
) -> ModelSelectionResult:
    """
    计算模型选择信息准则
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        feature_names: 特征名称
        constant: 是否包含常数项
        cv_folds: 交叉验证折数 (None表示不进行交叉验证，-1表示留一法)
        
    Returns:
        ModelSelectionResult: 模型选择结果
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
    
    # 残差平方和
    ssr = np.sum(residuals ** 2)
    
    # 总平方和
    sst = np.sum((y - np.mean(y)) ** 2)
    
    # R方和调整R方
    r_squared = 1 - (ssr / sst) if sst != 0 else 0
    adj_r_squared = 1 - ((ssr / (n - k)) / (sst / (n - 1))) if sst != 0 and n > k else 0
    
    # 对数似然 (假设误差服从正态分布)
    sigma_squared = ssr / n if n > 0 else 0
    # 避免log(0)或负数
    if sigma_squared <= 0:
        sigma_squared = 1e-10
    log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma_squared) + 1)
    
    # 信息准则
    aic = -2 * log_likelihood + 2 * k
    bic = -2 * log_likelihood + k * np.log(n) if n > 0 else np.inf
    hqic = -2 * log_likelihood + 2 * k * np.log(np.log(n)) if n > 1 else np.inf
    
    # 交叉验证
    cv_score = None
    if cv_folds is not None:
        cv_score = _cross_validation(y, X, cv_folds)
    
    return ModelSelectionResult(
        aic=float(aic),
        bic=float(bic),
        hqic=float(hqic),
        r_squared=float(r_squared),
        adj_r_squared=float(adj_r_squared),
        log_likelihood=float(log_likelihood),
        n_obs=n,
        n_params=k,
        cv_score=float(cv_score) if cv_score is not None else None
    )


def _cross_validation(y: np.ndarray, X: np.ndarray, folds: Optional[int]) -> float:
    """
    执行交叉验证
    
    Args:
        y: 因变量
        X: 自变量矩阵
        folds: 折数 (-1表示留一法，其他正数表示K折交叉验证)
        
    Returns:
        float: 交叉验证得分 (平均MSE)
    """
    n = len(y)
    
    if folds == -1 or folds >= n:
        # 留一法交叉验证
        folds = n
    
    if folds <= 1 or X.shape[0] != n:
        return None
    
    # 创建折叠索引
    indices = np.arange(n)
    np.random.seed(42)  # 固定随机种子以确保结果可重现
    np.random.shuffle(indices)
    fold_sizes = np.full(folds, n // folds)
    fold_sizes[:n % folds] += 1
    current = 0
    mse_scores = []
    
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        
        # 分割数据
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        try:
            # 检查是否有足够的数据进行训练和测试
            if X_train.shape[0] < X_train.shape[1] or X_train.shape[0] == 0 or X_test.shape[0] == 0:
                continue
                
            # 训练模型
            XtX = X_train.T @ X_train
            # 添加正则化防止矩阵奇异
            if XtX.shape[0] > 0:
                reg_strength = 1e-10 * np.trace(XtX) / XtX.shape[0] if np.trace(XtX) > 0 else 1e-10
                XtX_reg = XtX + reg_strength * np.eye(XtX.shape[0])
                beta_train = np.linalg.inv(XtX_reg) @ X_train.T @ y_train
            else:
                continue
            
            # 预测
            y_pred = X_test @ beta_train
            
            # 检查预测值是否有效
            if not np.all(np.isfinite(y_pred)):
                continue
                
            # 计算MSE
            mse = np.mean((y_test - y_pred) ** 2)
            # 检查MSE是否有效
            if np.isfinite(mse):
                mse_scores.append(mse)
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            # 如果出现数值问题，跳过这一折
            pass
        except Exception:
            # 捕获其他可能的异常
            pass
            
        current = stop
    
    return np.mean(mse_scores) if mse_scores and len(mse_scores) > 0 else None
