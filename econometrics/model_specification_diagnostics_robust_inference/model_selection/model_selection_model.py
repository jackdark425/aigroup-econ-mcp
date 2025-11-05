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
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

from tools.decorators import with_file_support_decorator as econometric_tool, validate_input


class GrangerCausalityResult(BaseModel):
    """格兰杰因果检验结果"""
    f_statistic: float = Field(..., description="F统计量")
    p_value: float = Field(..., description="p值")
    lag_order: int = Field(..., description="滞后阶数")
    n_obs: int = Field(..., description="观测数量")
    dependent_variable: str = Field(..., description="因变量")
    independent_variable: str = Field(..., description="格兰杰原因变量")


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


@econometric_tool("granger_causality_test")
@validate_input(data_type="timeseries")
def granger_causality_test(
    x_data: List[float],
    y_data: List[float], 
    max_lag: int = 1,
    add_constant: bool = True
) -> GrangerCausalityResult:
    """
    格兰杰因果检验
    
    Args:
        x_data: 可能的格兰杰原因变量
        y_data: 因变量
        max_lag: 最大滞后阶数
        add_constant: 是否添加常数项
        
    Returns:
        GrangerCausalityResult: 格兰杰因果检验结果
    """
    # 转换为numpy数组
    x = np.asarray(x_data, dtype=np.float64)
    y = np.asarray(y_data, dtype=np.float64)
    
    # 检查数据长度
    if len(x) != len(y):
        raise ValueError("x_data和y_data的长度必须相同")
    
    if len(x) <= max_lag:
        raise ValueError("数据长度必须大于滞后阶数")
    
    # 构建数据框用于statsmodels
    data = pd.DataFrame({'y': y, 'x': x})
    
    # 执行格兰杰因果检验
    try:
        # grangercausalitytests返回一个字典，键为滞后阶数
        test_result = grangercausalitytests(data, max_lag, addconst=add_constant, verbose=False)
        
        # 获取指定滞后阶数的结果（使用最大滞后阶数）
        lag_order = max_lag
        test_stats = test_result[lag_order][0]
        
        # 提取F统计量和p值（使用ssr F-test）
        f_statistic = test_stats['F test']
        f_stat = f_statistic[0]  # F统计量
        p_value = f_statistic[1]  # p值
        
    except Exception as e:
        # 如果检验失败，返回默认值
        f_stat = 0.0
        p_value = 1.0
        lag_order = max_lag
    
    return GrangerCausalityResult(
        f_statistic=float(f_stat),
        p_value=float(p_value),
        lag_order=lag_order,
        n_obs=len(y) - lag_order,  # 考虑滞后后的实际观测数
        dependent_variable="y",
        independent_variable="x"
    )


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
        X = sm.add_constant(X)
        if feature_names:
            feature_names = ["const"] + feature_names
        else:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    else:
        if not feature_names:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    
    # 执行OLS回归
    try:
        model = sm.OLS(y, X)
        results = model.fit()
    except Exception as e:
        raise ValueError(f"无法拟合模型: {str(e)}")
    
    # 提取统计量
    n = int(results.nobs)
    k = len(results.params)
    r_squared = float(results.rsquared)
    adj_r_squared = float(results.rsquared_adj)
    log_likelihood = float(results.llf)
    aic = float(results.aic)
    bic = float(results.bic)
    
    # 计算HQIC (statsmodels中没有直接提供HQIC)
    if n > 1 and np.log(n) != 0:
        hqic = -2 * log_likelihood + 2 * k * np.log(np.log(n))
    else:
        hqic = np.inf
    
    # 交叉验证
    cv_score = None
    if cv_folds is not None:
        cv_score = _cross_validation(y, X, cv_folds)
    
    return ModelSelectionResult(
        aic=aic,
        bic=bic,
        hqic=float(hqic) if np.isfinite(hqic) else np.inf,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        log_likelihood=log_likelihood,
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
    
    if folds is None or folds == 0:
        return None
    
    if folds == -1 or folds >= n:
        # 留一法交叉验证
        folds = n
    
    if folds <= 1 or X.shape[0] != n:
        return None
    
    # 检查是否有足够的数据进行训练和测试
    if X.shape[0] < X.shape[1]:
        return None
    
    # 创建折叠索引
    indices = np.arange(n)
    np.random.seed(42)  # 固定随机种子以确保结果可重现
    np.random.shuffle(indices)
    
    # 计算每折的大小
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
                
            # 训练模型，使用带正则化的求解方法
            try:
                # 使用statsmodels进行更稳定的回归
                train_model = sm.OLS(y_train, X_train)
                train_results = train_model.fit()
                beta_train = train_results.params
            except:
                # 如果statsmodels失败，使用numpy的最小二乘法
                # 添加正则化防止矩阵奇异
                XtX = X_train.T @ X_train
                if XtX.shape[0] > 0:
                    # 添加一个小的正则化项
                    reg_param = 1e-10 * np.trace(XtX) / XtX.shape[0] if np.trace(XtX) > 0 and XtX.shape[0] > 0 else 1e-10
                    XtX_reg = XtX + reg_param * np.eye(XtX.shape[0])
                    try:
                        beta_train = np.linalg.solve(XtX_reg, X_train.T @ y_train)
                    except np.linalg.LinAlgError:
                        # 如果仍然失败，使用伪逆
                        beta_train = np.linalg.pinv(XtX_reg) @ X_train.T @ y_train
                else:
                    continue
            
            # 预测
            try:
                y_pred = X_test @ beta_train
            except:
                continue
            
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