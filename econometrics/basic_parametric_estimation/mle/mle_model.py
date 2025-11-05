"""
最大似然估计 (MLE) 模型实现
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel


class MLEResult(BaseModel):
    """最大似然估计结果"""
    parameters: List[float] = Field(..., description="估计参数")
    std_errors: List[float] = Field(..., description="参数标准误")
    conf_int_lower: List[float] = Field(..., description="置信区间下界")
    conf_int_upper: List[float] = Field(..., description="置信区间上界")
    log_likelihood: float = Field(..., description="对数似然值")
    aic: float = Field(..., description="赤池信息准则")
    bic: float = Field(..., description="贝叶斯信息准则")
    convergence: bool = Field(..., description="是否收敛")
    n_obs: int = Field(..., description="观测数量")
    param_names: List[str] = Field(..., description="参数名称")


def mle_estimation(
    data: List[float],
    distribution: str = "normal",
    initial_params: Optional[List[float]] = None,
    confidence_level: float = 0.95
) -> MLEResult:
    """
    最大似然估计
    
    Args:
        data: 数据
        distribution: 分布类型 ('normal', 'poisson', 'exponential')
        initial_params: 初始参数值
        confidence_level: 置信水平
        
    Returns:
        MLEResult: 最大似然估计结果
        
    Raises:
        ValueError: 当输入数据无效时抛出异常
    """
    # 输入验证
    if not data:
        raise ValueError("数据不能为空")
    
    data = np.array(data, dtype=np.float64)
    n = len(data)
    
    # 检查数据有效性
    if np.isnan(data).any():
        raise ValueError("数据中包含缺失值(NaN)")
    
    if np.isinf(data).any():
        raise ValueError("数据中包含无穷大值")
    
    # 分布特定的验证
    if distribution == "exponential" and np.any(data < 0):
        raise ValueError("指数分布的数据必须为非负数")
    
    if distribution == "poisson" and (np.any(data < 0) or not np.all(data == np.floor(data))):
        raise ValueError("泊松分布的数据必须为非负整数")
    
    if distribution == "normal":
        # 正态分布的MLE
        return _normal_mle(data, initial_params, confidence_level)
    elif distribution == "poisson":
        # 泊松分布的MLE
        return _poisson_mle(data, initial_params, confidence_level)
    elif distribution == "exponential":
        # 指数分布的MLE
        return _exponential_mle(data, initial_params, confidence_level)
    else:
        raise ValueError(f"不支持的分布类型: {distribution}")


def _normal_mle(data: np.ndarray, initial_params: Optional[List[float]], confidence_level: float) -> MLEResult:
    """正态分布最大似然估计"""
    # 使用样本均值和标准差作为初始估计
    mu_hat = np.mean(data)
    sigma_hat = np.std(data, ddof=1)  # 使用样本标准差
    
    # 检查标准差是否为零
    if sigma_hat == 0:
        raise ValueError("数据标准差为零，无法进行正态分布MLE估计")
    
    # 使用statsmodels的MLE估计
    try:
        # 直接使用解析解
        n = len(data)
        log_likelihood = float(np.sum(stats.norm.logpdf(data, loc=mu_hat, scale=sigma_hat)))
        
        # 标准误
        std_error_mu = sigma_hat / np.sqrt(n)
        std_error_sigma = sigma_hat / np.sqrt(2 * n)
        std_errors = [std_error_mu, std_error_sigma]
        
        # 置信区间
        alpha = 1 - confidence_level
        z_value = stats.norm.ppf(1 - alpha/2)
        conf_int_lower = [mu_hat - z_value * std_error_mu, sigma_hat - z_value * std_error_sigma]
        conf_int_upper = [mu_hat + z_value * std_error_mu, sigma_hat + z_value * std_error_sigma]
        
        # 信息准则
        k = 2  # 参数数量
        aic = -2 * log_likelihood + 2 * k
        bic = -2 * log_likelihood + k * np.log(n)
        
        return MLEResult(
            parameters=[float(mu_hat), float(sigma_hat)],
            std_errors=std_errors,
            conf_int_lower=conf_int_lower,
            conf_int_upper=conf_int_upper,
            log_likelihood=log_likelihood,
            aic=float(aic),
            bic=float(bic),
            convergence=True,
            n_obs=n,
            param_names=["mu", "sigma"]
        )
    except Exception as e:
        raise ValueError(f"正态分布MLE估计失败: {str(e)}")


def _poisson_mle(data: np.ndarray, initial_params: Optional[List[float]], confidence_level: float) -> MLEResult:
    """泊松分布最大似然估计"""
    # 泊松分布的MLE有解析解：lambda_hat = mean(data)
    lambda_hat = np.mean(data)
    n = len(data)
    
    # 检查均值是否为零
    if lambda_hat == 0:
        raise ValueError("数据均值为零，无法进行泊松分布MLE估计")
    
    try:
        # 计算对数似然值
        log_likelihood = float(np.sum(stats.poisson.logpmf(data, lambda_hat)))
        
        # 标准误
        std_error = np.sqrt(lambda_hat / n)
        std_errors = [std_error]
        
        # 置信区间
        alpha = 1 - confidence_level
        z_value = stats.norm.ppf(1 - alpha/2)
        conf_int_lower = [lambda_hat - z_value * std_error]
        conf_int_upper = [lambda_hat + z_value * std_error]
        
        # 信息准则
        k = 1  # 参数数量
        aic = -2 * log_likelihood + 2 * k
        bic = -2 * log_likelihood + k * np.log(n)
        
        return MLEResult(
            parameters=[float(lambda_hat)],
            std_errors=std_errors,
            conf_int_lower=conf_int_lower,
            conf_int_upper=conf_int_upper,
            log_likelihood=log_likelihood,
            aic=float(aic),
            bic=float(bic),
            convergence=True,
            n_obs=n,
            param_names=["lambda"]
        )
    except Exception as e:
        raise ValueError(f"泊松分布MLE估计失败: {str(e)}")


def _exponential_mle(data: np.ndarray, initial_params: Optional[List[float]], confidence_level: float) -> MLEResult:
    """指数分布最大似然估计"""
    # 指数分布的MLE有解析解：lambda_hat = 1 / mean(data)
    mean_data = np.mean(data)
    if mean_data <= 0:
        raise ValueError("指数分布的数据均值必须为正数")
    
    lambda_hat = 1.0 / mean_data
    n = len(data)
    
    # 检查参数有效性
    if not np.isfinite(lambda_hat):
        raise ValueError("计算出的参数值无效")
    
    try:
        # 计算对数似然值
        log_likelihood = float(np.sum(stats.expon.logpdf(data, scale=1/lambda_hat)))
        
        # 标准误计算 (对于指数分布，标准误为lambda/sqrt(n))
        # 使用更精确的计算方法
        std_error = lambda_hat / np.sqrt(n)
        std_errors = [std_error]
        
        # 验证标准误的有效性
        if not np.isfinite(std_error) or std_error <= 0:
            raise ValueError("计算出的标准误无效")
        
        # 置信区间
        alpha = 1 - confidence_level
        z_value = stats.norm.ppf(1 - alpha/2)
        
        # 检查z值有效性
        if not np.isfinite(z_value):
            raise ValueError("计算出的临界值无效")
            
        conf_int_lower = [lambda_hat - z_value * std_error]
        conf_int_upper = [lambda_hat + z_value * std_error]
        
        # 检查置信区间边界有效性
        if not (np.isfinite(conf_int_lower[0]) and np.isfinite(conf_int_upper[0])):
            raise ValueError("计算出的置信区间无效")
            
        # 确保置信区间下限不为负
        conf_int_lower[0] = max(conf_int_lower[0], 1e-10)
        
        # 信息准则
        k = 1  # 参数数量
        aic = -2 * log_likelihood + 2 * k
        bic = -2 * log_likelihood + k * np.log(n)
        
        return MLEResult(
            parameters=[float(lambda_hat)],
            std_errors=std_errors,
            conf_int_lower=conf_int_lower,
            conf_int_upper=conf_int_upper,
            log_likelihood=log_likelihood,
            aic=float(aic),
            bic=float(bic),
            convergence=True,
            n_obs=n,
            param_names=["lambda"]
        )
    except Exception as e:
        raise ValueError(f"指数分布MLE估计失败: {str(e)}")