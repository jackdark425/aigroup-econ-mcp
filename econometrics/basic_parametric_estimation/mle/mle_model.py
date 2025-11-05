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
    
    data = np.array(data)
    n = len(data)
    
    # 分布特定的验证
    if distribution == "exponential" and np.any(data < 0):
        raise ValueError("指数分布的数据必须为非负数")
    
    if distribution == "poisson" and np.any(data < 0):
        raise ValueError("泊松分布的数据必须为非负数")
    
    if distribution == "normal":
        # 正态分布的MLE
        return _normal_mle_statsmodels(data, initial_params, confidence_level)
    elif distribution == "poisson":
        # 泊松分布的MLE
        return _poisson_mle_statsmodels(data, initial_params, confidence_level)
    elif distribution == "exponential":
        # 指数分布的MLE
        return _exponential_mle_statsmodels(data, initial_params, confidence_level)
    else:
        raise ValueError(f"不支持的分布类型: {distribution}")


def _normal_mle_statsmodels(data: np.ndarray, initial_params: Optional[List[float]], confidence_level: float) -> MLEResult:
    """使用statsmodels的正态分布最大似然估计"""
    try:
        # 检查数据有效性
        if len(data) == 1:
            # 单个数据点的特殊情况
            mu_hat = float(data[0])
            sigma_hat = 0.0
            log_likelihood = float(stats.norm.logpdf(data, loc=mu_hat, scale=1e-10).sum())
            std_errors = [0.0, 0.0]
            conf_int_lower = [mu_hat, sigma_hat]
            conf_int_upper = [mu_hat, sigma_hat]
        else:
            # 使用statsmodels的OLS作为初始估计
            X = np.ones((len(data), 1))
            mu_hat = np.mean(data)
            sigma_hat = np.std(data, ddof=1)  # 使用样本标准差
            
            # 计算对数似然值
            log_likelihood = float(np.sum(stats.norm.logpdf(data, loc=mu_hat, scale=sigma_hat)))
            
            # 标准误
            std_error_mu = sigma_hat / np.sqrt(len(data))
            std_error_sigma = sigma_hat / np.sqrt(2 * len(data))
            std_errors = [std_error_mu, std_error_sigma]
            
            # 置信区间
            alpha = 1 - confidence_level
            z_critical = stats.norm.ppf(1 - alpha/2)
            conf_int_lower = [mu_hat - z_critical * std_error_mu, 
                              sigma_hat - z_critical * std_error_sigma]
            conf_int_upper = [mu_hat + z_critical * std_error_mu, 
                              sigma_hat + z_critical * std_error_sigma]
        
        # 信息准则
        k = 2  # 参数数量
        aic = 2*k - 2*log_likelihood
        bic = k*np.log(len(data)) - 2*log_likelihood
        
        return MLEResult(
            parameters=[float(mu_hat), float(sigma_hat)],
            std_errors=std_errors,
            conf_int_lower=conf_int_lower,
            conf_int_upper=conf_int_upper,
            log_likelihood=float(log_likelihood),
            aic=float(aic),
            bic=float(bic),
            convergence=True,
            n_obs=len(data),
            param_names=["mu", "sigma"]
        )
    except Exception as e:
        # 出现错误时使用原始实现
        return _normal_mle(data, initial_params, confidence_level)


def _poisson_mle_statsmodels(data: np.ndarray, initial_params: Optional[List[float]], confidence_level: float) -> MLEResult:
    """使用statsmodels的泊松分布最大似然估计"""
    try:
        # 泊松分布的MLE有一个闭式解: lambda_hat = mean(data)
        lambda_hat = np.mean(data)
        
        if len(data) == 1:
            # 单个数据点的特殊情况
            log_likelihood = float(stats.poisson.logpmf(data, lambda_hat).sum())
            std_error = 0.0
            std_errors = [std_error]
            conf_int_lower = [lambda_hat]
            conf_int_upper = [lambda_hat]
        else:
            log_likelihood = float(np.sum(stats.poisson.logpmf(data, lambda_hat)))
            
            # 标准误: sqrt(lambda/n)
            std_error = np.sqrt(lambda_hat / len(data))
            std_errors = [float(std_error)]
            
            # 置信区间
            alpha = 1 - confidence_level
            z_critical = stats.norm.ppf(1 - alpha/2)
            conf_int_lower = [lambda_hat - z_critical * std_error]
            conf_int_upper = [lambda_hat + z_critical * std_error]
        
        # 信息准则
        k = 1  # 参数数量
        aic = 2*k - 2*log_likelihood
        bic = k*np.log(len(data)) - 2*log_likelihood
        
        return MLEResult(
            parameters=[float(lambda_hat)],
            std_errors=std_errors,
            conf_int_lower=conf_int_lower,
            conf_int_upper=conf_int_upper,
            log_likelihood=float(log_likelihood),
            aic=float(aic),
            bic=float(bic),
            convergence=True,
            n_obs=len(data),
            param_names=["lambda"]
        )
    except Exception as e:
        # 出现错误时使用原始实现
        return _poisson_mle(data, initial_params, confidence_level)


def _exponential_mle_statsmodels(data: np.ndarray, initial_params: Optional[List[float]], confidence_level: float) -> MLEResult:
    """使用statsmodels的指数分布最大似然估计"""
    try:
        # 指数分布的MLE有一个闭式解: lambda_hat = 1/mean(data)
        lambda_hat = 1.0 / np.mean(data)
        
        if len(data) == 1:
            # 单个数据点的特殊情况
            log_likelihood = float(stats.expon.logpdf(data, scale=1/lambda_hat).sum())
            std_error = 0.0
            std_errors = [std_error]
            conf_int_lower = [lambda_hat]
            conf_int_upper = [lambda_hat]
        else:
            log_likelihood = float(np.sum(stats.expon.logpdf(data, scale=1/lambda_hat)))
            
            # 标准误: lambda/sqrt(n)
            std_error = lambda_hat / np.sqrt(len(data))
            std_errors = [float(std_error)]
            
            # 置信区间
            alpha = 1 - confidence_level
            z_critical = stats.norm.ppf(1 - alpha/2)
            conf_int_lower = [lambda_hat - z_critical * std_error]
            conf_int_upper = [lambda_hat + z_critical * std_error]
        
        # 信息准则
        k = 1  # 参数数量
        aic = 2*k - 2*log_likelihood
        bic = k*np.log(len(data)) - 2*log_likelihood
        
        return MLEResult(
            parameters=[float(lambda_hat)],
            std_errors=std_errors,
            conf_int_lower=conf_int_lower,
            conf_int_upper=conf_int_upper,
            log_likelihood=float(log_likelihood),
            aic=float(aic),
            bic=float(bic),
            convergence=True,
            n_obs=len(data),
            param_names=["lambda"]
        )
    except Exception as e:
        # 出现错误时使用原始实现
        return _exponential_mle(data, initial_params, confidence_level)


def _normal_mle(data: np.ndarray, initial_params: Optional[List[float]], confidence_level: float) -> MLEResult:
    """正态分布的最大似然估计"""
    n = len(data)
    
    # 默认初始参数: [均值, 标准差]
    if initial_params is None:
        initial_params = [np.mean(data), np.std(data)]
    
    # 定义负对数似然函数
    def neg_log_likelihood(params):
        mu, sigma = params
        if sigma <= 0:
            return np.inf
        return -np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))
    
    # 优化
    result = minimize(neg_log_likelihood, initial_params, method='BFGS')
    
    if not result.success:
        convergence = False
    else:
        convergence = True
    
    mu_hat, sigma_hat = result.x
    log_likelihood = -result.fun
    
    # 计算标准误 (使用Fisher信息矩阵的逆)
    # 对于正态分布，Fisher信息矩阵是对角的
    fisher_info = np.array([[n/sigma_hat**2, 0], [0, n/(2*sigma_hat**2)]])
    cov_matrix = np.linalg.inv(fisher_info)
    std_errors = np.sqrt(np.diag(cov_matrix))
    
    # 置信区间
    alpha = 1 - confidence_level
    z_critical = stats.norm.ppf(1 - alpha/2)
    conf_int_lower = result.x - z_critical * std_errors
    conf_int_upper = result.x + z_critical * std_errors
    
    # 信息准则
    k = len(result.x)  # 参数数量
    aic = 2*k - 2*log_likelihood
    bic = k*np.log(n) - 2*log_likelihood
    
    return MLEResult(
        parameters=result.x.tolist(),
        std_errors=std_errors.tolist(),
        conf_int_lower=conf_int_lower.tolist(),
        conf_int_upper=conf_int_upper.tolist(),
        log_likelihood=float(log_likelihood),
        aic=float(aic),
        bic=float(bic),
        convergence=convergence,
        n_obs=n,
        param_names=["mu", "sigma"]
    )


def _poisson_mle(data: np.ndarray, initial_params: Optional[List[float]], confidence_level: float) -> MLEResult:
    """泊松分布的最大似然估计"""
    n = len(data)
    
    # 默认初始参数: [lambda]
    if initial_params is None:
        initial_params = [np.mean(data)]
    
    # 泊松分布的MLE有一个闭式解: lambda_hat = mean(data)
    lambda_hat = np.mean(data)
    log_likelihood = np.sum(stats.poisson.logpmf(data, lambda_hat))
    
    # 标准误: sqrt(lambda/n)
    std_error = np.sqrt(lambda_hat / n)
    
    # 置信区间
    alpha = 1 - confidence_level
    z_critical = stats.norm.ppf(1 - alpha/2)
    conf_int_lower = [lambda_hat - z_critical * std_error]
    conf_int_upper = [lambda_hat + z_critical * std_error]
    
    # 信息准则
    k = 1  # 参数数量
    aic = 2*k - 2*log_likelihood
    bic = k*np.log(n) - 2*log_likelihood
    
    return MLEResult(
        parameters=[float(lambda_hat)],
        std_errors=[float(std_error)],
        conf_int_lower=conf_int_lower,
        conf_int_upper=conf_int_upper,
        log_likelihood=float(log_likelihood),
        aic=float(aic),
        bic=float(bic),
        convergence=True,
        n_obs=n,
        param_names=["lambda"]
    )


def _exponential_mle(data: np.ndarray, initial_params: Optional[List[float]], confidence_level: float) -> MLEResult:
    """指数分布的最大似然估计"""
    n = len(data)
    
    # 指数分布的MLE有一个闭式解: lambda_hat = 1/mean(data)
    lambda_hat = 1.0 / np.mean(data)
    log_likelihood = np.sum(stats.expon.logpdf(data, scale=1/lambda_hat))
    
    # 标准误: lambda/sqrt(n)
    std_error = lambda_hat / np.sqrt(n)
    
    # 置信区间
    alpha = 1 - confidence_level
    z_critical = stats.norm.ppf(1 - alpha/2)
    conf_int_lower = [lambda_hat - z_critical * std_error]
    conf_int_upper = [lambda_hat + z_critical * std_error]
    
    # 信息准则
    k = 1  # 参数数量
    aic = 2*k - 2*log_likelihood
    bic = k*np.log(n) - 2*log_likelihood
    
    return MLEResult(
        parameters=[float(lambda_hat)],
        std_errors=[float(std_error)],
        conf_int_lower=conf_int_lower,
        conf_int_upper=conf_int_upper,
        log_likelihood=float(log_likelihood),
        aic=float(aic),
        bic=float(bic),
        convergence=True,
        n_obs=n,
        param_names=["lambda"]
    )