"""
微观离散与受限数据模型测试
"""

import numpy as np
import pandas as pd
import sys
import os
from scipy import stats

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

try:
    from econometrics.specific_data_modeling.micro_discrete_limited_data import (
        LogitModel,
        ProbitModel,
        TobitModel,
        PoissonModel,
        NegativeBinomialModel
    )
    HAS_MODELS = True
except ImportError as e:
    print(f"导入模型时出错: {e}")
    HAS_MODELS = False


def test_logit_model():
    """测试Logit模型"""
    print("测试Logit模型...")
    
    # 生成模拟数据
    np.random.seed(42)
    n = 1000
    X = np.random.normal(0, 1, (n, 2))
    coef_true = np.array([1.0, -0.5])
    linear_pred = np.dot(X, coef_true)
    prob = 1 / (1 + np.exp(-linear_pred))
    y = np.random.binomial(1, prob)
    
    # 拟合模型
    model = LogitModel()
    model.fit(X, y)
    
    print(f"真实系数: {coef_true}")
    print(f"估计系数: {model.results_.params[1:]}")  # 排除常数项
    print(f"常数项: {model.results_.params[0]}")
    print(f"AIC: {model.results_.aic}")
    print(f"BIC: {model.results_.bic}")
    print()


def test_probit_model():
    """测试Probit模型"""
    print("测试Probit模型...")
    
    # 生成模拟数据
    np.random.seed(42)
    n = 1000
    X = np.random.normal(0, 1, (n, 2))
    coef_true = np.array([0.5, -0.3])
    linear_pred = np.dot(X, coef_true)
    prob = np.clip(stats.norm.cdf(linear_pred), 1e-10, 1-1e-10)
    y = np.random.binomial(1, prob)
    
    # 拟合模型
    model = ProbitModel()
    model.fit(X, y)
    
    print(f"真实系数: {coef_true}")
    print(f"估计系数: {model.results_.params[1:]}")  # 排除常数项
    print(f"常数项: {model.results_.params[0]}")
    print(f"AIC: {model.results_.aic}")
    print(f"BIC: {model.results_.bic}")
    print()


def test_tobit_model():
    """测试Tobit模型"""
    print("测试Tobit模型...")
    
    # 生成模拟数据
    np.random.seed(42)
    n = 1000
    X = np.random.normal(0, 1, (n, 2))
    coef_true = np.array([1.0, -0.5])
    sigma_true = 0.5
    
    # 生成潜在变量
    y_latent = np.dot(X, coef_true) + np.random.normal(0, sigma_true, n)
    
    # 截断：低于0的值设为0
    y = np.where(y_latent > 0, y_latent, 0)
    
    # 拟合模型
    model = TobitModel(lower_bound=0)
    model.fit(X, y)
    
    print(f"真实系数: {coef_true}")
    print(f"估计系数: {model.results_.params[1:]}")  # 排除常数项
    print(f"常数项: {model.results_.params[0]}")
    print(f"对数似然: {model.results_.llf}")
    print()


def test_poisson_model():
    """测试泊松模型"""
    print("测试泊松模型...")
    
    # 生成模拟数据
    np.random.seed(42)
    n = 1000
    X = np.random.normal(0, 1, (n, 2))
    coef_true = np.array([0.5, -0.3])
    mu = np.exp(np.dot(X, coef_true))
    y = np.random.poisson(mu)
    
    # 拟合模型
    model = PoissonModel()
    model.fit(X, y)
    
    print(f"真实系数: {coef_true}")
    print(f"估计系数: {model.results_.params[1:]}")  # 排除常数项
    print(f"常数项: {model.results_.params[0]}")
    print(f"对数似然: {model.results_.llf}")
    print(f"AIC: {model.results_.aic}")
    print(f"BIC: {model.results_.bic}")
    
    # 预测测试
    y_pred = model.predict(X[:5])
    print(f"前5个样本的预测值: {y_pred}")
    print()


def test_negative_binomial_model():
    """测试负二项模型"""
    print("测试负二项模型...")
    
    # 生成模拟数据
    np.random.seed(42)
    n = 1000
    X = np.random.normal(0, 1, (n, 2))
    coef_true = np.array([0.5, -0.3])
    
    mu = np.exp(np.dot(X, coef_true))
    # 生成负二项分布数据
    alpha = 0.5
    size = 1.0 / alpha
    prob = size / (size + mu)
    y = np.random.negative_binomial(size, prob)
    
    # 拟合模型
    model = NegativeBinomialModel()
    model.fit(X, y)
    
    print(f"真实系数: {coef_true}")
    print(f"估计系数: {model.results_.params[1:-1]}")  # 排除常数项和alpha参数
    print(f"常数项: {model.results_.params[0]}")
    print(f"对数似然: {model.results_.llf}")
    print(f"AIC: {model.results_.aic}")
    print(f"BIC: {model.results_.bic}")
    
    # 预测测试
    y_pred = model.predict(X[:5])
    print(f"前5个样本的预测值: {y_pred}")
    print()


if __name__ == "__main__":
    if not HAS_MODELS:
        print("模型不可用，请确保已安装statsmodels库")
        exit(1)
        
    try:
        import statsmodels.api as sm
    except ImportError:
        print("需要安装statsmodels库")
        exit(1)
    
    print("微观离散与受限数据模型测试")
    print("=" * 50)
    
    test_logit_model()
    test_probit_model()
    test_tobit_model()
    test_poisson_model()
    test_negative_binomial_model()
    
    print("所有测试完成！")