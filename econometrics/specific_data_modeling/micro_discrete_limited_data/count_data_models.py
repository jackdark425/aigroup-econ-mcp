"""
计数数据模型模块
基于statsmodels等现有库实现
"""

import numpy as np
import pandas as pd
import math
from scipy import stats
try:
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import Poisson, NegativeBinomial
    from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    Poisson = NegativeBinomial = ZeroInflatedPoisson = ZeroInflatedNegativeBinomialP = None


class _PlaceholderModel:
    def __init__(self, *args, **kwargs):
        if not HAS_STATSMODELS:
            raise ImportError("需要安装statsmodels库: pip install statsmodels")
    
    def fit(self, *args, **kwargs):
        pass


class PoissonModel:
    """
    泊松回归模型 (基于statsmodels实现)
    """
    
    def __init__(self):
        if not HAS_STATSMODELS:
            raise ImportError("需要安装statsmodels库: pip install statsmodels")
        self.model_ = None
        self.results_ = None
        self.fitted_ = False
        
    def fit(self, X, y):
        """拟合泊松回归模型"""
        X = np.array(X)
        y = np.array(y)
        
        if np.any(y < 0) or np.any(y != np.floor(y)):
            raise ValueError("因变量必须是非负整数")
        
        # 添加常数项
        X_with_const = sm.add_constant(X)
        
        # 拟合模型
        self.model_ = Poisson(y, X_with_const)
        self.results_ = self.model_.fit(disp=0)
        self.fitted_ = True
        return self
    
    def predict(self, X):
        """预测计数期望值"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        X = np.array(X)
        X_with_const = sm.add_constant(X)
        return self.results_.predict(X_with_const)
    
    def predict_proba(self, X, max_count=20):
        """预测计数概率分布"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        # 使用statsmodels的预测方法
        X = np.array(X)
        X_with_const = sm.add_constant(X)
        mu = self.results_.predict(X_with_const)
        
        # 计算泊松概率
        probas = []
        for k in range(max_count + 1):
            prob = np.exp(-mu) * (mu ** k) / math.factorial(k)
            probas.append(prob)
        
        return np.array(probas).T
    
    def summary(self):
        """返回模型摘要"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        return self.results_.summary()


class NegativeBinomialModel:
    """
    负二项回归模型 (基于statsmodels实现)
    """
    
    def __init__(self, distr='nb2'):
        """
        初始化负二项回归模型
        
        参数:
        distr: 分布类型，'nb1' 或 'nb2' (默认)
        """
        if not HAS_STATSMODELS:
            raise ImportError("需要安装statsmodels库: pip install statsmodels")
        self.distr = distr
        self.model_ = None
        self.results_ = None
        self.fitted_ = False
        
    def fit(self, X, y):
        """拟合负二项回归模型"""
        X = np.array(X)
        y = np.array(y)
        
        if np.any(y < 0) or np.any(y != np.floor(y)):
            raise ValueError("因变量必须是非负整数")
        
        # 添加常数项
        X_with_const = sm.add_constant(X)
        
        # 拟合模型
        self.model_ = NegativeBinomial(y, X_with_const, loglike_method=self.distr)
        self.results_ = self.model_.fit(disp=0)
        self.fitted_ = True
        return self
    
    def predict(self, X):
        """预测计数期望值"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        X = np.array(X)
        X_with_const = sm.add_constant(X)
        return self.results_.predict(X_with_const)
    
    def predict_proba(self, X, max_count=20):
        """预测计数概率分布"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        # 使用模型预测均值
        X = np.array(X)
        X_with_const = sm.add_constant(X)
        mu = self.results_.predict(X_with_const)
        
        # 获取alpha参数
        alpha = self.results_.params[-1]  # 最后一个参数是ln(alpha)
        alpha = np.exp(alpha)
        
        # 计算负二项概率 (NB2参数化)
        probas = []
        for k in range(max_count + 1):
            # 负二项概率质量函数 - 使用scipy.stats
            prob = stats.nbinom.pmf(k, 1/alpha, 1/(1 + alpha * mu))
            probas.append(prob)
        
        return np.array(probas).T
    
    def summary(self):
        """返回模型摘要"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        return self.results_.summary()


class ZeroInflatedPoissonModel:
    """
    零膨胀泊松模型 (基于statsmodels实现)
    """
    
    def __init__(self, exog_infl=None):
        """
        初始化零膨胀泊松模型
        
        参数:
        exog_infl: 用于零膨胀部分的解释变量，默认为None（使用与计数部分相同的变量）
        """
        if not HAS_STATSMODELS:
            raise ImportError("需要安装statsmodels库: pip install statsmodels")
        self.exog_infl = exog_infl
        self.model_ = None
        self.results_ = None
        self.fitted_ = False
        
    def fit(self, X, y):
        """拟合零膨胀泊松模型"""
        X = np.array(X)
        y = np.array(y)
        
        if np.any(y < 0) or np.any(y != np.floor(y)):
            raise ValueError("因变量必须是非负整数")
        
        # 添加常数项
        X_with_const = sm.add_constant(X)
        
        # 零膨胀部分的解释变量
        if self.exog_infl is not None:
            exog_infl = sm.add_constant(np.array(self.exog_infl))
        else:
            exog_infl = X_with_const
        
        # 拟合模型
        self.model_ = ZeroInflatedPoisson(
            endog=y, 
            exog=X_with_const, 
            exog_infl=exog_infl,
            inflation='logit'
        )
        self.results_ = self.model_.fit(disp=0)
        self.fitted_ = True
        return self
    
    def predict(self, X):
        """预测计数期望值"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        X = np.array(X)
        X_with_const = sm.add_constant(X)
        return self.results_.predict(X_with_const)
    
    def predict_proba(self, X):
        """预测计数概率分布"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        X = np.array(X)
        X_with_const = sm.add_constant(X)
        return self.results_.predict(which='prob', exog=X_with_const)
    
    def summary(self):
        """返回模型摘要"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        return self.results_.summary()


class ZeroInflatedNegativeBinomialModel:
    """
    零膨胀负二项模型 (基于statsmodels实现)
    """
    
    def __init__(self, exog_infl=None, distr='nb2'):
        """
        初始化零膨胀负二项模型
        
        参数:
        exog_infl: 用于零膨胀部分的解释变量，默认为None（使用与计数部分相同的变量）
        distr: 分布类型，'nb1' 或 'nb2' (默认)
        """
        if not HAS_STATSMODELS:
            raise ImportError("需要安装statsmodels库: pip install statsmodels")
        self.exog_infl = exog_infl
        self.distr = distr
        self.model_ = None
        self.results_ = None
        self.fitted_ = False
        
    def fit(self, X, y):
        """拟合零膨胀负二项模型"""
        X = np.array(X)
        y = np.array(y)
        
        if np.any(y < 0) or np.any(y != np.floor(y)):
            raise ValueError("因变量必须是非负整数")
        
        # 添加常数项
        X_with_const = sm.add_constant(X)
        
        # 零膨胀部分的解释变量
        if self.exog_infl is not None:
            exog_infl = sm.add_constant(np.array(self.exog_infl))
        else:
            exog_infl = X_with_const
        
        # 拟合模型
        self.model_ = ZeroInflatedNegativeBinomialP(
            endog=y, 
            exog=X_with_const, 
            exog_infl=exog_infl,
            inflation='logit',
            loglike_method=self.distr
        )
        self.results_ = self.model_.fit(disp=0)
        self.fitted_ = True
        return self
    
    def predict(self, X):
        """预测计数期望值"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        X = np.array(X)
        X_with_const = sm.add_constant(X)
        return self.results_.predict(X_with_const)
    
    def predict_proba(self, X):
        """预测计数概率分布"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        X = np.array(X)
        X_with_const = sm.add_constant(X)
        return self.results_.predict(which='prob', exog=X_with_const)
    
    def summary(self):
        """返回模型摘要"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        return self.results_.summary()


# 如果statsmodels不可用，则使用占位符
if not HAS_STATSMODELS:
    PoissonModel = _PlaceholderModel
    NegativeBinomialModel = _PlaceholderModel
    ZeroInflatedPoissonModel = _PlaceholderModel
    ZeroInflatedNegativeBinomialModel = _PlaceholderModel