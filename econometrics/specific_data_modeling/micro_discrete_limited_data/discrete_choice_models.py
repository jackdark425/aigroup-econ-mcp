"""
离散选择模型模块
基于statsmodels等现有库实现
"""

import numpy as np
import pandas as pd
try:
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import Logit, Probit, MNLogit
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    # 注意: statsmodels目前没有内置ConditionalLogit，需要自定义或使用其他库
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    Logit = Probit = MNLogit = OrderedModel = None

# 占位符类以防statsmodels不可用
class _PlaceholderModel:
    def __init__(self, *args, **kwargs):
        if not HAS_STATSMODELS:
            raise ImportError("需要安装statsmodels库: pip install statsmodels")
    
    def fit(self, *args, **kwargs):
        pass


class LogitModel:
    """
    Logistic回归模型 (基于statsmodels实现)
    """
    
    def __init__(self):
        self.model_ = None
        self.results_ = None
        self.fitted_ = False
        
    def fit(self, X, y):
        """拟合Logistic回归模型"""
        if not HAS_STATSMODELS:
            raise ImportError("需要安装statsmodels库: pip install statsmodels")
            
        X = np.array(X)
        y = np.array(y)
        
        # 添加常数项
        X_with_const = sm.add_constant(X)
        
        # 拟合模型
        self.model_ = Logit(y, X_with_const)
        self.results_ = self.model_.fit(disp=0)
        self.fitted_ = True
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        X = np.array(X)
        X_with_const = sm.add_constant(X)
        return self.results_.predict(X_with_const)
    
    def predict(self, X, threshold=0.5):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def summary(self):
        """返回模型摘要"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        return self.results_.summary()


class ProbitModel:
    """
    Probit回归模型 (基于statsmodels实现)
    """
    
    def __init__(self):
        self.model_ = None
        self.results_ = None
        self.fitted_ = False
        
    def fit(self, X, y):
        """拟合Probit回归模型"""
        if not HAS_STATSMODELS:
            raise ImportError("需要安装statsmodels库: pip install statsmodels")
            
        X = np.array(X)
        y = np.array(y)
        
        # 添加常数项
        X_with_const = sm.add_constant(X)
        
        # 拟合模型
        self.model_ = Probit(y, X_with_const)
        self.results_ = self.model_.fit(disp=0)
        self.fitted_ = True
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        X = np.array(X)
        X_with_const = sm.add_constant(X)
        return self.results_.predict(X_with_const)
    
    def predict(self, X, threshold=0.5):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def summary(self):
        """返回模型摘要"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        return self.results_.summary()


class MultinomialLogit:
    """
    多项Logit模型 (基于statsmodels实现)
    """
    
    def __init__(self):
        self.model_ = None
        self.results_ = None
        self.fitted_ = False
        self.classes_ = None
        
    def fit(self, X, y):
        """拟合多项Logit模型"""
        if not HAS_STATSMODELS:
            raise ImportError("需要安装statsmodels库: pip install statsmodels")
            
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        
        # 添加常数项
        X_with_const = sm.add_constant(X)
        
        # 拟合模型
        self.model_ = MNLogit(y, X_with_const)
        self.results_ = self.model_.fit(disp=0)
        self.fitted_ = True
        return self
    
    def predict_proba(self, X):
        """预测各类别的概率"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        X = np.array(X)
        X_with_const = sm.add_constant(X)
        return self.results_.predict(X_with_const)
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def summary(self):
        """返回模型摘要"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        return self.results_.summary()


class OrderedLogit:
    """
    有序Logit模型 (基于statsmodels实现)
    """
    
    def __init__(self):
        self.model_ = None
        self.results_ = None
        self.fitted_ = False
        self.classes_ = None
        
    def fit(self, X, y):
        """拟合有序Logit模型"""
        if not HAS_STATSMODELS:
            raise ImportError("需要安装statsmodels库: pip install statsmodels")
            
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        
        # OrderedModel不允许包含常数项
        # 直接使用X，不添加常数项
        self.model_ = OrderedModel(y, X, distr='logit')
        self.results_ = self.model_.fit(method='bfgs', disp=0)
        self.fitted_ = True
        return self
    
    def predict_proba(self, X):
        """预测各类别的概率"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        X = np.array(X)
        # 预测时也不添加常数项
        return self.results_.predict(X)
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def summary(self):
        """返回模型摘要"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        return self.results_.summary()


class ConditionalLogit:
    """
    条件Logit模型 
    注意: statsmodels目前没有内置实现，此为简化版本
    """
    
    def __init__(self):
        self.params_ = None
        self.fitted_ = False
        
    def fit(self, X, y, groups):
        """拟合条件Logit模型"""
        # 简化的条件Logit实现
        # 在实际应用中可能需要使用其他专门库如pylogit或mne-logit
        X = np.array(X)
        y = np.array(y)
        groups = np.array(groups)
        
        n_samples, n_features = X.shape
        
        # 使用scipy优化器进行简单实现
        from scipy.optimize import minimize
        
        def neg_log_likelihood(params):
            beta = params
            loglik = 0
            unique_groups = np.unique(groups)
            
            for group_id in unique_groups:
                group_mask = (groups == group_id)
                X_g = X[group_mask]
                y_g = y[group_mask]
                
                scores = np.dot(X_g, beta)
                probs = np.exp(scores)
                probs = probs / np.sum(probs)
                probs = np.clip(probs, 1e-15, 1-1e-15)
                
                loglik += np.sum(y_g * np.log(probs))
            
            return -loglik
        
        initial_params = np.random.normal(0, 0.1, n_features)
        result = minimize(neg_log_likelihood, initial_params, method='BFGS')
        
        if result.success:
            self.params_ = result.x
            self.fitted_ = True
        else:
            raise RuntimeError("模型优化失败")
            
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        X = np.array(X)
        scores = np.dot(X, self.params_)
        exp_scores = np.exp(scores - np.max(scores))  # 数值稳定性
        return exp_scores / np.sum(exp_scores)
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)


# 如果statsmodels不可用，则使用占位符
if not HAS_STATSMODELS:
    LogitModel = _PlaceholderModel
    ProbitModel = _PlaceholderModel
    MultinomialLogit = _PlaceholderModel
    OrderedLogit = _PlaceholderModel
    ConditionalLogit = _PlaceholderModel