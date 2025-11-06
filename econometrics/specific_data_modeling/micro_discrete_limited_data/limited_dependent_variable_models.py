"""
受限因变量模型模块
基于statsmodels等现有库实现
"""

import numpy as np
import pandas as pd
from scipy import stats
try:
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    OLS = None

try:
    from statsmodels.base.model import GenericLikelihoodModel
    HAS_GENERIC_MODEL = True
except ImportError:
    HAS_GENERIC_MODEL = False


class _PlaceholderModel:
    def __init__(self, *args, **kwargs):
        if not HAS_STATSMODELS:
            raise ImportError("需要安装statsmodels库: pip install statsmodels")
    
    def fit(self, *args, **kwargs):
        pass


class TobitModel:
    """
    Tobit模型（截断回归模型）
    由于statsmodels中没有内置的Tobit模型，这里提供一个基于GenericLikelihoodModel的实现
    """
    
    def __init__(self, lower_bound=0, upper_bound=None):
        """
        初始化Tobit模型
        
        参数:
        lower_bound: 下界阈值，默认为0
        upper_bound: 上界阈值，默认为None（无上界）
        """
        if not HAS_STATSMODELS:
            raise ImportError("需要安装statsmodels库: pip install statsmodels")
            
        if not HAS_GENERIC_MODEL:
            raise ImportError("需要statsmodels的GenericLikelihoodModel支持")
            
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.model_ = None
        self.results_ = None
        self.fitted_ = False
        
    def fit(self, X, y):
        """拟合Tobit模型"""
        X = np.array(X)
        y = np.array(y)
        
        # 添加常数项
        X_with_const = sm.add_constant(X)
        
        # 定义Tobit似然函数
        class TobitLikelihoodModel(GenericLikelihoodModel):
            def __init__(self, endog, exog, lower_bound=0, upper_bound=None, **kwds):
                self.lower_bound = lower_bound
                self.upper_bound = upper_bound
                super(TobitLikelihoodModel, self).__init__(endog, exog, **kwds)
            
            def loglikeobs(self, params):
                # 分离系数和sigma
                beta = params[:-1]
                sigma = params[-1]
                
                if sigma <= 0:
                    return np.full_like(self.endog, -np.inf)
                
                # 预测值
                xb = np.dot(self.exog, beta)
                z = (self.endog - xb) / sigma
                
                # 计算对数似然
                if self.upper_bound is None:
                    # 只有下界的情况
                    censored = self.endog <= self.lower_bound
                    uncensored = ~censored
                    
                    ll = np.zeros_like(self.endog)
                    # 截断观测的对数似然
                    ll[censored] = stats.norm.logcdf((self.lower_bound - xb[censored]) / sigma)
                    # 未截断观测的对数似然
                    ll[uncensored] = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * z[uncensored]**2
                else:
                    # 双边截断的情况
                    left_censored = self.endog <= self.lower_bound
                    right_censored = self.endog >= self.upper_bound
                    uncensored = ~(left_censored | right_censored)
                    
                    ll = np.zeros_like(self.endog)
                    # 左截断观测的对数似然
                    ll[left_censored] = stats.norm.logcdf((self.lower_bound - xb[left_censored]) / sigma)
                    # 右截断观测的对数似然
                    ll[right_censored] = stats.norm.logsf((self.upper_bound - xb[right_censored]) / sigma)
                    # 未截断观测的对数似然
                    ll[uncensored] = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * z[uncensored]**2
                
                return ll
        
        # 创建并拟合模型
        self.model_ = TobitLikelihoodModel(
            endog=y, 
            exog=X_with_const, 
            lower_bound=self.lower_bound, 
            upper_bound=self.upper_bound
        )
        
        # 初始化参数
        n_features = X_with_const.shape[1]
        initial_params = np.concatenate([
            np.zeros(n_features),  # beta
            [np.std(y[y > self.lower_bound]) if self.upper_bound is None else np.std(y)]  # sigma
        ])
        
        self.results_ = self.model_.fit(start_params=initial_params, method='bfgs', disp=0)
        self.fitted_ = True
        return self
    
    def predict(self, X):
        """预测期望值"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        X = np.array(X)
        X_with_const = sm.add_constant(X)
        
        # 手动计算预测值
        beta = self.results_.params[:-1]  # 排除sigma参数
        sigma = self.results_.params[-1]
        
        xb = np.dot(X_with_const, beta)
        
        if self.upper_bound is None:
            # 只有下界的情况
            z = (self.lower_bound - xb) / sigma
            lambda_val = stats.norm.pdf(z) / np.clip(1 - stats.norm.cdf(z), 1e-10, 1)
            return xb + sigma * lambda_val
        else:
            # 双边截断的情况
            z_lower = (self.lower_bound - xb) / sigma
            z_upper = (self.upper_bound - xb) / sigma
            
            lambda_lower = stats.norm.pdf(z_lower) / np.clip(stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower), 1e-10, 1)
            lambda_upper = stats.norm.pdf(z_upper) / np.clip(stats.norm.cdf(z_upper) - stats.norm.cdf(z_lower), 1e-10, 1)
            
            return xb + sigma * (lambda_lower - lambda_upper)
    
    def predict_linear(self, X):
        """预测线性预测值"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        X = np.array(X)
        X_with_const = sm.add_constant(X)
        xb = np.dot(X_with_const, self.results_.params[:-1])  # 排除sigma参数
        return xb
    
    def summary(self):
        """返回模型摘要"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        return self.results_.summary()


class HeckmanModel:
    """
    Heckman两阶段选择模型 (基于statsmodels实现)
    """
    
    def __init__(self):
        self.selection_model_ = None
        self.selection_results_ = None
        self.outcome_model_ = None
        self.outcome_results_ = None
        self.fitted_ = False
        
    def fit(self, X_select, Z, y, s):
        """
        拟合Heckman模型
        
        参数:
        X_select: 选择方程的解释变量矩阵
        Z: 结果方程的解释变量矩阵
        y: 结果变量向量（仅对选择样本可观测）
        s: 选择指示变量向量（1表示被选择，0表示未被选择）
        """
        if not HAS_STATSMODELS:
            raise ImportError("需要安装statsmodels库: pip install statsmodels")
            
        X_select = np.array(X_select)
        Z = np.array(Z)
        y = np.array(y)
        s = np.array(s)
        
        # 第一阶段：Probit模型估计选择方程
        X_select_with_const = sm.add_constant(X_select)
        self.selection_model_ = sm.Probit(s, X_select_with_const)
        self.selection_results_ = self.selection_model_.fit(disp=0)
        
        # 计算逆米尔斯比率 (Inverse Mills Ratio)
        X_select_linpred = np.dot(X_select_with_const, self.selection_results_.params)
        mills_ratio = stats.norm.pdf(X_select_linpred) / np.clip(stats.norm.cdf(X_select_linpred), 1e-10, 1-1e-10)
        # 对于未被选择的样本，米尔斯比率为0
        mills_ratio = mills_ratio * s
        
        # 第二阶段：加入逆米尔斯比率的结果方程OLS
        Z_with_mills = np.column_stack([Z, mills_ratio])
        Z_with_mills_const = sm.add_constant(Z_with_mills)
        
        # 只对被选择的样本进行回归
        selected_mask = s == 1
        Z_selected = Z_with_mills_const[selected_mask]
        y_selected = y[selected_mask]
        
        self.outcome_model_ = OLS(y_selected, Z_selected)
        self.outcome_results_ = self.outcome_model_.fit()
        
        self.fitted_ = True
        return self
    
    def predict(self, X_select, Z):
        """预测结果值"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
            
        X_select = np.array(X_select)
        Z = np.array(Z)
        
        # 添加常数项
        X_select_with_const = sm.add_constant(X_select)
        
        # 计算逆米尔斯比率
        X_select_linpred = np.dot(X_select_with_const, self.selection_results_.params)
        mills_ratio = stats.norm.pdf(X_select_linpred) / np.clip(stats.norm.cdf(X_select_linpred), 1e-10, 1-1e-10)
        
        # 构建预测矩阵：Z + 逆米尔斯比率 + 常数项
        Z_with_mills = np.column_stack([Z, mills_ratio])
        Z_with_mills_const = sm.add_constant(Z_with_mills)
        
        # 计算结果方程预测值
        outcome_pred = self.outcome_results_.predict(Z_with_mills_const)
        
        return outcome_pred
    
    def summary(self):
        """返回模型摘要"""
        if not self.fitted_:
            raise ValueError("模型尚未拟合")
        return {
            'selection_summary': self.selection_results_.summary(),
            'outcome_summary': self.outcome_results_.summary()
        }


# 如果statsmodels不可用，则使用占位符
if not HAS_STATSMODELS:
    TobitModel = _PlaceholderModel
    HeckmanModel = _PlaceholderModel

def multinomial_logit():
    """
    多项Logit模型占位符
    """
    pass


def nested_logit():
    """
    嵌套Logit模型占位符
    """
    pass