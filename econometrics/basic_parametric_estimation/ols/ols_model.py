"""
普通最小二乘法 (OLS) 模型实现
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


class OLSResult(BaseModel):
    """OLS回归结果"""
    coefficients: List[float] = Field(..., description="回归系数")
    std_errors: List[float] = Field(..., description="系数标准误")
    t_values: List[float] = Field(..., description="t统计量")
    p_values: List[float] = Field(..., description="p值")
    conf_int_lower: List[float] = Field(..., description="置信区间下界")
    conf_int_upper: List[float] = Field(..., description="置信区间上界")
    r_squared: float = Field(..., description="R方")
    adj_r_squared: float = Field(..., description="调整R方")
    f_statistic: float = Field(..., description="F统计量")
    f_p_value: float = Field(..., description="F统计量p值")
    aic: float = Field(..., description="赤池信息准则")
    bic: float = Field(..., description="贝叶斯信息准则")
    n_obs: int = Field(..., description="观测数量")
    feature_names: List[str] = Field(..., description="特征名称")


def ols_regression(
    y_data: List[float],
    x_data: List[List[float]], 
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    confidence_level: float = 0.95
) -> OLSResult:
    """
    普通最小二乘法回归
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        feature_names: 特征名称
        constant: 是否包含常数项
        confidence_level: 置信水平
        
    Returns:
        OLSResult: OLS回归结果
        
    Raises:
        ValueError: 当输入数据无效时抛出异常
    """
    # 输入验证
    if not y_data or not x_data:
        raise ValueError("因变量和自变量数据不能为空")
    
    # 转换为numpy数组
    y = np.array(y_data, dtype=np.float64)
    
    # 确保X是二维数组
    if x_data and isinstance(x_data[0], (int, float)):
        # 单个特征的情况，需要转置
        X = np.array(x_data, dtype=np.float64).reshape(-1, 1)
    else:
        X = np.array(x_data, dtype=np.float64)
    
    # 验证数据维度一致性
    if len(y) != X.shape[0]:
        raise ValueError(f"因变量长度({len(y)})与自变量长度({X.shape[0]})不一致")
    
    # 检查是否有足够的数据点
    if len(y) < X.shape[1] + (1 if constant else 0):
        raise ValueError("数据点数量不足以估计模型参数")
    
    # 检查是否存在缺失值或无穷大值
    if np.isnan(y).any() or np.isnan(X).any():
        raise ValueError("数据中包含缺失值(NaN)")
    
    if np.isinf(y).any() or np.isinf(X).any():
        raise ValueError("数据中包含无穷大值")
    
    # 添加常数项
    if constant:
        X = sm.add_constant(X)
        if feature_names:
            feature_names = ["const"] + feature_names
        else:
            feature_names = ["const"] + [f"x{i}" for i in range(X.shape[1]-1)]
    else:
        if not feature_names:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    
    # 使用statsmodels执行OLS回归
    try:
        model = sm.OLS(y, X)
        results = model.fit()
    except Exception as e:
        raise ValueError(f"无法拟合OLS模型: {str(e)}")
    
    # 提取结果
    coefficients = results.params.tolist()
    std_errors = results.bse.tolist()
    t_values = results.tvalues.tolist()
    p_values = results.pvalues.tolist()
    
    # 计算置信区间
    alpha = 1 - confidence_level
    conf_int = results.conf_int(alpha=alpha)
    conf_int_lower = conf_int[:, 0].tolist()
    conf_int_upper = conf_int[:, 1].tolist()
    
    # 其他统计量
    r_squared = float(results.rsquared)
    adj_r_squared = float(results.rsquared_adj)
    
    # F统计量
    f_statistic = float(results.fvalue) if not np.isnan(results.fvalue) else 0.0
    f_p_value = float(results.f_pvalue) if not np.isnan(results.f_pvalue) else 1.0
    
    # 信息准则
    aic = float(results.aic)
    bic = float(results.bic)
    
    return OLSResult(
        coefficients=coefficients,
        std_errors=std_errors,
        t_values=t_values,
        p_values=p_values,
        conf_int_lower=conf_int_lower,
        conf_int_upper=conf_int_upper,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        f_statistic=f_statistic,
        f_p_value=f_p_value,
        aic=aic,
        bic=bic,
        n_obs=int(results.nobs),
        feature_names=feature_names
    )