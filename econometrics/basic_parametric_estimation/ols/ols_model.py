"""
普通最小二乘法 (OLS) 模型实现
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from scipy import stats


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
    y = np.array(y_data)
    
    # 确保X是二维数组
    if isinstance(x_data[0], (int, float)):
        # 单个特征的情况，需要转置
        X = np.array(x_data).reshape(-1, 1)
    else:
        X = np.array(x_data)  # 移除转置，保持原始维度
    
    # 验证数据维度一致性
    if len(y) != X.shape[0]:
        raise ValueError(f"因变量长度({len(y)})与自变量长度({X.shape[0]})不一致")
    
    # 特殊情况处理：单个观测值
    if len(y) == 1:
        if constant:
            # 只有一个观测值且包含常数项，无法估计标准误
            coefficients = [float(y[0]), 0.0]  # 截距为y值，斜率为0
            std_errors = [0.0, 0.0]
            t_values = [np.inf, np.inf]
            p_values = [0.0, 0.0]
            conf_int_lower = [float(y[0]), 0.0]
            conf_int_upper = [float(y[0]), 0.0]
            r_squared = 1.0
            adj_r_squared = np.nan
            f_statistic = np.inf
            f_p_value = 0.0
            aic = np.inf
            bic = np.inf
            
            if feature_names:
                feature_names = ["const"] + feature_names
            else:
                feature_names = ["const", "x0"]
        else:
            # 只有一个观测值且不包含常数项
            coefficients = [float(y[0])]  # 系数为y值
            std_errors = [0.0]
            t_values = [np.inf]
            p_values = [0.0]
            conf_int_lower = [float(y[0])]
            conf_int_upper = [float(y[0])]
            r_squared = 1.0
            adj_r_squared = np.nan
            f_statistic = np.inf
            f_p_value = 0.0
            aic = np.inf
            bic = np.inf
            
            if not feature_names:
                feature_names = ["x0"]
        
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
            n_obs=1,
            feature_names=feature_names
        )
    
    try:
        # 导入statsmodels
        import statsmodels.api as sm
        
        # 添加常数项
        if constant:
            X = sm.add_constant(X)
            if feature_names:
                feature_names = ["const"] + feature_names
            else:
                feature_names = [f"x{i-1}" for i in range(X.shape[1])]  # 从0开始编号
        else:
            if not feature_names:
                feature_names = [f"x{i}" for i in range(X.shape[1])]
        
        # 使用statsmodels进行OLS回归
        model = sm.OLS(y, X)
        results = model.fit()
        
        # 提取结果
        coefficients = results.params.tolist()
        std_errors = results.bse.tolist()
        t_values = results.tvalues.tolist()
        p_values = results.pvalues.tolist()
        
        # 计算置信区间
        conf_int = results.conf_int(alpha=1-confidence_level)
        conf_int_lower = conf_int[:, 0].tolist()
        conf_int_upper = conf_int[:, 1].tolist()
        
        # 其他统计量
        r_squared = float(results.rsquared)
        adj_r_squared = float(results.rsquared_adj)
        f_statistic = float(results.fvalue) if not np.isnan(results.fvalue) else np.inf
        f_p_value = float(results.f_pvalue) if not np.isnan(results.f_pvalue) else 0.0
        aic = float(results.aic)
        bic = float(results.bic)
        n_obs = int(results.nobs)
        
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
            n_obs=n_obs,
            feature_names=feature_names
        )
    except Exception as e:
        # 出现错误时使用原始实现
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
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            XtX_inv = np.linalg.pinv(X.T @ X)
            
        beta = XtX_inv @ X.T @ y
        
        # 计算预测值和残差
        y_pred = X @ beta
        residuals = y - y_pred
        
        # 计算各种统计量
        n = len(y)
        k = len(beta)
        df_resid = n - k
        
        # 残差平方和
        ssr = np.sum(residuals ** 2)
        
        # 总平方和
        sst = np.sum((y - np.mean(y)) ** 2)
        
        # 回归平方和
        ssr_reg = sst - ssr
        
        # 均方误差
        mse = ssr / df_resid if df_resid > 0 else np.inf
        
        # R方和调整R方
        r_squared = 1 - (ssr / sst) if sst != 0 else 1.0
        adj_r_squared = 1 - ((ssr / df_resid) / (sst / (n - 1))) if df_resid > 0 and sst != 0 else np.nan
        
        # 系数标准误
        var_beta = mse * XtX_inv
        std_errors = np.sqrt(np.diag(var_beta))
        
        # t统计量和p值
        t_values = beta / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df_resid))
        
        # 置信区间
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df_resid)
        conf_int_lower = beta - t_critical * std_errors
        conf_int_upper = beta + t_critical * std_errors
        
        # F统计量
        f_statistic = (ssr_reg / (k - 1)) / mse if mse != 0 and mse != np.inf else np.inf
        f_p_value = 1 - stats.f.cdf(f_statistic, k - 1, df_resid) if f_statistic != np.inf else 0.0
        
        # 信息准则 (简化计算)
        aic = n * np.log(ssr/n) + 2 * k if ssr > 0 else np.inf
        bic = n * np.log(ssr/n) + k * np.log(n) if ssr > 0 else np.inf
        
        return OLSResult(
            coefficients=beta.tolist(),
            std_errors=std_errors.tolist(),
            t_values=t_values.tolist(),
            p_values=p_values.tolist(),
            conf_int_lower=conf_int_lower.tolist(),
            conf_int_upper=conf_int_upper.tolist(),
            r_squared=float(r_squared),
            adj_r_squared=float(adj_r_squared),
            f_statistic=float(f_statistic),
            f_p_value=float(f_p_value),
            aic=float(aic),
            bic=float(bic),
            n_obs=n,
            feature_names=feature_names
        )