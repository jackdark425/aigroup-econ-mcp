"""
模型诊断测试 (Diagnostic Tests) 模块实现

包括各种统计检验方法：
- 异方差检验（White、Breusch-Pagan）
- 自相关检验（Durbin-Watson、Ljung-Box）
- 正态性检验（Jarque-Bera）
- 多重共线性诊断（VIF）
- 内生性检验（Durbin-Wu-Hausman）
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor

from tools.decorators import with_file_support_decorator as econometric_tool, validate_input


class DiagnosticTestsResult(BaseModel):
    """模型诊断测试结果"""
    het_breuschpagan_stat: Optional[float] = Field(None, description="Breusch-Pagan异方差检验统计量")
    het_breuschpagan_pvalue: Optional[float] = Field(None, description="Breusch-Pagan异方差检验p值")
    het_white_stat: Optional[float] = Field(None, description="White异方差检验统计量")
    het_white_pvalue: Optional[float] = Field(None, description="White异方差检验p值")
    dw_statistic: Optional[float] = Field(None, description="Durbin-Watson自相关检验统计量")
    jb_statistic: Optional[float] = Field(None, description="Jarque-Bera正态性检验统计量")
    jb_pvalue: Optional[float] = Field(None, description="Jarque-Bera正态性检验p值")
    vif_values: Optional[List[float]] = Field(None, description="方差膨胀因子(VIF)")
    feature_names: Optional[List[str]] = Field(None, description="特征名称")


@econometric_tool("diagnostic_tests")
@validate_input(data_type="econometric")
def diagnostic_tests(
    y_data: List[float],
    x_data: List[List[float]], 
    feature_names: Optional[List[str]] = None,
    constant: bool = True
) -> DiagnosticTestsResult:
    """
    执行多种模型诊断测试
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        feature_names: 特征名称
        constant: 是否包含常数项
        
    Returns:
        DiagnosticTestsResult: 诊断测试结果
    """
    # 转换为numpy数组并确保浮点精度
    y = np.asarray(y_data, dtype=np.float64)
    X = np.asarray(x_data, dtype=np.float64)
    
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
        ols_model = sm.OLS(y, X)
        ols_results = ols_model.fit()
    except Exception as e:
        # 如果OLS失败，返回默认结果
        return DiagnosticTestsResult(
            het_breuschpagan_stat=None,
            het_breuschpagan_pvalue=None,
            het_white_stat=None,
            het_white_pvalue=None,
            dw_statistic=None,
            jb_statistic=None,
            jb_pvalue=None,
            vif_values=None,
            feature_names=feature_names[1:] if feature_names and len(feature_names) > 1 else None
        )
    
    # 计算预测值和残差
    y_pred = ols_results.fittedvalues
    residuals = ols_results.resid
    
    # Breusch-Pagan异方差检验
    try:
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X)
        bp_stat = float(bp_stat)
        bp_pvalue = float(bp_pvalue)
    except:
        bp_stat = None
        bp_pvalue = None
    
    # White异方差检验
    try:
        white_stat, white_pvalue, _, _ = het_white(residuals, X)
        white_stat = float(white_stat)
        white_pvalue = float(white_pvalue)
    except:
        white_stat = None
        white_pvalue = None
    
    # Durbin-Watson自相关检验
    try:
        dw_stat = float(sm.stats.durbin_watson(residuals))
    except:
        dw_stat = None
    
    # Jarque-Bera正态性检验
    try:
        jb_stat, jb_pvalue, _, _ = jarque_bera(residuals)
        jb_stat = float(jb_stat)
        jb_pvalue = float(jb_pvalue)
    except:
        jb_stat = None
        jb_pvalue = None
    
    # VIF计算（方差膨胀因子）
    try:
        vif_values = []
        # 只对自变量计算VIF（跳过常数项）
        for i in range(1 if constant else 0, X.shape[1]):
            vif = variance_inflation_factor(X, i)
            vif_values.append(float(vif))
    except:
        vif_values = None
    
    return DiagnosticTestsResult(
        het_breuschpagan_stat=bp_stat,
        het_breuschpagan_pvalue=bp_pvalue,
        het_white_stat=white_stat,
        het_white_pvalue=white_pvalue,
        dw_statistic=dw_stat,
        jb_statistic=jb_stat,
        jb_pvalue=jb_pvalue,
        vif_values=vif_values,
        feature_names=feature_names[1:] if constant and feature_names and len(feature_names) > 1 else feature_names
    )