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
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX)
        beta = XtX_inv @ X.T @ y
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，添加正则化项
        k_reg = X.shape[1]
        XtX_reg = XtX + np.eye(k_reg) * 1e-10 * np.trace(XtX) / k_reg
        XtX_inv = np.linalg.inv(XtX_reg)
        beta = XtX_inv @ X.T @ y
    
    # 计算预测值和残差
    y_pred = X @ beta
    residuals = y - y_pred
    
    # 计算各种统计量
    n = len(y)
    k = len(beta)
    
    # 避免空数据或极小数据集
    if n < 2 or k < 1:
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
    
    # Breusch-Pagan异方差检验
    # 对残差平方进行辅助回归
    residuals_squared = residuals ** 2
    mean_res_squared = np.mean(residuals_squared)
    
    # 辅助回归
    aux_X = X
    try:
        aux_beta = np.linalg.inv(aux_X.T @ aux_X) @ aux_X.T @ residuals_squared
        aux_pred = aux_X @ aux_beta
        ssr_aux = np.sum((residuals_squared - aux_pred) ** 2)
        rss_aux = np.sum((aux_pred - mean_res_squared) ** 2)
        
        # BP统计量
        if (rss_aux + ssr_aux) > 1e-10:  # 避免除以零或极小数
            bp_stat = n * rss_aux / (rss_aux + ssr_aux)
            bp_pvalue = 1 - stats.chi2.cdf(max(bp_stat, 0), k)  # 避免负统计量
        else:
            bp_stat = 0
            bp_pvalue = 1
    except np.linalg.LinAlgError:
        # 矩阵奇异时无法计算
        bp_stat = None
        bp_pvalue = None
    
    # White异方差检验 (简化版 - 不包含交叉项)
    # 辅助回归：残差平方 ~ X + X^2
    try:
        # 避免对常数项平方
        X_squares = X[:, 1:] ** 2 if X.shape[1] > 1 else np.empty((n, 0))
        X_with_squares = np.column_stack([X, X_squares]) if X_squares.size > 0 else X
        aux_X_white = X_with_squares
        
        if aux_X_white.shape[1] > 0:  # 确保有变量可以回归
            aux_beta_white = np.linalg.inv(aux_X_white.T @ aux_X_white) @ aux_X_white.T @ residuals_squared
            aux_pred_white = aux_X_white @ aux_beta_white
            ssr_white = np.sum((residuals_squared - aux_pred_white) ** 2)
            rss_white = np.sum((aux_pred_white - mean_res_squared) ** 2)
            
            # White统计量
            if (rss_white + ssr_white) > 1e-10:  # 避免除以零或极小数
                white_stat = n * rss_white / (rss_white + ssr_white)
                k_white = aux_X_white.shape[1]
                white_pvalue = 1 - stats.chi2.cdf(max(white_stat, 0), k_white)  # 避免负统计量
            else:
                white_stat = 0
                white_pvalue = 1
        else:
            white_stat = 0
            white_pvalue = 1
    except (np.linalg.LinAlgError, ValueError):
        # 矩阵奇异或数值问题时无法计算
        white_stat = None
        white_pvalue = None
    
    # Durbin-Watson自相关检验
    if len(residuals) > 1:
        diff_residuals = np.diff(residuals)
        sum_diff_squared = np.sum(diff_residuals ** 2)
        sum_res_squared = np.sum(residuals ** 2)
        
        if sum_res_squared > 1e-10:
            dw_stat = sum_diff_squared / sum_res_squared
        else:
            dw_stat = 2.0  # 无自相关的中性值
    else:
        dw_stat = None
    
    # Jarque-Bera正态性检验
    if len(residuals) > 2:
        # 偏度和峰度
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        
        if residual_std > 1e-10:  # 避免除以零或极小标准差
            standardized_residuals = (residuals - residual_mean) / residual_std
            
            # 使用更稳健的方法计算偏度和峰度
            skewness = np.mean(standardized_residuals ** 3)
            kurtosis = np.mean(standardized_residuals ** 4)
            
            # JB统计量
            jb_stat = n * (skewness ** 2 / 6 + (kurtosis - 3) ** 2 / 24)
            jb_stat = max(jb_stat, 0)  # 避免负统计量
            jb_pvalue = 1 - stats.chi2.cdf(jb_stat, 2)
        else:
            jb_stat = 0
            jb_pvalue = 1
    else:
        jb_stat = None
        jb_pvalue = None
    
    # VIF计算（方差膨胀因子）
    vif_values = []
    if X.shape[1] > 1:  # 只有当有多个变量时才计算VIF
        for i in range(1, X.shape[1]):  # 跳过常数项
            # 对第i个变量对其他所有变量进行回归
            X_others = np.delete(X, i, axis=1)
            y_current = X[:, i]
            
            try:
                # 检查X_others是否为空
                if X_others.size == 0:
                    vif_values.append(1.0)  # 如果没有其他变量，VIF为1
                    continue
                    
                beta_aux = np.linalg.inv(X_others.T @ X_others) @ X_others.T @ y_current
                y_pred_aux = X_others @ beta_aux
                ssr_aux_vif = np.sum((y_current - y_pred_aux) ** 2)
                sst_aux_vif = np.sum((y_current - np.mean(y_current)) ** 2)
                # 避免除以零
                r2_aux = 1 - ssr_aux_vif / sst_aux_vif if sst_aux_vif != 0 and ssr_aux_vif <= sst_aux_vif else 0
                # 避免除以零或负数
                vif = 1 / (1 - r2_aux) if r2_aux < 0.999999 else float('inf')  # 防止1-1的情况
                vif_values.append(vif if np.isfinite(vif) else 1e6)  # 限制最大值
            except np.linalg.LinAlgError:
                # 矩阵奇异时无法计算，设置为一个大的值
                vif_values.append(1e6)
    else:
        vif_values = None
    
    return DiagnosticTestsResult(
        het_breuschpagan_stat=float(bp_stat) if bp_stat is not None else None,
        het_breuschpagan_pvalue=float(bp_pvalue) if bp_pvalue is not None else None,
        het_white_stat=float(white_stat) if white_stat is not None else None,
        het_white_pvalue=float(white_pvalue) if white_pvalue is not None else None,
        dw_statistic=float(dw_stat) if dw_stat is not None else None,
        jb_statistic=float(jb_stat) if jb_stat is not None else None,
        jb_pvalue=float(jb_pvalue) if jb_pvalue is not None else None,
        vif_values=vif_values,
        feature_names=feature_names[1:] if feature_names and len(feature_names) > 1 else None  # 不包括常数项
    )