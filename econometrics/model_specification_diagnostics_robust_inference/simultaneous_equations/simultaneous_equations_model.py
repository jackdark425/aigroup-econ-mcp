"""
联立方程模型 (Simultaneous Equations Models) 模块实现

处理双向因果关系的模型方法
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from scipy import stats

from tools.decorators import with_file_support_decorator as econometric_tool, validate_input


class SimultaneousEquationsResult(BaseModel):
    """联立方程模型结果"""
    coefficients: List[List[float]] = Field(..., description="各方程的回归系数")
    std_errors: List[List[float]] = Field(..., description="各方程的系数标准误")
    t_values: List[List[float]] = Field(..., description="各方程的t统计量")
    p_values: List[List[float]] = Field(..., description="各方程的p值")
    r_squared: List[float] = Field(..., description="各方程的R方")
    adj_r_squared: List[float] = Field(..., description="各方程的调整R方")
    n_obs: int = Field(..., description="观测数量")
    equation_names: List[str] = Field(..., description="方程名称")
    endogenous_vars: List[str] = Field(..., description="内生变量名称")
    exogenous_vars: List[str] = Field(..., description="外生变量名称")


@econometric_tool("two_stage_least_squares")
@validate_input(data_type="econometric")
def two_stage_least_squares(
    y_data: List[List[float]],  # 每个方程的因变量
    x_data: List[List[List[float]]],  # 每个方程的自变量
    instruments: List[List[float]],  # 工具变量
    equation_names: Optional[List[str]] = None,
    endogenous_vars: Optional[List[str]] = None,
    exogenous_vars: Optional[List[str]] = None,
    constant: bool = True
) -> SimultaneousEquationsResult:
    """
    两阶段最小二乘法（2SLS）用于联立方程模型
    
    Args:
        y_data: 每个方程的因变量数据列表
        x_data: 每个方程的自变量数据列表
        instruments: 工具变量数据
        equation_names: 方程名称
        endogenous_vars: 内生变量名称
        exogenous_vars: 外生变量名称
        constant: 是否包含常数项
        
    Returns:
        SimultaneousEquationsResult: 联立方程模型结果
    """
    # 转换为numpy数组
    Y = [np.asarray(y, dtype=np.float64) for y in y_data]  # 每个方程的因变量
    X = [np.asarray(x, dtype=np.float64) for x in x_data]  # 每个方程的自变量
    Z = np.asarray(instruments, dtype=np.float64)          # 工具变量
    
    # 检查数据是否为空
    if not Y or not X or Z.size == 0:
        raise ValueError("输入数据不能为空")
    
    n_equations = len(Y)
    if n_equations == 0:
        raise ValueError("至少需要一个方程")
    
    n_obs = len(Y[0])
    if n_obs == 0:
        raise ValueError("观测数据不能为空")
    
    # 检查维度一致性
    for i in range(n_equations):
        if len(Y[i]) != n_obs:
            raise ValueError(f"方程{i+1}的观测数量({len(Y[i])})必须与其他方程相同({n_obs})")
        if len(X[i]) != n_obs:
            raise ValueError(f"方程{i+1}自变量数据的观测数量必须与因变量相同")
        if len(X[i]) > 0 and len(X[i][0]) == 0:
            raise ValueError(f"方程{i+1}必须包含至少一个自变量")
    
    if Z.shape[0] != n_obs:
        raise ValueError(f"工具变量的观测数量({Z.shape[0]})必须与其他变量相同({n_obs})")
    
    # 检查工具变量数量
    n_instruments = Z.shape[1]
    if n_instruments == 0:
        raise ValueError("工具变量不能为空")
    
    # 添加常数项
    if constant:
        for i in range(n_equations):
            X[i] = np.column_stack([np.ones(n_obs), X[i]])
        Z = np.column_stack([np.ones(n_obs), Z])
        if exogenous_vars:
            exogenous_vars = ["const"] + exogenous_vars
        else:
            exogenous_vars = [f"exog_{j}" for j in range(Z.shape[1])]
    
    # 检查识别条件（工具变量数量应至少等于内生变量数量）
    n_endog_vars = max([x.shape[1] for x in X]) if X else 0
    if n_instruments < n_endog_vars:
        # 发出警告但不终止
        pass  # 在实际应用中可能会发出警告
    
    # 第二阶段：使用2SLS进行回归
    coefficients = []
    std_errors = []
    t_values = []
    p_values = []
    r_squared_vals = []
    adj_r_squared_vals = []
    
    for i in range(n_equations):
        # 获取当前方程的因变量和自变量
        Y_i = Y[i]
        X_i = X[i]
        
        # 检查当前方程数据
        if len(Y_i) == 0 or X_i.shape[0] == 0:
            # 添加默认值
            k_vars = X_i.shape[1] if X_i.shape[0] > 0 else 1
            coefficients.append([0.0] * k_vars)
            std_errors.append([1.0] * k_vars)
            t_values.append([0.0] * k_vars)
            p_values.append([1.0] * k_vars)
            r_squared_vals.append(0.0)
            adj_r_squared_vals.append(0.0)
            continue
        
        n_i, k_i = X_i.shape
        
        # 第一阶段：对所有自变量进行工具变量回归 X_i ~ Z
        try:
            ZtZ = Z.T @ Z
            # 添加正则化防止矩阵奇异
            reg_strength = 1e-10 * np.trace(ZtZ) / ZtZ.shape[0] if ZtZ.shape[0] > 0 and np.trace(ZtZ) > 0 else 1e-10
            ZtZ_reg = ZtZ + reg_strength * np.eye(ZtZ.shape[0])
            ZtZ_inv = np.linalg.inv(ZtZ_reg)
        except np.linalg.LinAlgError:
            # 如果矩阵仍然奇异，使用伪逆
            ZtZ = Z.T @ Z
            ZtZ_inv = np.linalg.pinv(ZtZ + 1e-10 * np.eye(ZtZ.shape[0]))
        
        # 预测X的值
        X_hat = Z @ ZtZ_inv @ Z.T @ X_i
        
        # 第二阶段：使用预测的X值回归 Y_i ~ X_hat
        try:
            XtX = X_hat.T @ X_hat
            # 添加正则化防止矩阵奇异
            reg_strength = 1e-10 * np.trace(XtX) / XtX.shape[0] if XtX.shape[0] > 0 and np.trace(XtX) > 0 else 1e-10
            XtX_reg = XtX + reg_strength * np.eye(XtX.shape[0])
            XtX_inv = np.linalg.inv(XtX_reg)
        except np.linalg.LinAlgError:
            # 如果矩阵仍然奇异，使用伪逆
            XtX = X_hat.T @ X_hat
            XtX_inv = np.linalg.pinv(XtX + 1e-10 * np.eye(XtX.shape[0]))
        
        # 估计系数
        beta = XtX_inv @ X_hat.T @ Y_i
        
        # 计算预测值和残差
        Y_pred = X_i @ beta
        residuals = Y_i - Y_pred
        
        # 计算统计量
        df_resid = max(n_i - k_i, 1)
        
        # 残差平方和
        ssr = np.sum(residuals ** 2)
        
        # 总平方和
        y_mean_i = np.mean(Y_i)
        sst = np.sum((Y_i - y_mean_i) ** 2)
        
        # 均方误差（用于标准误计算）
        mse = ssr / df_resid if df_resid > 0 else ssr
        
        # R方和调整R方
        r_squared = 1 - (ssr / sst) if sst > 1e-10 else 0
        adj_r_squared = 1 - ((ssr / df_resid) / (sst / max(n_i - 1, 1))) if sst > 1e-10 and n_i > k_i else 0
        
        # 系数标准误 (使用工具变量估计的协方差矩阵)
        var_beta = mse * XtX_inv
        # 避免负方差
        diag_var_beta = np.diag(var_beta)
        diag_var_beta = np.maximum(diag_var_beta, 0)
        std_err = np.sqrt(diag_var_beta)
        # 避免标准误为零
        std_err = np.maximum(std_err, 1e-12)
        
        # t统计量和p值
        t_val = np.divide(beta, std_err, out=np.zeros_like(beta), where=std_err!=0)
        p_val = 2 * (1 - stats.t.cdf(np.abs(t_val), df_resid))
        
        coefficients.append(beta.tolist())
        std_errors.append(std_err.tolist())
        t_values.append(t_val.tolist())
        p_values.append(p_val.tolist())
        r_squared_vals.append(float(r_squared))
        adj_r_squared_vals.append(float(adj_r_squared))
    
    if not equation_names:
        equation_names = [f"equation_{i+1}" for i in range(n_equations)]
    
    if not endogenous_vars:
        endogenous_vars = [f"endog_{i}" for i in range(n_equations)]
    
    if not exogenous_vars:
        # 计算工具变量数量（减去常数项）
        n_inst = Z.shape[1] - 1 if constant else Z.shape[1]
        exogenous_vars = [f"exog_{i}" for i in range(n_inst)]
    
    return SimultaneousEquationsResult(
        coefficients=coefficients,
        std_errors=std_errors,
        t_values=t_values,
        p_values=p_values,
        r_squared=r_squared_vals,
        adj_r_squared=adj_r_squared_vals,
        n_obs=n_obs,
        equation_names=equation_names,
        endogenous_vars=endogenous_vars,
        exogenous_vars=exogenous_vars
    )