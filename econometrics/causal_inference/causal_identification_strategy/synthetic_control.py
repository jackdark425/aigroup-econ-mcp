"""
合成控制法实现
"""

from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


class SyntheticControlResult(BaseModel):
    """合成控制法结果"""
    method: str = Field(default="Synthetic Control Method", description="使用的因果识别方法")
    treatment_effect: float = Field(..., description="处理效应估计值")
    synthetic_weights: List[float] = Field(..., description="合成控制权重")
    n_observations: int = Field(..., description="观测数量")
    donor_units: List[str] = Field(..., description="对照单元列表")
    pre_treatment_fit: Dict[str, float] = Field(..., description="处理前拟合度量")


def synthetic_control_method(
    outcome: List[float],
    treatment_period: int,
    treated_unit: str,
    donor_units: List[str],
    time_periods: List[str]
) -> SyntheticControlResult:
    """
    合成控制法
    
    合成控制法通过构造一个"合成"对照单元来评估处理效应，该对照单元是多个未处理单元的加权组合。
    
    Args:
        outcome: 结果变量（所有单元的时序数据）
        treatment_period: 处理开始的时间期
        treated_unit: 处理单元名称
        donor_units: 对照单元名称列表
        time_periods: 时间期列表
        
    Returns:
        SyntheticControlResult: 合成控制法结果
    """
    # 假设数据按单元排列，每个单元连续排列其时间序列
    n_units = len(donor_units) + 1  # 包括处理单元
    n_time = len(time_periods)
    
    if len(outcome) != n_units * n_time:
        raise ValueError("结果变量长度应等于单元数乘以时间期数")
    
    # 重塑数据为(单元, 时间)矩阵
    outcome_matrix = np.array(outcome).reshape(n_units, n_time)
    
    # 确定处理单元索引
    treated_idx = 0  # 假设处理单元是第一个
    
    # 提取处理前时期的数据
    pre_treatment_periods = treatment_period
    treated_pre = outcome_matrix[treated_idx, :pre_treatment_periods]
    donors_pre = outcome_matrix[1:, :pre_treatment_periods]  # 排除处理单元
    
    # 定义优化目标函数（最小化均方预测误差）
    def objective(weights):
        synthetic = donors_pre.T @ weights
        mse = np.mean((treated_pre - synthetic) ** 2)
        return mse
    
    # 约束条件：权重非负且和为1
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # 权重和为1
    ]
    bounds = [(0, 1) for _ in range(len(donor_units))]  # 权重在[0,1]之间
    
    # 初始权重
    initial_weights = np.ones(len(donor_units)) / len(donor_units)
    
    # 优化求解
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    
    # 计算合成对照单元的结果 - 修复矩阵乘法维度问题
    # outcome_matrix[1:] 形状: (n_units-1, n_time)
    # optimal_weights 形状: (n_units-1,)
    # 我们需要 (n_time, n_units-1) @ (n_units-1,) = (n_time,)
    synthetic_control = outcome_matrix[1:].T @ optimal_weights  # 所有时期
    
    # 计算处理效应（处理后时期）
    post_treatment_outcome = outcome_matrix[treated_idx, treatment_period:]
    post_treatment_synthetic = synthetic_control[treatment_period:]
    treatment_effect = np.mean(post_treatment_outcome - post_treatment_synthetic)
    
    # 计算处理前拟合度量
    pre_treatment_synthetic = synthetic_control[:treatment_period]
    pre_treatment_r2 = 1 - np.sum((treated_pre - pre_treatment_synthetic) ** 2) / \
                       np.sum((treated_pre - np.mean(treated_pre)) ** 2)
    
    pre_treatment_fit = {
        "R-squared": float(pre_treatment_r2),
        "RMSE": float(np.sqrt(np.mean((treated_pre - pre_treatment_synthetic) ** 2)))
    }
    
    return SyntheticControlResult(
        treatment_effect=float(treatment_effect),
        synthetic_weights=optimal_weights.tolist(),
        n_observations=len(outcome),
        donor_units=donor_units,
        pre_treatment_fit=pre_treatment_fit
    )