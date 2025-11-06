"""
Oaxaca-Blinder分解
用于分解两组之间的平均差异（如工资差距）
基于 statsmodels 和自定义实现
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    sm = None


class OaxacaResult(BaseModel):
    """Oaxaca-Blinder分解结果"""
    total_difference: float = Field(..., description="总差异")
    explained_part: float = Field(..., description="可解释部分（禀赋效应）")
    unexplained_part: float = Field(..., description="不可解释部分（系数效应）")
    explained_pct: float = Field(..., description="可解释部分百分比")
    unexplained_pct: float = Field(..., description="不可解释部分百分比")
    group_a_mean: float = Field(..., description="A组平均值")
    group_b_mean: float = Field(..., description="B组平均值")
    group_a_coefficients: List[float] = Field(..., description="A组回归系数")
    group_b_coefficients: List[float] = Field(..., description="B组回归系数")
    detailed_explained: List[float] = Field(..., description="各变量的可解释部分")
    detailed_unexplained: List[float] = Field(..., description="各变量的不可解释部分")
    feature_names: List[str] = Field(..., description="特征名称")
    n_obs_a: int = Field(..., description="A组观测数")
    n_obs_b: int = Field(..., description="B组观测数")
    summary: str = Field(..., description="摘要信息")


def oaxaca_blinder_decomposition(
    y_a: List[float],
    x_a: List[List[float]],
    y_b: List[float],
    x_b: List[List[float]],
    feature_names: Optional[List[str]] = None,
    weight_matrix: str = "pooled"
) -> OaxacaResult:
    """
    Oaxaca-Blinder分解
    分解两组之间的平均差异为可解释部分和不可解释部分
    
    Args:
        y_a: A组因变量（如男性工资）
        x_a: A组自变量
        y_b: B组因变量（如女性工资）
        x_b: B组自变量
        feature_names: 特征名称
        weight_matrix: 权重矩阵类型 - "pooled"(pooled权重), "group_a", "group_b"
        
    Returns:
        OaxacaResult: Oaxaca-Blinder分解结果
        
    Raises:
        ImportError: statsmodels库未安装
        ValueError: 输入数据无效
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels库未安装。请运行: pip install statsmodels")
    
    # 输入验证
    if not y_a or not x_a or not y_b or not x_b:
        raise ValueError("所有输入数据不能为空")
    
    # 数据准备
    y_a_arr = np.array(y_a, dtype=np.float64)
    X_a_arr = np.array(x_a, dtype=np.float64)
    y_b_arr = np.array(y_b, dtype=np.float64)
    X_b_arr = np.array(x_b, dtype=np.float64)
    
    # 确保X是二维数组
    if X_a_arr.ndim == 1:
        X_a_arr = X_a_arr.reshape(-1, 1)
    if X_b_arr.ndim == 1:
        X_b_arr = X_b_arr.reshape(-1, 1)
    
    n_a = len(y_a_arr)
    n_b = len(y_b_arr)
    k = X_a_arr.shape[1]
    
    # 数据验证
    if X_a_arr.shape[1] != X_b_arr.shape[1]:
        raise ValueError("两组的自变量数量必须相同")
    
    # 添加常数项
    X_a_const = sm.add_constant(X_a_arr)
    X_b_const = sm.add_constant(X_b_arr)
    
    # 特征名称
    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(k)]
    all_feature_names = ["const"] + feature_names
    
    # 对两组分别进行OLS回归
    model_a = sm.OLS(y_a_arr, X_a_const).fit()
    model_b = sm.OLS(y_b_arr, X_b_const).fit()
    
    # 提取系数
    beta_a = model_a.params
    beta_b = model_b.params
    
    # 计算两组的平均特征值
    X_a_mean = X_a_const.mean(axis=0)
    X_b_mean = X_b_const.mean(axis=0)
    
    # 计算两组的平均因变量
    y_a_mean = float(y_a_arr.mean())
    y_b_mean = float(y_b_arr.mean())
    
    # 总差异
    total_diff = y_a_mean - y_b_mean
    
    # 根据权重矩阵选择参考系数
    if weight_matrix == "pooled":
        # 使用pooled回归的系数作为参考
        y_pooled = np.concatenate([y_a_arr, y_b_arr])
        X_pooled = np.vstack([X_a_const, X_b_const])
        model_pooled = sm.OLS(y_pooled, X_pooled).fit()
        beta_ref = model_pooled.params
    elif weight_matrix == "group_a":
        beta_ref = beta_a
    elif weight_matrix == "group_b":
        beta_ref = beta_b
    else:
        raise ValueError(f"不支持的权重矩阵类型: {weight_matrix}")
    
    # Oaxaca分解
    # 可解释部分（禀赋效应）: (X̄ₐ - X̄ᵦ)' β*
    explained = (X_a_mean - X_b_mean) @ beta_ref
    
    # 不可解释部分（系数效应）: X̄ₐ'(βₐ - β*) + X̄ᵦ'(β* - βᵦ)
    unexplained = X_a_mean @ (beta_a - beta_ref) + X_b_mean @ (beta_ref - beta_b)
    
    # 详细分解（每个变量的贡献）
    detailed_explained = ((X_a_mean - X_b_mean) * beta_ref).tolist()
    detailed_unexplained = (
        X_a_mean * (beta_a - beta_ref) + X_b_mean * (beta_ref - beta_b)
    ).tolist()
    
    # 百分比
    explained_pct = (explained / total_diff * 100) if total_diff != 0 else 0.0
    unexplained_pct = (unexplained / total_diff * 100) if total_diff != 0 else 0.0
    
    # 生成摘要
    summary = f"""Oaxaca-Blinder分解:
- 总差异: {total_diff:.4f}
  - A组平均: {y_a_mean:.4f} (n={n_a})
  - B组平均: {y_b_mean:.4f} (n={n_b})

分解结果:
- 可解释部分（禀赋效应）: {explained:.4f} ({explained_pct:.1f}%)
- 不可解释部分（系数效应）: {unexplained:.4f} ({unexplained_pct:.1f}%)

各变量贡献:
"""
    for i, name in enumerate(all_feature_names):
        summary += f"  {name}:\n"
        summary += f"    - 禀赋效应: {detailed_explained[i]:.4f}\n"
        summary += f"    - 系数效应: {detailed_unexplained[i]:.4f}\n"
    
    return OaxacaResult(
        total_difference=float(total_diff),
        explained_part=float(explained),
        unexplained_part=float(unexplained),
        explained_pct=float(explained_pct),
        unexplained_pct=float(unexplained_pct),
        group_a_mean=y_a_mean,
        group_b_mean=y_b_mean,
        group_a_coefficients=beta_a.tolist(),
        group_b_coefficients=beta_b.tolist(),
        detailed_explained=detailed_explained,
        detailed_unexplained=detailed_unexplained,
        feature_names=all_feature_names,
        n_obs_a=n_a,
        n_obs_b=n_b,
        summary=summary
    )