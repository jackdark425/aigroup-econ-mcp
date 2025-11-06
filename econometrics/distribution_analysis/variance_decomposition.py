"""
方差分解 (Variance Decomposition / ANOVA)
基于 scipy 和 statsmodels 实现
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import numpy as np

try:
    from scipy import stats
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
    sm = None


class VarianceDecompositionResult(BaseModel):
    """方差分解结果"""
    total_variance: float = Field(..., description="总方差")
    between_group_variance: float = Field(..., description="组间方差")
    within_group_variance: float = Field(..., description="组内方差")
    f_statistic: float = Field(..., description="F统计量")
    p_value: float = Field(..., description="P值")
    eta_squared: float = Field(..., description="Eta平方（效应量）")
    omega_squared: float = Field(..., description="Omega平方（偏效应量）")
    group_means: Dict[str, float] = Field(..., description="各组均值")
    group_variances: Dict[str, float] = Field(..., description="各组方差")
    group_sizes: Dict[str, int] = Field(..., description="各组样本量")
    n_groups: int = Field(..., description="组数")
    total_n: int = Field(..., description="总样本量")
    summary: str = Field(..., description="摘要信息")


def variance_decomposition(
    values: List[float],
    groups: List[str],
    group_names: Optional[List[str]] = None
) -> VarianceDecompositionResult:
    """
    方差分解 / 单因素ANOVA
    
    Args:
        values: 观测值列表
        groups: 组别标识列表
        group_names: 组名称映射
        
    Returns:
        VarianceDecompositionResult: 方差分解结果
        
    Raises:
        ImportError: scipy库未安装
        ValueError: 输入数据无效
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy和statsmodels库未安装。请运行: pip install scipy statsmodels")
    
    # 输入验证
    if not values or not groups:
        raise ValueError("values和groups不能为空")
    
    if len(values) != len(groups):
        raise ValueError(f"values长度({len(values)})与groups长度({len(groups)})不一致")
    
    # 数据准备
    y = np.array(values, dtype=np.float64)
    g = np.array(groups)
    
    # 获取唯一组别
    unique_groups = np.unique(g)
    n_groups = len(unique_groups)
    
    if n_groups < 2:
        raise ValueError("至少需要2个组进行方差分解")
    
    # 计算总体统计量
    grand_mean = y.mean()
    total_variance = y.var(ddof=1)
    total_n = len(y)
    
    # 计算各组统计量
    group_means = {}
    group_variances = {}
    group_sizes = {}
    
    # 按组分组数据
    groups_data = []
    for group_id in unique_groups:
        mask = g == group_id
        group_data = y[mask]
        groups_data.append(group_data)
        
        group_key = str(group_id)
        group_means[group_key] = float(group_data.mean())
        group_variances[group_key] = float(group_data.var(ddof=1))
        group_sizes[group_key] = int(len(group_data))
    
    # 执行单因素ANOVA
    f_stat, p_value = stats.f_oneway(*groups_data)
    
    # 计算组间方差和组内方差
    # SS_between = Σnᵢ(ȳᵢ - ȳ)²
    ss_between = sum(
        group_sizes[str(gid)] * (group_means[str(gid)] - grand_mean)**2 
        for gid in unique_groups
    )
    
    # SS_within = Σ(nᵢ - 1)sᵢ²
    ss_within = sum(
        (group_sizes[str(gid)] - 1) * group_variances[str(gid)] 
        for gid in unique_groups
    )
   
    # SS_total
    ss_total = (total_n - 1) * total_variance
    
    # 自由度
    df_between = n_groups - 1
    df_within = total_n - n_groups
    
    # 均方
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    # 组间方差和组内方差（作为总方差的比例）
    between_group_var = ss_between / (total_n - 1)
    within_group_var = ss_within / (total_n - 1)
    
    # 效应量
    # Eta平方 = SS_between / SS_total
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
    
    # Omega平方（偏效应量）
    omega_squared = (ss_between - df_between * ms_within) / (ss_total + ms_within)
    omega_squared = max(0.0, omega_squared)  # 确保非负
    
    # 生成摘要
    summary = f"""方差分解 (ANOVA) 分析:
- 总样本量: {total_n}
- 组数: {n_groups}
- 总方差: {total_variance:.4f}

方差分解:
- 组间方差: {between_group_var:.4f} ({eta_squared*100:.1f}%)
- 组内方差: {within_group_var:.4f} ({(1-eta_squared)*100:.1f}%)

F检验:
- F统计量: {f_stat:.4f}
- P值: {p_value:.4f}
- 结论: {'组间差异显著' if p_value < 0.05 else '组间差异不显著'}

效应量:
- Eta²: {eta_squared:.4f}
- Omega²: {omega_squared:.4f}

各组均值:
"""
    for gid in unique_groups:
        gkey = str(gid)
        summary += f"  {gkey}: {group_means[gkey]:.4f} (n={group_sizes[gkey]}, s²={group_variances[gkey]:.4f})\n"
    
    return VarianceDecompositionResult(
        total_variance=float(total_variance),
        between_group_variance=float(between_group_var),
        within_group_variance=float(within_group_var),
        f_statistic=float(f_stat),
        p_value=float(p_value),
        eta_squared=float(eta_squared),
        omega_squared=float(omega_squared),
        group_means=group_means,
        group_variances=group_variances,
        group_sizes=group_sizes,
        n_groups=n_groups,
        total_n=total_n,
        summary=summary
    )