"""
置换检验 (Permutation Test)
非参数假设检验方法
基于 scipy.stats 实现
"""

from typing import List, Optional
from pydantic import BaseModel, Field
import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None


class PermutationTestResult(BaseModel):
    """置换检验结果"""
    statistic: float = Field(..., description="观测统计量")
    p_value: float = Field(..., description="P值")
    null_distribution_mean: float = Field(..., description="零假设分布均值")
    null_distribution_std: float = Field(..., description="零假设分布标准差")
    n_permutations: int = Field(..., description="置换次数")
    alternative: str = Field(..., description="备择假设")
    test_type: str = Field(..., description="检验类型")
    n_sample_a: int = Field(..., description="样本A大小")
    n_sample_b: int = Field(..., description="样本B大小")
    permutation_distribution: List[float] = Field(..., description="置换分布（前100个）")
    summary: str = Field(..., description="摘要信息")


def permutation_test(
    sample_a: List[float],
    sample_b: List[float],
    test_type: str = "mean_difference",
    alternative: str = "two-sided",
    n_permutations: int = 10000,
    random_state: Optional[int] = None
) -> PermutationTestResult:
    """
    置换检验（两样本）
    
    Args:
        sample_a: 样本A
        sample_b: 样本B
        test_type: 检验类型 - "mean_difference"(均值差异), 
                   "median_difference"(中位数差异),
                   "variance_ratio"(方差比)
        alternative: 备择假设 - "two-sided", "less", "greater"
        n_permutations: 置换次数
        random_state: 随机种子
        
    Returns:
        PermutationTestResult: 置换检验结果
        
    Raises:
        ImportError: scipy库未安装
        ValueError: 输入数据无效
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy库未安装。请运行: pip install scipy")
    
    # 输入验证
    if not sample_a or not sample_b:
        raise ValueError("两个样本都不能为空")
    
    # 数据准备
    a = np.array(sample_a, dtype=np.float64)
    b = np.array(sample_b, dtype=np.float64)
    
    n_a = len(a)
    n_b = len(b)
    
    # 设置随机种子
    if random_state is not None:
        np.random.seed(random_state)
    
    # 合并数据
    combined = np.concatenate([a, b])
    n_total = len(combined)
    
    # 定义统计量函数
    if test_type == "mean_difference":
        def stat_func(x, y):
            return np.mean(x) - np.mean(y)
    elif test_type == "median_difference":
        def stat_func(x, y):
            return np.median(x) - np.median(y)
    elif test_type == "variance_ratio":
        def stat_func(x, y):
            return np.var(x, ddof=1) / np.var(y, ddof=1) if np.var(y, ddof=1) > 0 else 0
    else:
        raise ValueError(f"不支持的检验类型: {test_type}")
    
    # 计算观测统计量
    observed_stat = stat_func(a, b)
    
    # 执行置换检验
    perm_stats = []
    for _ in range(n_permutations):
        # 随机置换
        perm = np.random.permutation(combined)
        perm_a = perm[:n_a]
        perm_b = perm[n_a:]
        perm_stat = stat_func(perm_a, perm_b)
        perm_stats.append(perm_stat)
    
    perm_stats = np.array(perm_stats)
    
    # 计算p值
    if alternative == "two-sided":
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
    elif alternative == "greater":
        p_value = np.mean(perm_stats >= observed_stat)
    elif alternative == "less":
        p_value = np.mean(perm_stats <= observed_stat)
    else:
        raise ValueError(f"不支持的备择假设: {alternative}")
    
    # 零假设分布的统计特征
    null_mean = float(perm_stats.mean())
    null_std = float(perm_stats.std(ddof=1))
    
    # 保存前100个置换统计量
    perm_dist_sample = perm_stats[:min(100, len(perm_stats))].tolist()
    
    # 判断显著性
    if p_value < 0.01:
        significance = "高度显著"
    elif p_value < 0.05:
        significance = "显著"
    elif p_value < 0.10:
        significance = "边际显著"
    else:
        significance = "不显著"
    
    # 生成摘要
    test_names = {
        "mean_difference": "均值差异",
        "median_difference": "中位数差异",
        "variance_ratio": "方差比"
    }
    
    summary = f"""置换检验:
- 检验类型: {test_names.get(test_type, test_type)}
- 备择假设: {alternative}
- 置换次数: {n_permutations}

样本信息:
- 样本A: n={n_a}, 均值={a.mean():.4f}
- 样本B: n={n_b}, 均值={b.mean():.4f}

检验结果:
- 观测统计量: {observed_stat:.4f}
- P值: {p_value:.4f}
- 显著性: {significance}

零假设分布:
- 均值: {null_mean:.4f}
- 标准差: {null_std:.4f}
"""
    
    return PermutationTestResult(
        statistic=float(observed_stat),
        p_value=float(p_value),
        null_distribution_mean=null_mean,
        null_distribution_std=null_std,
        n_permutations=n_permutations,
        alternative=alternative,
        test_type=test_type,
        n_sample_a=n_a,
        n_sample_b=n_b,
        permutation_distribution=perm_dist_sample,
        summary=summary
    )