"""
Bootstrap重采样推断方法
基于 scipy.stats 实现多种Bootstrap方法
"""

from typing import List, Optional, Callable, Tuple, Dict
from pydantic import BaseModel, Field
import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None


class BootstrapResult(BaseModel):
    """Bootstrap推断结果"""
    statistic: float = Field(..., description="统计量估计值")
    bootstrap_mean: float = Field(..., description="Bootstrap均值")
    bootstrap_std: float = Field(..., description="Bootstrap标准误")
    confidence_interval: Tuple[float, float] = Field(..., description="置信区间")
    bias: float = Field(..., description="偏差估计")
    confidence_level: float = Field(..., description="置信水平")
    n_bootstrap: int = Field(..., description="Bootstrap重采样次数")
    method: str = Field(..., description="Bootstrap方法")
    bootstrap_distribution: List[float] = Field(..., description="Bootstrap统计量分布（前100个）")
    summary: str = Field(..., description="摘要信息")


def bootstrap_inference(
    data: List[float],
    statistic_func: Optional[str] = "mean",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    method: str = "percentile",
    random_state: Optional[int] = None
) -> BootstrapResult:
    """
    Bootstrap置信区间估计
    
    Args:
        data: 样本数据
        statistic_func: 统计量函数 - "mean"(均值), "median"(中位数), 
                        "std"(标准差), "var"(方差)
        n_bootstrap: Bootstrap重采样次数
        confidence_level: 置信水平
        method: 置信区间方法 - "percentile"(百分位法), "bca"(BCa法)
        random_state: 随机种子
        
    Returns:
        BootstrapResult: Bootstrap推断结果
        
    Raises:
        ImportError: scipy库未安装
        ValueError: 输入数据无效
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy库未安装。请运行: pip install scipy")
    
    # 输入验证
    if not data:
        raise ValueError("data不能为空")
    
    # 数据准备
    data_arr = np.array(data, dtype=np.float64)
    n = len(data_arr)
    
    # 设置随机种子
    if random_state is not None:
        np.random.seed(random_state)
    
    # 定义统计量函数
    if statistic_func == "mean":
        stat_fn = np.mean
    elif statistic_func == "median":
        stat_fn = np.median
    elif statistic_func == "std":
        stat_fn = lambda x: np.std(x, ddof=1)
    elif statistic_func == "var":
        stat_fn = lambda x: np.var(x, ddof=1)
    elif callable(statistic_func):
        stat_fn = statistic_func
    else:
        raise ValueError(f"不支持的统计量: {statistic_func}")
    
    # 计算原始统计量
    original_stat = float(stat_fn(data_arr))
    
    # 执行Bootstrap重采样
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # 有放回抽样
        bootstrap_sample = np.random.choice(data_arr, size=n, replace=True)
        bootstrap_stat = stat_fn(bootstrap_sample)
        bootstrap_stats.append(bootstrap_stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # 计算Bootstrap统计量
    bootstrap_mean = float(bootstrap_stats.mean())
    bootstrap_std = float(bootstrap_stats.std(ddof=1))
    bias = bootstrap_mean - original_stat
    
    # 计算置信区间
    alpha = 1 - confidence_level
    
    if method == "percentile":
        # 百分位法
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        ci_lower = float(np.percentile(bootstrap_stats, lower_percentile))
        ci_upper = float(np.percentile(bootstrap_stats, upper_percentile))
    elif method == "normal":
        # 正态近似法
        z_score = stats.norm.ppf(1 - alpha / 2)
        ci_lower = original_stat - z_score * bootstrap_std
        ci_upper = original_stat + z_score * bootstrap_std
    elif method == "basic":
        # 基本Bootstrap法
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        ci_lower = 2 * original_stat - float(np.percentile(bootstrap_stats, upper_percentile))
        ci_upper = 2 * original_stat - float(np.percentile(bootstrap_stats, lower_percentile))
    else:
        raise ValueError(f"不支持的置信区间方法: {method}")
    
    # 保存前100个Bootstrap统计量（用于展示）
    bootstrap_dist_sample = bootstrap_stats[:min(100, len(bootstrap_stats))].tolist()
    
    # 生成摘要
    summary = f"""Bootstrap推断:
- 样本量: {n}
- Bootstrap次数: {n_bootstrap}
- 统计量: {statistic_func}
- 置信区间方法: {method}

估计结果:
- 统计量估计: {original_stat:.4f}
- Bootstrap均值: {bootstrap_mean:.4f}
- Bootstrap标准误: {bootstrap_std:.4f}
- 偏差: {bias:.4f}

{int(confidence_level*100)}% 置信区间:
- 下界: {ci_lower:.4f}
- 上界: {ci_upper:.4f}
- 区间宽度: {ci_upper - ci_lower:.4f}
"""
    
    return BootstrapResult(
        statistic=original_stat,
        bootstrap_mean=bootstrap_mean,
        bootstrap_std=bootstrap_std,
        confidence_interval=(ci_lower, ci_upper),
        bias=bias,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        method=method,
        bootstrap_distribution=bootstrap_dist_sample,
        summary=summary
    )