"""
统计分析工具
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any
from pydantic import BaseModel
import statsmodels.api as sm


class DescriptiveStats(BaseModel):
    """描述性统计结果"""
    mean: float
    median: float
    std: float
    min: float
    max: float
    skewness: float
    kurtosis: float
    count: int


class CorrelationResult(BaseModel):
    """相关性分析结果"""
    correlation_matrix: Dict[str, Dict[str, float]]
    method: str


def calculate_descriptive_stats(data: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
    """计算多变量描述性统计量"""
    results = {}
    for var_name, var_data in data.items():
        # 使用numpy计算统计量，避免pandas问题
        arr = np.array(var_data, dtype=float)
        
        stats_result = DescriptiveStats(
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            skewness=float(stats.skew(arr)),
            kurtosis=float(stats.kurtosis(arr)),
            count=len(arr)
        )
        # 转换为字典格式
        results[var_name] = stats_result.dict()
    return results


def calculate_correlation_matrix(
    data: Dict[str, List[float]],
    method: str = "pearson"
) -> CorrelationResult:
    """计算相关系数矩阵"""
    df = pd.DataFrame(data)
    corr_matrix = df.corr(method=method)

    return CorrelationResult(
        correlation_matrix=corr_matrix.to_dict(),
        method=method
    )


def perform_hypothesis_test(
    data1: List[float],
    data2: List[float] = None,
    test_type: str = "t_test",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """执行假设检验"""
    if test_type == "t_test":
        if data2 is None:
            # 单样本t检验
            t_stat, p_value = stats.ttest_1samp(data1, 0)
            test_name = "单样本t检验"
        else:
            # 双样本t检验
            t_stat, p_value = stats.ttest_ind(data1, data2)
            test_name = "双样本t检验"

        return {
            "test_type": test_name,
            "statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < alpha,
            "alpha": alpha
        }

    elif test_type == "f_test":
        # F检验（方差齐性检验）
        if data2 is None:
            raise ValueError("F检验需要两组数据")

        f_stat, p_value = stats.f_oneway(data1, data2)
        return {
            "test_type": "F检验",
            "statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < alpha,
            "alpha": alpha
        }

    elif test_type == "chi_square":
        # 卡方检验
        # 这里简化实现，实际需要频数数据
        chi2_stat, p_value = stats.chisquare(data1)
        return {
            "test_type": "卡方检验",
            "statistic": chi2_stat,
            "p_value": p_value,
            "significant": p_value < alpha,
            "alpha": alpha
        }

    elif test_type == "adf":
        # ADF单位根检验
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(data1)
        return {
            "test_type": "ADF单位根检验",
            "statistic": adf_result[0],
            "p_value": adf_result[1],
            "critical_values": adf_result[4],
            "significant": adf_result[1] < alpha,
            "alpha": alpha
        }
    
    else:
        raise ValueError(f"不支持的检验类型: {test_type}")


def normality_test(data: List[float]) -> Dict[str, Any]:
    """正态性检验"""
    # Shapiro-Wilk检验
    shapiro_stat, shapiro_p = stats.shapiro(data)

    # Kolmogorov-Smirnov检验
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))

    return {
        "shapiro_wilk": {
            "statistic": shapiro_stat,
            "p_value": shapiro_p,
            "normal": shapiro_p > 0.05
        },
        "kolmogorov_smirnov": {
            "statistic": ks_stat,
            "p_value": ks_p,
            "normal": ks_p > 0.05
        }
    }