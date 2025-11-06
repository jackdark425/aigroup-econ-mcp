"""
时间序列分解
将时间序列分解为趋势、季节和随机成分
基于 statsmodels 实现
"""

from typing import List, Optional
from pydantic import BaseModel, Field
import numpy as np

try:
    from statsmodels.tsa.seasonal import seasonal_decompose, STL
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    seasonal_decompose = None
    STL = None


class TimeSeriesDecompositionResult(BaseModel):
    """时间序列分解结果"""
    trend: List[float] = Field(..., description="趋势成分")
    seasonal: List[float] = Field(..., description="季节成分")
    residual: List[float] = Field(..., description="随机成分")
    observed: List[float] = Field(..., description="原始观测值")
    decomposition_type: str = Field(..., description="分解类型（加法/乘法）")
    method: str = Field(..., description="分解方法")
    period: int = Field(..., description="季节周期")
    trend_strength: float = Field(..., description="趋势强度")
    seasonal_strength: float = Field(..., description="季节强度")
    n_observations: int = Field(..., description="观测数量")
    summary: str = Field(..., description="摘要信息")


def time_series_decomposition(
    data: List[float],
    period: int = 12,
    model: str = "additive",
    method: str = "classical",
    extrapolate_trend: str = "freq"
) -> TimeSeriesDecompositionResult:
    """
    时间序列分解
    
    Args:
        data: 时间序列数据
        period: 季节周期（如12表示月度数据的年周期）
        model: 分解模型类型 - "additive"(加法模型) 或 "multiplicative"(乘法模型)
        method: 分解方法 - "classical"(经典分解) 或 "stl"(STL分解)
        extrapolate_trend: 趋势外推方法
        
    Returns:
        TimeSeriesDecompositionResult: 时间序列分解结果
        
    Raises:
        ImportError: statsmodels库未安装
        ValueError: 输入数据无效
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels库未安装。请运行: pip install statsmodels")
    
    # 输入验证
    if not data:
        raise ValueError("data不能为空")
    
    if len(data) < 2 * period:
        raise ValueError(f"数据点数({len(data)})应至少为季节周期({period})的2倍")
    
    # 数据准备
    y = np.array(data, dtype=np.float64)
    n = len(y)
    
    # 检查缺失值
    if np.isnan(y).any():
        raise ValueError("数据中包含缺失值")
    
    # 执行分解
    if method == "classical":
        # 使用经典分解方法
        decomposition = seasonal_decompose(
            y,
            model=model,
            period=period,
            extrapolate_trend=extrapolate_trend
        )
    elif method == "stl":
        # 使用STL分解（仅支持加法模型）
        if model != "additive":
            raise ValueError("STL分解仅支持加法模型")
        decomposition = STL(y, period=period).fit()
    else:
        raise ValueError(f"不支持的分解方法: {method}")
    
    # 提取成分
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    # 处理NaN值（在趋势计算中可能产生）
    # 使用线性插值填充
    if np.isnan(trend).any():
        mask = ~np.isnan(trend)
        indices = np.arange(len(trend))
        trend = np.interp(indices, indices[mask], trend[mask])
    
    # 计算趋势和季节强度
    # 趋势强度 = 1 - Var(残差) / Var(去季节化序列)
    deseasonalized = y - seasonal
    var_resid = np.var(residual[~np.isnan(residual)])
    var_deseas = np.var(deseasonalized[~np.isnan(deseasonalized)])
    trend_strength = 1 - (var_resid / var_deseas) if var_deseas > 0 else 0.0
    trend_strength = max(0.0, min(1.0, trend_strength))
    
    # 季节强度 = 1 - Var(残差) / Var(去趋势化序列)
    detrended = y - trend
    var_detrend = np.var(detrended[~np.isnan(detrended)])
    seasonal_strength = 1 - (var_resid / var_detrend) if var_detrend > 0 else 0.0
    seasonal_strength = max(0.0, min(1.0, seasonal_strength))
    
    # 生成摘要
    summary = f"""时间序列分解:
- 观测数量: {n}
- 季节周期: {period}
- 分解模型: {model}
- 分解方法: {method}

成分方差:
- 趋势方差: {np.var(trend[~np.isnan(trend)]):.4f}
- 季节方差: {np.var(seasonal[~np.isnan(seasonal)]):.4f}
- 残差方差: {var_resid:.4f}

强度指标:
- 趋势强度: {trend_strength:.4f} ({'强' if trend_strength > 0.6 else '中' if trend_strength > 0.3 else '弱'})
- 季节强度: {seasonal_strength:.4f} ({'强' if seasonal_strength > 0.6 else '中' if seasonal_strength > 0.3 else '弱'})

解释:
- {model}模型: y = {'趋势 + 季节 + 随机' if model == 'additive' else '趋势 × 季节 × 随机'}
"""
    
    return TimeSeriesDecompositionResult(
        trend=trend.tolist(),
        seasonal=seasonal.tolist(),
        residual=residual.tolist(),
        observed=y.tolist(),
        decomposition_type=model,
        method=method,
        period=period,
        trend_strength=float(trend_strength),
        seasonal_strength=float(seasonal_strength),
        n_observations=n,
        summary=summary
    )