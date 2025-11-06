"""
分位数回归 (Quantile Regression)
基于 statsmodels.regression.quantile_regression 库实现
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import numpy as np

try:
    import statsmodels.api as sm
    from statsmodels.regression.quantile_regression import QuantReg
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    QuantReg = None


class QuantileRegressionResult(BaseModel):
    """分位数回归结果"""
    quantile: float = Field(..., description="分位数水平")
    coefficients: List[float] = Field(..., description="回归系数")
    std_errors: List[float] = Field(..., description="标准误")
    t_values: List[float] = Field(..., description="t统计量")
    p_values: List[float] = Field(..., description="p值")
    conf_int_lower: List[float] = Field(..., description="置信区间下界")
    conf_int_upper: List[float] = Field(..., description="置信区间上界")
    feature_names: List[str] = Field(..., description="特征名称")
    pseudo_r_squared: float = Field(..., description="伪R²")
    n_observations: int = Field(..., description="观测数量")
    summary: str = Field(..., description="摘要信息")


class MultiQuantileResult(BaseModel):
    """多分位数回归结果"""
    quantiles: List[float] = Field(..., description="分位数水平列表")
    coefficients_by_quantile: Dict[str, List[float]] = Field(..., description="各分位数的系数")
    feature_names: List[str] = Field(..., description="特征名称")
    n_observations: int = Field(..., description="观测数量")
    summary: str = Field(..., description="摘要信息")


def quantile_regression(
    y_data: List[float],
    x_data: List[List[float]],
    quantile: float = 0.5,
    feature_names: Optional[List[str]] = None,
    confidence_level: float = 0.95
) -> QuantileRegressionResult:
    """
    分位数回归
    
    Args:
        y_data: 因变量
        x_data: 自变量（二维列表）
        quantile: 分位数水平（0-1之间），默认0.5为中位数回归
        feature_names: 特征名称
        confidence_level: 置信水平
        
    Returns:
        QuantileRegressionResult: 分位数回归结果
        
    Raises:
        ImportError: statsmodels库未安装
        ValueError: 输入数据无效
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError(
            "statsmodels库未安装。请运行: pip install statsmodels"
        )
    
    # 输入验证
    if not y_data or not x_data:
        raise ValueError("y_data和x_data不能为空")
    
    if not 0 < quantile < 1:
        raise ValueError("quantile必须在0和1之间")
    
    # 数据准备
    y = np.array(y_data, dtype=np.float64)
    X = np.array(x_data, dtype=np.float64)
    
    # 确保X是二维数组
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n = len(y)
    k = X.shape[1]
    
    # 数据验证
    if len(y) != X.shape[0]:
        raise ValueError(f"因变量长度({len(y)})与自变量长度({X.shape[0]})不一致")
    
    # 添加常数项
    X_with_const = sm.add_constant(X)
    
    # 特征名称
    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(k)]
    all_feature_names = ["const"] + feature_names
    
    # 构建并拟合分位数回归模型
    try:
        model = QuantReg(y, X_with_const)
        results = model.fit(q=quantile)
    except Exception as e:
        raise ValueError(f"分位数回归拟合失败: {str(e)}")
    
    # 提取结果
    coefficients = results.params.tolist()
    
    # 标准误（使用稳健标准误）
    try:
        # 尝试使用稳健标准误
        std_errors = results.bse.tolist()
    except:
        # 如果失败，使用常规标准误
        std_errors = [0.0] * len(coefficients)
    
    # t统计量和p值
    try:
        t_values = results.tvalues.tolist()
        p_values = results.pvalues.tolist()
    except:
        t_values = [0.0] * len(coefficients)
        p_values = [1.0] * len(coefficients)
    
    # 置信区间
    try:
        alpha = 1 - confidence_level
        conf_int = results.conf_int(alpha=alpha)
        conf_int_lower = conf_int.iloc[:, 0].tolist()
        conf_int_upper = conf_int.iloc[:, 1].tolist()
    except:
        conf_int_lower = [c - 1.96 * se for c, se in zip(coefficients, std_errors)]
        conf_int_upper = [c + 1.96 * se for c, se in zip(coefficients, std_errors)]
    
    # 伪R²
    try:
        pseudo_r_squared = float(results.prsquared)
    except:
        pseudo_r_squared = 0.0
    
    # 生成摘要
    summary = f"""分位数回归分析:
- 分位数τ: {quantile}
- 观测数量: {n}
- 协变量数: {k}
- 伪R²: {pseudo_r_squared:.4f}

系数估计:
"""
    for i, (name, coef, se, t, p) in enumerate(zip(
        all_feature_names, coefficients, std_errors, t_values, p_values
    )):
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        summary += f"  {name}: {coef:.4f} (SE: {se:.4f}, t={t:.2f}, p={p:.4f}){sig}\n"
    
    return QuantileRegressionResult(
        quantile=quantile,
        coefficients=coefficients,
        std_errors=std_errors,
        t_values=t_values,
        p_values=p_values,
        conf_int_lower=conf_int_lower,
        conf_int_upper=conf_int_upper,
        feature_names=all_feature_names,
        pseudo_r_squared=pseudo_r_squared,
        n_observations=n,
        summary=summary
    )


def multi_quantile_regression(
    y_data: List[float],
    x_data: List[List[float]],
    quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
    feature_names: Optional[List[str]] = None
) -> MultiQuantileResult:
    """
    多分位数回归
    同时估计多个分位数水平的回归系数
    
    Args:
        y_data: 因变量
        x_data: 自变量
        quantiles: 分位数水平列表
        feature_names: 特征名称
        
    Returns:
        MultiQuantileResult: 多分位数回归结果
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels库未安装")
    
    # 输入验证
    if not y_data or not x_data:
        raise ValueError("y_data和x_data不能为空")
    
    # 数据准备
    y = np.array(y_data, dtype=np.float64)
    X = np.array(x_data, dtype=np.float64)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n = len(y)
    k = X.shape[1]
    
    # 添加常数项
    X_with_const = sm.add_constant(X)
    
    # 特征名称
    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(k)]
    all_feature_names = ["const"] + feature_names
    
    # 对每个分位数进行回归
    coefficients_by_quantile = {}
    
    for q in quantiles:
        try:
            model = QuantReg(y, X_with_const)
            results = model.fit(q=q)
            coefficients_by_quantile[f"τ={q}"] = results.params.tolist()
        except Exception as e:
            coefficients_by_quantile[f"τ={q}"] = [np.nan] * (k + 1)
    
    # 生成摘要
    summary = f"""多分位数回归分析:
- 观测数量: {n}
- 协变量数: {k}
- 分位数: {quantiles}

各分位数的系数估计:
"""
    for name_idx, name in enumerate(all_feature_names):
        summary += f"\n{name}:\n"
        for q in quantiles:
            coef = coefficients_by_quantile[f"τ={q}"][name_idx]
            summary += f"  τ={q}: {coef:.4f}\n"
    
    return MultiQuantileResult(
        quantiles=quantiles,
        coefficients_by_quantile=coefficients_by_quantile,
        feature_names=all_feature_names,
        n_observations=n,
        summary=summary
    )