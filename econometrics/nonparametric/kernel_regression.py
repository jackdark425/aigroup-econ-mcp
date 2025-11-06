"""
核回归 (Kernel Regression)
基于 statsmodels.nonparametric 库实现
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np

try:
    from statsmodels.nonparametric.kernel_regression import KernelReg
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    KernelReg = None


class KernelRegressionResult(BaseModel):
    """核回归结果"""
    fitted_values: List[float] = Field(..., description="拟合值")
    residuals: List[float] = Field(..., description="残差")
    bandwidth: List[float] = Field(..., description="带宽参数")
    kernel_type: str = Field(..., description="核函数类型")
    n_observations: int = Field(..., description="观测数量")
    n_predictors: int = Field(..., description="预测变量数量")
    r_squared: float = Field(..., description="R²统计量")
    aic: Optional[float] = Field(None, description="AIC信息准则")
    summary: str = Field(..., description="摘要信息")


def kernel_regression(
    y_data: List[float],
    x_data: List[List[float]],
    kernel_type: str = "gaussian",
    bandwidth: Optional[List[float]] = None,
    bandwidth_method: str = "cv_ls",
    variable_type: Optional[str] = None
) -> KernelRegressionResult:
    """
    核回归估计
    
    Args:
        y_data: 因变量
        x_data: 自变量（二维列表）
        kernel_type: 核函数类型 - "gaussian"(高斯), "epanechnikov"(Epanechnikov核),
                     "uniform"(均匀核), "triangular"(三角核), "biweight"(双权核)
        bandwidth: 带宽参数（每个变量一个），如果为None则自动选择
        bandwidth_method: 带宽选择方法 - "cv_ls"(交叉验证最小二乘),
                          "aic"(AIC准则), "normal_reference"(正态参考)
        variable_type: 变量类型 - None(全部连续), "c"(连续), "u"(无序分类), "o"(有序分类)
                       可以是字符串（如 "cco"表示3个变量：连续、连续、有序）
        
    Returns:
        KernelRegressionResult: 核回归结果
        
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
    
    # 变量类型设置
    if variable_type is None:
        var_type = 'c' * k  # 默认全部为连续变量
    else:
        var_type = variable_type
        if len(var_type) != k:
            raise ValueError(f"variable_type长度({len(var_type)})与自变量数量({k})不一致")
    
    # 构建核回归模型
    try:
        if bandwidth is None:
            # 自动选择带宽
            kr = KernelReg(
                endog=y,
                exog=X,
                var_type=var_type,
                reg_type='ll',  # 局部线性回归
                bw=bandwidth_method
            )
        else:
            # 使用指定带宽
            if len(bandwidth) != k:
                raise ValueError(f"bandwidth长度({len(bandwidth)})与自变量数量({k})不一致")
            kr = KernelReg(
                endog=y,
                exog=X,
                var_type=var_type,
                reg_type='ll',
                bw=np.array(bandwidth)
            )
    except Exception as e:
        raise ValueError(f"核回归模型构建失败: {str(e)}")
    
    # 拟合值
    fitted_values, _ = kr.fit(X)
    fitted_values = fitted_values.flatten()
    
    # 残差
    residuals = y - fitted_values
    
    # 带宽
    bw = kr.bw.tolist() if hasattr(kr.bw, 'tolist') else [float(kr.bw)]
    
    # R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # AIC（近似计算）
    try:
        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
        aic = float(2 * k - 2 * log_likelihood)
    except:
        aic = None
    
    # 生成摘要
    summary = f"""核回归分析:
- 观测数量: {n}
- 预测变量: {k}
- 核函数: {kernel_type}
- 带宽: {[f'{b:.4f}' for b in bw]}
- 带宽方法: {bandwidth_method}
- R²: {r_squared:.4f}
"""
    if aic is not None:
        summary += f"- AIC: {aic:.2f}\n"
    
    return KernelRegressionResult(
        fitted_values=fitted_values.tolist(),
        residuals=residuals.tolist(),
        bandwidth=bw,
        kernel_type=kernel_type,
        n_observations=n,
        n_predictors=k,
        r_squared=r_squared,
        aic=aic,
        summary=summary
    )