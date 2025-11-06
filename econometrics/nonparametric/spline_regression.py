"""
样条回归
基于 sklearn 和 scipy 实现
"""

from typing import List, Optional
from pydantic import BaseModel, Field
import numpy as np

try:
    from sklearn.preprocessing import SplineTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SplineTransformer = None


class SplineRegressionResult(BaseModel):
    """样条回归结果"""
    fitted_values: List[float] = Field(..., description="拟合值")
    residuals: List[float] = Field(..., description="残差")
    coefficients: List[float] = Field(..., description="样条基函数系数")
    n_knots: int = Field(..., description="节点数")
    degree: int = Field(..., description="样条次数")
    r_squared: float = Field(..., description="R²")
    n_observations: int = Field(..., description="观测数量")
    summary: str = Field(..., description="摘要信息")


def spline_regression(
    y_data: List[float],
    x_data: List[float],
    n_knots: int = 5,
    degree: int = 3,
    knots: str = "uniform"
) -> SplineRegressionResult:
    """
    样条回归
    
    Args:
        y_data: 因变量
        x_data: 自变量（单变量）
        n_knots: 节点数量
        degree: 样条次数（通常3表示三次样条）
        knots: 节点分布 - "uniform"(均匀), "quantile"(分位数)
        
    Returns:
        SplineRegressionResult: 样条回归结果
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn库未安装。请运行: pip install scikit-learn")
    
    # 数据准备
    y = np.array(y_data, dtype=np.float64)
    X = np.array(x_data, dtype=np.float64).reshape(-1, 1)
    
    n = len(y)
    
    # 创建样条转换器+线性回归管道
    pipeline = Pipeline([
        ('spline', SplineTransformer(n_knots=n_knots, degree=degree, knots=knots)),
        ('linear', LinearRegression())
    ])
    
    # 拟合模型
    pipeline.fit(X, y)
    
    # 预测
    y_pred = pipeline.predict(X)
    
    # 残差和R²
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # 系数
    coefficients = pipeline.named_steps['linear'].coef_.tolist()
    
    summary = f"""样条回归:
- 观测数量: {n}
- 节点数: {n_knots}
- 样条次数: {degree}
- 节点分布: {knots}
- R²: {r_squared:.4f}
- 样条基函数数量: {len(coefficients)}
"""
    
    return SplineRegressionResult(
        fitted_values=y_pred.tolist(),
        residuals=residuals.tolist(),
        coefficients=coefficients,
        n_knots=n_knots,
        degree=degree,
        r_squared=r_squared,
        n_observations=n,
        summary=summary
    )