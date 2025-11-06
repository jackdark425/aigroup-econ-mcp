"""
广义可加模型 (Generalized Additive Model - GAM)
基于 pygam 库实现
"""

from typing import List, Optional
from pydantic import BaseModel, Field
import numpy as np

try:
    from pygam import LinearGAM, LogisticGAM, s, f
    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False
    LinearGAM = None


class GAMResult(BaseModel):
    """GAM模型结果"""
    fitted_values: List[float] = Field(..., description="拟合值")
    residuals: List[float] = Field(..., description="残差")
    deviance: float = Field(..., description="偏差")
    aic: float = Field(..., description="AIC信息准则")
    aicc: float = Field(..., description="AICc信息准则")
    r_squared: float = Field(..., description="伪R²")
    n_splines: List[int] = Field(..., description="每个特征的样条数")
    problem_type: str = Field(..., description="问题类型")
    n_observations: int = Field(..., description="观测数量")
    summary: str = Field(..., description="摘要信息")


def gam_model(
    y_data: List[float],
    x_data: List[List[float]],
    problem_type: str = "regression",
    n_splines: int = 10,
    lam: float = 0.6
) -> GAMResult:
    """
    广义可加模型
    
    Args:
        y_data: 因变量
        x_data: 自变量（二维列表）
        problem_type: 问题类型 - "regression"(回归) 或 "classification"(分类)
        n_splines: 每个特征的样条数
        lam: 平滑参数（lambda）
        
    Returns:
        GAMResult: GAM模型结果
    """
    if not PYGAM_AVAILABLE:
        raise ImportError("pygam库未安装。请运行: pip install pygam")
    
    # 数据准备
    y = np.array(y_data, dtype=np.float64)
    X = np.array(x_data, dtype=np.float64)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n, k = X.shape
    
    # 创建GAM模型
    if problem_type == "regression":
        gam = LinearGAM(s(0, n_splines=n_splines, lam=lam))
        for i in range(1, k):
            gam = LinearGAM(s(i, n_splines=n_splines, lam=lam))
    elif problem_type == "classification":
        gam = LogisticGAM(s(0, n_splines=n_splines, lam=lam))
    else:
        raise ValueError(f"不支持的问题类型: {problem_type}")
    
    # 拟合模型
    gam.fit(X, y)
    
    # 拟合值
    y_pred = gam.predict(X)
    
    # 残差
    residuals = y - y_pred
    
    # 模型统计量
    deviance = float(gam.statistics_['deviance'])
    aic = float(gam.statistics_['AIC'])
    aicc = float(gam.statistics_['AICc'])
    
    # 伪R²
    r_squared = float(gam.statistics_['pseudo_r2']['explained_deviance'])
    
    # 样条数信息
    n_splines_list = [n_splines] * k
    
    summary = f"""广义可加模型 (GAM):
- 观测数量: {n}
- 特征数量: {k}
- 问题类型: {problem_type}
- 样条数: {n_splines}
- 平滑参数: {lam}
- 偏差: {deviance:.4f}
- AIC: {aic:.2f}
- AICc: {aicc:.2f}
- 伪R²: {r_squared:.4f}
"""
    
    return GAMResult(
        fitted_values=y_pred.tolist(),
        residuals=residuals.tolist(),
        deviance=deviance,
        aic=aic,
        aicc=aicc,
        r_squared=r_squared,
        n_splines=n_splines_list,
        problem_type=problem_type,
        n_observations=n,
        summary=summary
    )