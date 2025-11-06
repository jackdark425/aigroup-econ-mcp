"""
空间回归模型
基于 spreg 库实现空间滞后模型(SAR)和空间误差模型(SEM)
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import numpy as np

try:
    from spreg import OLS as Spreg_OLS
    from spreg import ML_Lag, ML_Error, GM_Lag, GM_Error
    from libpysal.weights import W
    SPREG_AVAILABLE = True
except ImportError:
    SPREG_AVAILABLE = False
    ML_Lag = None
    ML_Error = None
    GM_Lag = None
    GM_Error = None
    W = None


class SpatialRegressionResult(BaseModel):
    """空间回归模型结果"""
    model_type: str = Field(..., description="模型类型 (SAR/SEM)")
    method: str = Field(..., description="估计方法 (ML/GMM)")
    coefficients: List[float] = Field(..., description="回归系数")
    std_errors: List[float] = Field(..., description="标准误")
    z_scores: List[float] = Field(..., description="Z统计量")
    p_values: List[float] = Field(..., description="P值")
    feature_names: List[str] = Field(..., description="特征名称")
    spatial_param: float = Field(..., description="空间参数（rho或lambda）")
    spatial_param_se: float = Field(..., description="空间参数标准误")
    r_squared: Optional[float] = Field(None, description="伪R方")
    log_likelihood: float = Field(..., description="对数似然值")
    aic: float = Field(..., description="AIC信息准则")
    schwarz: float = Field(..., description="Schwarz准则(BIC)")
    n_observations: int = Field(..., description="观测数量")
    summary: str = Field(..., description="摘要信息")


def spatial_lag_model(
    y_data: List[float],
    x_data: List[List[float]],
    neighbors: dict,
    weights: Optional[dict] = None,
    feature_names: Optional[List[str]] = None,
    method: str = "ml"
) -> SpatialRegressionResult:
    """
    空间滞后模型 (Spatial Lag Model - SAR)
    模型形式: y = ρWy + Xβ + ε
    
    Args:
        y_data: 因变量
        x_data: 自变量（二维列表）
        neighbors: 邻居字典
        weights: 权重字典
        feature_names: 特征名称
        method: 估计方法 - "ml"(最大似然) 或 "gmm"(广义矩估计)
        
    Returns:
        SpatialRegressionResult: 空间滞后模型结果
        
    Raises:
        ImportError: spreg库未安装
        ValueError: 输入数据无效
    """
    if not SPREG_AVAILABLE:
        raise ImportError(
            "spreg库未安装。请运行: pip install spreg\n"
            "或: pip install pysal"
        )
    
    # 输入验证
    if not y_data or not x_data:
        raise ValueError("y_data和x_data不能为空")
    
    # 数据准备
    y = np.array(y_data).reshape(-1, 1)
    X = np.array(x_data)
    
    # 确保X是二维数组
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n = len(y)
    k = X.shape[1]
    
    # 构建权重对象
    if weights is None:
        weights = {i: [1.0] * len(neighbors[i]) for i in neighbors}
    
    # 确保邻居字典的键是整数
    neighbors_int = {int(k): [int(n) for n in v] for k, v in neighbors.items()}
    weights_int = {int(k): v for k, v in weights.items()}
    
    w = W(neighbors_int, weights_int)
    w.transform = 'r'  # 行标准化
    
    # 特征名称
    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(k)]
    
    # 估计模型
    if method.lower() == "ml":
        model = ML_Lag(y, X, w, name_y='y', name_x=feature_names)
    elif method.lower() == "gmm":
        model = GM_Lag(y, X, w, name_y='y', name_x=feature_names)
    else:
        raise ValueError(f"不支持的方法: {method}。支持: ml, gmm")
    
    # 提取结果
    # 系数包括常数项和自变量系数
    coefficients = model.betas.flatten().tolist()
    
    # 标准误、Z值、P值
    std_errors = np.sqrt(np.diag(model.vm)).tolist()
    
    # 处理z_stat - 可能是列表或numpy数组
    if hasattr(model.z_stat, 'shape'):
        # numpy数组
        z_scores = model.z_stat[:, 0].tolist()
        p_values = model.z_stat[:, 1].tolist()
    else:
        # 列表
        z_scores = [stat[0] for stat in model.z_stat] if model.z_stat else []
        p_values = [stat[1] for stat in model.z_stat] if model.z_stat else []
    
    # 空间参数（rho）
    # 在ML_Lag中，rho是最后一个参数
    spatial_param = float(model.rho)
    
    # 尝试获取rho的标准误
    try:
        # rho的标准误通常在vm矩阵的最后一个对角元素
        spatial_param_se = float(np.sqrt(model.vm[-1, -1]))
    except:
        spatial_param_se = 0.0
    
    # 伪R方（如果可用）
    try:
        r_squared = float(model.pr2) if hasattr(model, 'pr2') else None
    except:
        r_squared = None
    
    # 对数似然值
    log_likelihood = float(model.logll) if hasattr(model, 'logll') else 0.0
    
    # 信息准则
    aic = float(model.aic) if hasattr(model, 'aic') else 0.0
    schwarz = float(model.schwarz) if hasattr(model, 'schwarz') else 0.0
    
    # 添加常数项到特征名称
    all_feature_names = ['const'] + feature_names
    
    # 生成摘要
    summary = f"""空间滞后模型 (SAR) - {method.upper()}估计:
- 观测数量: {n}
- 自变量数: {k}
- 空间参数 ρ: {spatial_param:.4f} (标准误: {spatial_param_se:.4f})
- 对数似然: {log_likelihood:.2f}
- AIC: {aic:.2f}
- BIC: {schwarz:.2f}
"""
    if r_squared is not None:
        summary += f"- 伪R²: {r_squared:.4f}\n"
    
    return SpatialRegressionResult(
        model_type="SAR",
        method=method.upper(),
        coefficients=coefficients,
        std_errors=std_errors,
        z_scores=z_scores,
        p_values=p_values,
        feature_names=all_feature_names,
        spatial_param=spatial_param,
        spatial_param_se=spatial_param_se,
        r_squared=r_squared,
        log_likelihood=log_likelihood,
        aic=aic,
        schwarz=schwarz,
        n_observations=n,
        summary=summary
    )


def spatial_error_model(
    y_data: List[float],
    x_data: List[List[float]],
    neighbors: dict,
    weights: Optional[dict] = None,
    feature_names: Optional[List[str]] = None,
    method: str = "ml"
) -> SpatialRegressionResult:
    """
    空间误差模型 (Spatial Error Model - SEM)
    模型形式: y = Xβ + u, u = λWu + ε
    
    Args:
        y_data: 因变量
        x_data: 自变量（二维列表）
        neighbors: 邻居字典
        weights: 权重字典
        feature_names: 特征名称
        method: 估计方法 - "ml"(最大似然) 或 "gmm"(广义矩估计)
        
    Returns:
        SpatialRegressionResult: 空间误差模型结果
    """
    if not SPREG_AVAILABLE:
        raise ImportError("spreg库未安装")
    
    # 输入验证
    if not y_data or not x_data:
        raise ValueError("y_data和x_data不能为空")
    
    # 数据准备
    y = np.array(y_data).reshape(-1, 1)
    X = np.array(x_data)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n = len(y)
    k = X.shape[1]
    
    # 构建权重对象
    if weights is None:
        weights = {i: [1.0] * len(neighbors[i]) for i in neighbors}
    
    # 确保邻居字典的键是整数
    neighbors_int = {int(k): [int(n) for n in v] for k, v in neighbors.items()}
    weights_int = {int(k): v for k, v in weights.items()}
    
    w = W(neighbors_int, weights_int)
    w.transform = 'r'
    
    # 特征名称
    if feature_names is None:
        feature_names = [f"X{i+1}" for i in range(k)]
    
    # 估计模型
    if method.lower() == "ml":
        model = ML_Error(y, X, w, name_y='y', name_x=feature_names)
    elif method.lower() == "gmm":
        model = GM_Error(y, X, w, name_y='y', name_x=feature_names)
    else:
        raise ValueError(f"不支持的方法: {method}")
    
    # 提取结果
    coefficients = model.betas.flatten().tolist()
    std_errors = np.sqrt(np.diag(model.vm)).tolist()
    
    # 处理z_stat - 可能是列表或numpy数组
    if hasattr(model.z_stat, 'shape'):
        # numpy数组
        z_scores = model.z_stat[:, 0].tolist()
        p_values = model.z_stat[:, 1].tolist()
    else:
        # 列表
        z_scores = [stat[0] for stat in model.z_stat] if model.z_stat else []
        p_values = [stat[1] for stat in model.z_stat] if model.z_stat else []
    
    # 空间参数（lambda）
    spatial_param = float(model.lam)
    
    try:
        spatial_param_se = float(np.sqrt(model.vm[-1, -1]))
    except:
        spatial_param_se = 0.0
    
    # 伪R方
    try:
        r_squared = float(model.pr2) if hasattr(model, 'pr2') else None
    except:
        r_squared = None
    
    # 对数似然值和信息准则
    log_likelihood = float(model.logll) if hasattr(model, 'logll') else 0.0
    aic = float(model.aic) if hasattr(model, 'aic') else 0.0
    schwarz = float(model.schwarz) if hasattr(model, 'schwarz') else 0.0
    
    all_feature_names = ['const'] + feature_names
    
    # 生成摘要
    summary = f"""空间误差模型 (SEM) - {method.upper()}估计:
- 观测数量: {n}
- 自变量数: {k}
- 空间参数 λ: {spatial_param:.4f} (标准误: {spatial_param_se:.4f})
- 对数似然: {log_likelihood:.2f}
- AIC: {aic:.2f}
- BIC: {schwarz:.2f}
"""
    if r_squared is not None:
        summary += f"- 伪R²: {r_squared:.4f}\n"
    
    return SpatialRegressionResult(
        model_type="SEM",
        method=method.upper(),
        coefficients=coefficients,
        std_errors=std_errors,
        z_scores=z_scores,
        p_values=p_values,
        feature_names=all_feature_names,
        spatial_param=spatial_param,
        spatial_param_se=spatial_param_se,
        r_squared=r_squared,
        log_likelihood=log_likelihood,
        aic=aic,
        schwarz=schwarz,
        n_observations=n,
        summary=summary
    )