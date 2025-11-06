"""
空间杜宾模型 (Spatial Durbin Model - SDM)
基于 spreg 库实现
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import numpy as np

try:
    from spreg import OLS_Regimes, ML_Lag
    from libpysal.weights import W
    SPREG_AVAILABLE = True
except ImportError:
    SPREG_AVAILABLE = False
    W = None


class SpatialDurbinResult(BaseModel):
    """空间杜宾模型结果"""
    coefficients: List[float] = Field(..., description="回归系数")
    std_errors: List[float] = Field(..., description="标准误")
    z_scores: List[float] = Field(..., description="Z统计量")
    p_values: List[float] = Field(..., description="P值")
    feature_names: List[str] = Field(..., description="特征名称（包括WX）")
    spatial_lag_coef: float = Field(..., description="空间滞后系数ρ")
    spatial_lag_se: float = Field(..., description="空间滞后系数标准误")
    log_likelihood: float = Field(..., description="对数似然值")
    aic: float = Field(..., description="AIC信息准则")
    schwarz: float = Field(..., description="BIC信息准则")
    n_observations: int = Field(..., description="观测数量")
    summary: str = Field(..., description="摘要信息")


def spatial_durbin_model(
    y_data: List[float],
    x_data: List[List[float]],
    neighbors: dict,
    weights: Optional[dict] = None,
    feature_names: Optional[List[str]] = None
) -> SpatialDurbinResult:
    """
    空间杜宾模型 (SDM)
    模型形式: y = ρWy + Xβ + WXθ + ε
    
    Args:
        y_data: 因变量
        x_data: 自变量（二维列表）
        neighbors: 邻居字典
        weights: 权重字典
        feature_names: 特征名称
        
    Returns:
        SpatialDurbinResult: 空间杜宾模型结果
        
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
    
    # 计算WX（空间滞后的自变量）
    try:
        from scipy import sparse
        W_matrix = w.sparse
        WX = W_matrix.dot(X)
    except ImportError:
        # 如果scipy不可用，使用numpy实现
        W_matrix = np.zeros((n, n))
        for i in w.neighbors:
            for j_idx, j in enumerate(w.neighbors[i]):
                W_matrix[i, j] = w.weights[i][j_idx]
        WX = W_matrix.dot(X)
    
    # 合并X和WX
    X_full = np.hstack([X, WX])
    
    # 特征名称（包括WX）
    wx_names = [f"W_{name}" for name in feature_names]
    all_feature_names = feature_names + wx_names
    
    # 使用ML_Lag但包含WX作为额外的解释变量
    # 这实际上是SDM的一种实现方式
    try:
        # 创建包含WX的模型
        model = ML_Lag(y, X_full, w, name_y='y', name_x=all_feature_names)
    except Exception as e:
        raise ValueError(f"空间杜宾模型估计失败: {str(e)}")
    
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
    
    # 空间滞后系数
    spatial_lag_coef = float(model.rho)
    try:
        spatial_lag_se = float(np.sqrt(model.vm[-1, -1]))
    except:
        spatial_lag_se = 0.0
    
    # 模型拟合指标
    log_likelihood = float(model.logll) if hasattr(model, 'logll') else 0.0
    aic = float(model.aic) if hasattr(model, 'aic') else 0.0
    schwarz = float(model.schwarz) if hasattr(model, 'schwarz') else 0.0
    
    # 添加常数项到特征名称
    final_feature_names = ['const'] + all_feature_names
    
    # 生成摘要
    summary = f"""空间杜宾模型 (SDM):
- 观测数量: {n}
- 自变量数: {k}
- 空间滞后系数 ρ: {spatial_lag_coef:.4f} (SE: {spatial_lag_se:.4f})
- 对数似然: {log_likelihood:.2f}
- AIC: {aic:.2f}
- BIC: {schwarz:.2f}

说明: SDM同时包含Wy和WX，捕捉自变量的空间溢出效应
"""
    
    return SpatialDurbinResult(
        coefficients=coefficients,
        std_errors=std_errors,
        z_scores=z_scores,
        p_values=p_values,
        feature_names=final_feature_names,
        spatial_lag_coef=spatial_lag_coef,
        spatial_lag_se=spatial_lag_se,
        log_likelihood=log_likelihood,
        aic=aic,
        schwarz=schwarz,
        n_observations=n,
        summary=summary
    )