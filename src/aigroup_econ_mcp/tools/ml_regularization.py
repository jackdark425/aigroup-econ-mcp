"""
正则化回归方法模块
包含Lasso和Ridge回归算法
"""

import numpy as np
from typing import List, Optional
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from .ml_models import RegularizedRegressionResult


def lasso_regression(
    y_data: List[float],
    x_data: List[List[float]],
    feature_names: Optional[List[str]] = None,
    alpha: float = 1.0,
    random_state: int = 42
) -> RegularizedRegressionResult:
    """
    Lasso回归（L1正则化）
    
    📊 功能说明：
    使用L1正则化的线性回归，能够进行特征选择和稀疏建模。
    
    📈 算法特点：
    - 特征选择：自动将不重要的特征系数压缩为0
    - 稀疏解：产生稀疏的系数向量
    - 可解释性：保留重要特征，去除冗余特征
    - 处理多重共线性：对高度相关的特征进行选择
    
    💡 使用场景：
    - 高维数据特征选择
    - 多重共线性问题
    - 稀疏建模需求
    - 可解释性要求高的场景
    
    ⚠️ 注意事项：
    - 对alpha参数敏感
    - 可能过度压缩重要特征
    - 需要数据标准化
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据，二维列表格式
        feature_names: 特征名称列表
        alpha: 正则化强度，默认1.0
        random_state: 随机种子
    
    Returns:
        RegularizedRegressionResult: Lasso回归结果
    """
    return _regularized_regression(
        y_data, x_data, feature_names, alpha, random_state, "lasso"
    )


def ridge_regression(
    y_data: List[float],
    x_data: List[List[float]],
    feature_names: Optional[List[str]] = None,
    alpha: float = 1.0,
    random_state: int = 42
) -> RegularizedRegressionResult:
    """
    Ridge回归（L2正则化）
    
    📊 功能说明：
    使用L2正则化的线性回归，能够处理多重共线性问题。
    
    📈 算法特点：
    - 稳定性：对多重共线性稳健
    - 收缩系数：将所有系数向0收缩
    - 无特征选择：保留所有特征
    - 数值稳定性：改善矩阵条件数
    
    💡 使用场景：
    - 多重共线性问题
    - 需要稳定估计的场景
    - 所有特征都可能有贡献的情况
    - 小样本高维数据
    
    ⚠️ 注意事项：
    - 不进行特征选择
    - 对alpha参数敏感
    - 需要数据标准化
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据，二维列表格式
        feature_names: 特征名称列表
        alpha: 正则化强度，默认1.0
        random_state: 随机种子
    
    Returns:
        RegularizedRegressionResult: Ridge回归结果
    """
    return _regularized_regression(
        y_data, x_data, feature_names, alpha, random_state, "ridge"
    )


def _regularized_regression(
    y_data: List[float],
    x_data: List[List[float]],
    feature_names: Optional[List[str]],
    alpha: float,
    random_state: int,
    model_type: str
) -> RegularizedRegressionResult:
    """正则化回归内部实现"""
    # 数据验证
    if not y_data or not x_data:
        raise ValueError("因变量和自变量数据不能为空")
    
    if len(y_data) != len(x_data):
        raise ValueError(f"因变量和自变量的观测数量不一致: y_data={len(y_data)}, x_data={len(x_data)}")
    
    # 准备数据
    X = np.array(x_data)
    y = np.array(y_data)
    
    # 特征名称处理
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    elif len(feature_names) != X.shape[1]:
        raise ValueError(f"特征名称数量({len(feature_names)})与自变量数量({X.shape[1]})不匹配")
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = (y - np.mean(y)) / np.std(y)  # 标准化因变量
    
    # 选择模型
    if model_type == "lasso":
        model = Lasso(alpha=alpha, random_state=random_state, max_iter=10000)
    elif model_type == "ridge":
        model = Ridge(alpha=alpha, random_state=random_state)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 训练模型
    model.fit(X_scaled, y_scaled)
    
    # 预测
    y_pred_scaled = model.predict(X_scaled)
    
    # 将预测值转换回原始尺度
    y_pred = y_pred_scaled * np.std(y) + np.mean(y)
    
    # 计算评估指标
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # 系数（注意：由于标准化，系数需要适当解释）
    coefficients = dict(zip(feature_names, model.coef_))
    
    return RegularizedRegressionResult(
        model_type=model_type,
        r2_score=r2,
        mse=mse,
        mae=mae,
        n_obs=len(y),
        feature_names=feature_names,
        alpha=alpha,
        coefficients=coefficients
    )