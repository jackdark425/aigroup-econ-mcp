"""
集成学习方法模块
包含随机森林和梯度提升树回归算法
"""

import numpy as np
from typing import List, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from .ml_models import RandomForestResult, GradientBoostingResult


def random_forest_regression(
    y_data: List[float],
    x_data: List[List[float]],
    feature_names: Optional[List[str]] = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42
) -> RandomForestResult:
    """
    随机森林回归
    
    📊 功能说明：
    使用随机森林算法进行回归分析，适用于非线性关系和复杂交互效应。
    
    📈 算法特点：
    - 集成学习：多个决策树的组合
    - 抗过拟合：通过袋外样本和特征随机选择
    - 非线性建模：能够捕捉复杂的非线性关系
    - 特征重要性：提供特征重要性排序
    
    💡 使用场景：
    - 复杂非线性关系建模
    - 特征重要性分析
    - 高维数据回归
    - 稳健预测建模
    
    ⚠️ 注意事项：
    - 计算复杂度较高
    - 需要调整超参数（n_estimators, max_depth）
    - 对异常值相对稳健
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据，二维列表格式
        feature_names: 特征名称列表
        n_estimators: 树的数量，默认100
        max_depth: 最大深度，None表示不限制
        random_state: 随机种子
    
    Returns:
        RandomForestResult: 随机森林回归结果
    """
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
    
    # 训练随机森林模型
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        oob_score=True
    )
    rf_model.fit(X_scaled, y)
    
    # 预测
    y_pred = rf_model.predict(X_scaled)
    
    # 计算评估指标
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # 特征重要性
    feature_importance = dict(zip(feature_names, rf_model.feature_importances_))
    
    return RandomForestResult(
        model_type="random_forest",
        r2_score=r2,
        mse=mse,
        mae=mae,
        n_obs=len(y),
        feature_names=feature_names,
        feature_importance=feature_importance,
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth is not None else 0,  # 0表示无限制
        oob_score=rf_model.oob_score_ if hasattr(rf_model, 'oob_score_') else None
    )


def gradient_boosting_regression(
    y_data: List[float],
    x_data: List[List[float]],
    feature_names: Optional[List[str]] = None,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int = 42
) -> GradientBoostingResult:
    """
    梯度提升树回归
    
    📊 功能说明：
    使用梯度提升算法进行回归分析，通过逐步优化残差来提升模型性能。
    
    📈 算法特点：
    - 逐步优化：通过梯度下降逐步改进模型
    - 高精度：通常比随机森林有更好的预测精度
    - 正则化：通过学习率和树深度控制过拟合
    - 特征重要性：提供特征重要性排序
    
    💡 使用场景：
    - 高精度预测需求
    - 结构化数据建模
    - 竞赛和实际应用
    - 需要精细调优的场景
    
    ⚠️ 注意事项：
    - 对超参数敏感
    - 训练时间较长
    - 容易过拟合（需要仔细调参）
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据，二维列表格式
        feature_names: 特征名称列表
        n_estimators: 树的数量，默认100
        learning_rate: 学习率，默认0.1
        max_depth: 最大深度，默认3
        random_state: 随机种子
    
    Returns:
        GradientBoostingResult: 梯度提升树回归结果
    """
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
    
    # 训练梯度提升树模型
    gb_model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    gb_model.fit(X_scaled, y)
    
    # 预测
    y_pred = gb_model.predict(X_scaled)
    
    # 计算评估指标
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # 特征重要性
    feature_importance = dict(zip(feature_names, gb_model.feature_importances_))
    
    return GradientBoostingResult(
        model_type="gradient_boosting",
        r2_score=r2,
        mse=mse,
        mae=mae,
        n_obs=len(y),
        feature_names=feature_names,
        feature_importance=feature_importance,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )