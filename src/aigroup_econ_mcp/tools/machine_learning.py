
"""
机器学习集成模块
提供基于scikit-learn的机器学习算法，用于经济数据分析
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MLModelResult(BaseModel):
    """机器学习模型结果基类"""
    model_type: str = Field(description="模型类型")
    r2_score: float = Field(description="R²得分")
    mse: float = Field(description="均方误差")
    mae: float = Field(description="平均绝对误差")
    n_obs: int = Field(description="样本数量")
    feature_names: List[str] = Field(description="特征名称")
    feature_importance: Optional[Dict[str, float]] = Field(default=None, description="特征重要性")


class RandomForestResult(MLModelResult):
    """随机森林回归结果"""
    n_estimators: int = Field(description="树的数量")
    max_depth: int = Field(description="最大深度")
    oob_score: Optional[float] = Field(default=None, description="袋外得分")


class GradientBoostingResult(MLModelResult):
    """梯度提升树回归结果"""
    n_estimators: int = Field(description="树的数量")
    learning_rate: float = Field(description="学习率")
    max_depth: int = Field(description="最大深度")


class RegularizedRegressionResult(MLModelResult):
    """正则化回归结果"""
    alpha: float = Field(description="正则化强度")
    coefficients: Dict[str, float] = Field(description="回归系数")


class CrossValidationResult(BaseModel):
    """交叉验证结果"""
    model_type: str = Field(description="模型类型")
    cv_scores: List[float] = Field(description="交叉验证得分")
    mean_score: float = Field(description="平均得分")
    std_score: float = Field(description="标准差")
    n_splits: int = Field(description="交叉验证折数")


class FeatureImportanceResult(BaseModel):
    """特征重要性分析结果"""
    feature_importance: Dict[str, float] = Field(description="特征重要性分数")
    sorted_features: List[Tuple[str, float]] = Field(description="按重要性排序的特征")
    top_features: List[str] = Field(description="最重要的特征")


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


def cross_validation(
    y_data: List[float],
    x_data: List[List[float]],
    model_type: str = "random_forest",
    cv_folds: int = 5,
    scoring: str = "r2",
    **model_params
) -> CrossValidationResult:
    """
    交叉验证
    
    📊 功能说明：
    通过交叉验证评估模型的泛化能力和稳定性。
    
    📈 验证方法：
    - K折交叉验证：将数据分为K份，轮流使用K-1份训练，1份测试
    - 稳定性评估：通过多次验证评估模型稳定性
    - 泛化能力：评估模型在未见数据上的表现
    
    💡 使用场景：
    - 模型选择和比较
    - 超参数调优
    - 评估模型稳定性
    - 防止过拟合
    
    ⚠️ 注意事项：
    - 计算成本较高
    - 需要足够的数据量
    - 折数选择影响结果稳定性
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据，二维列表格式
        model_type: 模型类型（random_forest, gradient_boosting, lasso, ridge）
        cv_folds: 交叉验证折数，默认5
        scoring: 评分指标，默认"r2"
        **model_params: 模型参数
    
    Returns:
        CrossValidationResult: 交叉验证结果
    """
    # 数据验证
    if not y_data or not x_data:
        raise ValueError("因变量和自变量数据不能为空")
    
    if len(y_data) != len(x_data):
        raise ValueError(f"因变量和自变量的观测数量不一致: y_data={len(y_data)}, x_data={len(x_data)}")
    
    if cv_folds < 2 or cv_folds > len(y_data):
        raise ValueError(f"交叉验证折数应在2到样本数量之间: cv_folds={cv_folds}, n_obs={len(y_data)}")
    
    # 准备数据
    X = np.array(x_data)
    y = np.array(y_data)
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 选择模型
    if model_type == "random_forest":
        model = RandomForestRegressor(**model_params)
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(**model_params)
    elif model_type == "lasso":
        model = Lasso(**model_params)
    elif model_type == "ridge":
        model = Ridge(**model_params)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 执行交叉验证
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring)
    
    return CrossValidationResult(
        model_type=model_type,
        cv_scores=cv_scores.tolist(),
        mean_score=np.mean(cv_scores),
        std_score=np.std(cv_scores),
        n_splits=cv_folds
    )


def feature_importance_analysis(
    y_data: List[float],
    x_data: List[List[float]],
    feature_names: Optional[List[str]] = None,
    method: str = "random_forest",
    top_k: int = 5
) -> FeatureImportanceResult:
    """
    特征重要性分析
    
    📊 功能说明：
    分析各个特征对预测目标的重要性，帮助理解数据中的关键因素。
    
    📈 分析方法：
    - 基于模型：使用机器学习模型计算特征重要性
    - 排序分析：按重要性对特征进行排序
    - 关键特征识别：识别最重要的top-k个特征
    
    💡 使用场景：
    - 特征选择和降维
    - 模型可解释性分析
    - 业务洞察提取
    - 数据理解增强
    
    ⚠️ 注意事项：
    - 不同方法可能给出不同的重要性排序
    - 重要性分数是相对的，不是绝对的
    - 需要结合业务知识解释结果
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据，二维列表格式
        feature_names: 特征名称列表
        method: 分析方法（random_forest, gradient_boosting）
        top_k: 最重要的特征数量，默认5
    
    Returns:
        FeatureImportanceResult: 特征重要性分析结果
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
    
    # 选择模型并计算特征重要性
    if method == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif method == "gradient_boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"不支持的特征重要性分析方法: {method}")
    
    # 训练模型
    model.fit(X_scaled, y)
    
    # 获取特征重要性
    importance_scores = model.feature_importances_
    feature_importance = dict(zip(feature_names, importance_scores))
    
    # 按重要性排序
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # 获取最重要的特征
    top_features = [feature for feature, score in sorted_features[:top_k]]
    
    return FeatureImportanceResult(
        feature_importance=feature_importance,
        sorted_features=sorted_features,
        top_features=top_features
    )


def compare_ml_models(
    y_data: List[float],
    x_data: List[List[float]],
    feature_names: Optional[List[str]] = None,
    models: List[str] = None
) -> Dict[str, Any]:
    """
    比较多个机器学习模型
    
    📊 功能说明：
    同时运行多个机器学习模型并比较它们的性能，帮助选择最佳模型。
    
    📈 比较指标：
    - R²得分：模型解释方差的比例
    - 均方误差：预测误差的平方平均
    - 平均绝对误差：预测误差的绝对平均
    - 特征重要性：模型认为的重要特征
    
    💡 使用场景：
    - 模型选择和比较
    - 算法性能评估
    - 项目初始阶段模型筛选
    - 基准模型建立
    
    ⚠️ 注意事项：
    - 不同模型有不同的假设和适用场景
    - 需要结合交叉验证结果
    - 考虑模型复杂度和计算成本
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据，二维列表格式
        feature_names: 特征名称列表
        models: 要比较的模型列表，默认比较所有模型
    
    Returns:
        Dict[str, Any]: 模型比较结果
    """
    if models is None:
        models = ["random_forest", "gradient_boosting", "lasso", "ridge"]
    
    results = {}
    
    for model_name in models:
        try:
            if model_name == "random_forest":
                result = random_forest_regression(y_data, x_data, feature_names)
            elif model_name == "gradient_boosting":
                result = gradient_boosting_regression(y_data, x_data, feature_names)
            elif model_name == "lasso":
                result = lasso_regression(y_data, x_data, feature_names)
            elif model_name == "ridge":
                result = ridge_regression(y_data, x_data, feature_names)
            else:
                continue
            
            results[model_name] = result.model_dump()
            
        except Exception as e:
            print(f"模型 {model_name} 运行失败: {e}")
            continue
    
    # 找出最佳模型（基于R²得分）
    best_model = None
    best_r2 = -float('inf')
    
    for model_name, result in results.items():
        if result['r2_score'] > best_r2:
            best_r2 = result['r2_score']
            best_model = model_name
    
    return {
        "model_results": results,
        "best_model": best_model,
        "best_r2": best_r2,
        "comparison_summary": {
            "total_models": len(results),
            "successful_models": len(results),
            "best_performing": best_model
        }
    }