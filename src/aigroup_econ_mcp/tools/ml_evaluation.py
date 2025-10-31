"""
机器学习评估和比较模块
包含交叉验证、特征重要性分析和模型比较功能
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
import warnings
warnings.filterwarnings('ignore')

from .ml_models import CrossValidationResult, FeatureImportanceResult
from .ml_ensemble import random_forest_regression, gradient_boosting_regression
from .ml_regularization import lasso_regression, ridge_regression


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