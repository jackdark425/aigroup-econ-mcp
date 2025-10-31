"""
机器学习模型数据类定义
定义各种机器学习算法的结果数据结构
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


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