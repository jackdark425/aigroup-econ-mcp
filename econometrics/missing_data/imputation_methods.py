"""
缺失数据插补方法
基于 sklearn.impute 实现
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import numpy as np

try:
    from sklearn.impute import SimpleImputer, IterativeImputer
    from sklearn.experimental import enable_iterative_imputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SimpleImputer = None
    IterativeImputer = None


class SimpleImputationResult(BaseModel):
    """简单插补结果"""
    imputed_data: List[List[float]] = Field(..., description="插补后的数据")
    missing_mask: List[List[bool]] = Field(..., description="缺失值掩码")
    n_missing: int = Field(..., description="缺失值总数")
    missing_rate: float = Field(..., description="缺失率")
    imputation_method: str = Field(..., description="插补方法")
    fill_values: List[float] = Field(..., description="填充值（每列）")
    n_observations: int = Field(..., description="观测数量")
    n_features: int = Field(..., description="特征数量")
    summary: str = Field(..., description="摘要信息")


class MultipleImputationResult(BaseModel):
    """多重插补结果"""
    imputed_datasets: List[List[List[float]]] = Field(..., description="多个插补数据集")
    n_imputations: int = Field(..., description="插补次数")
    missing_mask: List[List[bool]] = Field(..., description="缺失值掩码")
    n_missing: int = Field(..., description="缺失值总数")
    missing_rate: float = Field(..., description="缺失率")
    convergence_info: Dict = Field(..., description="收敛信息")
    n_observations: int = Field(..., description="观测数量")
    n_features: int = Field(..., description="特征数量")
    summary: str = Field(..., description="摘要信息")


def simple_imputation(
    data: List[List[float]],
    strategy: str = "mean",
    fill_value: Optional[float] = None
) -> SimpleImputationResult:
    """
    简单插补方法
    
    Args:
        data: 含缺失值的数据（二维列表，NaN表示缺失）
        strategy: 插补策略 - "mean"(均值), "median"(中位数), 
                  "most_frequent"(众数), "constant"(常数)
        fill_value: 当strategy="constant"时使用的填充值
        
    Returns:
        SimpleImputationResult: 简单插补结果
        
    Raises:
        ImportError: sklearn库未安装
        ValueError: 输入数据无效
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn库未安装。请运行: pip install scikit-learn")
    
    # 输入验证
    if not data:
        raise ValueError("data不能为空")
    
    # 转换为numpy数组
    X = np.array(data, dtype=np.float64)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n, k = X.shape
    
    # 创建缺失值掩码
    missing_mask = np.isnan(X)
    n_missing = int(missing_mask.sum())
    missing_rate = float(n_missing / (n * k))
    
    # 简单插补
    if strategy == "constant":
        if fill_value is None:
            fill_value = 0.0
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    else:
        imputer = SimpleImputer(strategy=strategy)
    
    # 执行插补
    X_imputed = imputer.fit_transform(X)
    
    # 填充值
    fill_values = imputer.statistics_.tolist()
    
    # 生成摘要
    summary = f"""简单插补:
- 观测数量: {n}
- 特征数量: {k}
- 缺失值数量: {n_missing}
- 缺失率: {missing_rate*100:.2f}%
- 插补策略: {strategy}

各列填充值:
"""
    for i, val in enumerate(fill_values):
        col_missing = int(missing_mask[:, i].sum())
        summary += f"  列{i+1}: {val:.4f} (缺失{col_missing}个)\n"
    
    return SimpleImputationResult(
        imputed_data=X_imputed.tolist(),
        missing_mask=missing_mask.tolist(),
        n_missing=n_missing,
        missing_rate=missing_rate,
        imputation_method=strategy,
        fill_values=fill_values,
        n_observations=n,
        n_features=k,
        summary=summary
    )


def multiple_imputation(
    data: List[List[float]],
    n_imputations: int = 5,
    max_iter: int = 10,
    random_state: Optional[int] = None
) -> MultipleImputationResult:
    """
    多重插补 (MICE - Multivariate Imputation by Chained Equations)
    
    Args:
        data: 含缺失值的数据
        n_imputations: 生成的插补数据集数量
        max_iter: 最大迭代次数
        random_state: 随机种子
        
    Returns:
        MultipleImputationResult: 多重插补结果
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn库未安装")
    
    # 输入验证
    if not data:
        raise ValueError("data不能为空")
    
    X = np.array(data, dtype=np.float64)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n, k = X.shape
    
    # 缺失值统计
    missing_mask = np.isnan(X)
    n_missing = int(missing_mask.sum())
    missing_rate = float(n_missing / (n * k))
    
    # 执行多重插补
    imputed_datasets = []
    convergence_info = {"iterations": [], "converged": []}
    
    for i in range(n_imputations):
        # 设置随机种子
        seed = random_state + i if random_state is not None else None
        
        # 创建迭代插补器
        imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=seed,
            verbose=0
        )
        
        # 执行插补
        X_imputed = imputer.fit_transform(X)
        imputed_datasets.append(X_imputed.tolist())
        
        # 记录收敛信息
        convergence_info["iterations"].append(imputer.n_iter_)
        convergence_info["converged"].append(imputer.n_iter_ < max_iter)
    
    # 计算平均收敛迭代数
    avg_iter = np.mean(convergence_info["iterations"])
    n_converged = sum(convergence_info["converged"])
    
    # 生成摘要
    summary = f"""多重插补 (MICE):
- 观测数量: {n}
- 特征数量: {k}
- 缺失值数量: {n_missing}
- 缺失率: {missing_rate*100:.2f}%
- 插补次数: {n_imputations}
- 最大迭代: {max_iter}

收敛信息:
- 平均迭代数: {avg_iter:.1f}
- 收敛数据集: {n_converged}/{n_imputations}

说明: 生成{n_imputations}个完整的插补数据集，
可用于后续分析并合并结果（Rubin规则）
"""
    
    return MultipleImputationResult(
        imputed_datasets=imputed_datasets,
        n_imputations=n_imputations,
        missing_mask=missing_mask.tolist(),
        n_missing=n_missing,
        missing_rate=missing_rate,
        convergence_info=convergence_info,
        n_observations=n,
        n_features=k,
        summary=summary
    )