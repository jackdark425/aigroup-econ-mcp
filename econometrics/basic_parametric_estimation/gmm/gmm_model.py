"""
广义矩估计 (GMM) 模型实现
"""

from typing import List, Optional
from pydantic import BaseModel, Field
import numpy as np
from scipy import stats


class GMMResult(BaseModel):
    """广义矩估计结果"""
    coefficients: List[float] = Field(..., description="估计系数")
    std_errors: List[float] = Field(..., description="系数标准误")
    t_values: List[float] = Field(..., description="t统计量")
    p_values: List[float] = Field(..., description="p值")
    conf_int_lower: List[float] = Field(..., description="置信区间下界")
    conf_int_upper: List[float] = Field(..., description="置信区间上界")
    j_statistic: float = Field(..., description="J统计量")
    j_p_value: float = Field(..., description="J统计量p值")
    weight_matrix: List[List[float]] = Field(..., description="权重矩阵")
    n_obs: int = Field(..., description="观测数量")
    n_moments: int = Field(..., description="矩条件数量")
    feature_names: List[str] = Field(..., description="特征名称")


def _safe_inverse(matrix, reg_param=1e-10):
    """安全的矩阵求逆函数"""
    try:
        # 检查矩阵是否为空或非二维
        if matrix.size == 0 or matrix.ndim != 2:
            raise ValueError("矩阵为空或不是二维数组")
        
        # 检查矩阵是否包含无效值
        if np.isnan(matrix).any() or np.isinf(matrix).any():
            raise ValueError("矩阵包含NaN或无穷大值")
            
        # 尝试直接求逆
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，添加正则化项
        try:
            # 确保矩阵是方阵
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("矩阵不是方阵，无法求逆")
                
            # 添加正则化项
            reg_matrix = matrix + np.eye(matrix.shape[0]) * reg_param
            return np.linalg.inv(reg_matrix)
        except np.linalg.LinAlgError:
            # 如果仍然失败，使用伪逆
            return np.linalg.pinv(matrix)


def gmm_estimation(
    y_data: List[float],
    x_data: List[List[float]], 
    instruments: Optional[List[List[float]]] = None,
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    confidence_level: float = 0.95
) -> GMMResult:
    """
    广义矩估计
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        instruments: 工具变量数据 (如果为None，则使用x_data作为工具变量，退化为OLS)
        feature_names: 特征名称
        constant: 是否包含常数项
        confidence_level: 置信水平
        
    Returns:
        GMMResult: 广义矩估计结果
        
    Raises:
        ValueError: 当输入数据无效时抛出异常
    """
    # 输入验证
    if not y_data or not x_data:
        raise ValueError("因变量和自变量数据不能为空")
    
    # 转换为numpy数组
    y = np.array(y_data)
    
    # 确保X是二维数组
    if isinstance(x_data[0], (int, float)):
        # 单个特征的情况
        X = np.array(x_data).reshape(-1, 1)
    else:
        X = np.array(x_data)
    
    # 验证数据维度一致性
    if len(y) != X.shape[0]:
        raise ValueError(f"因变量长度({len(y)})与自变量长度({X.shape[0]})不一致")
    
    n, k = X.shape
    
    # 处理工具变量
    if instruments is None:
        # 如果没有提供工具变量，则使用自变量作为工具变量（退化为OLS）
        Z = X.copy()
    else:
        # 确保工具变量是二维数组
        if isinstance(instruments[0], (int, float)):
            Z = np.array(instruments).reshape(-1, 1)
        else:
            Z = np.array(instruments)
        
        # 验证工具变量维度
        if len(Z) != len(y):
            raise ValueError(f"工具变量长度({len(Z)})与因变量长度({len(y)})不一致")
    
    # 添加常数项
    if constant:
        X = np.column_stack([np.ones(n), X])
        Z = np.column_stack([np.ones(n), Z])
        if feature_names:
            feature_names = ["const"] + feature_names
        else:
            feature_names = [f"const"] + [f"x{i}" for i in range(X.shape[1]-1)]
    else:
        if not feature_names:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    
    # 手动实现GMM估计
    try:
        # 初始化权重矩阵为单位矩阵
        W = np.eye(Z.shape[1])
        
        # 迭代估计直到收敛
        for iteration in range(100):  # 最大迭代次数
            # 一步GMM估计
            # X'Z W Z'X beta = X'Z W Z'y
            XZ = X.T @ Z
            ZY = Z.T @ y
            
            # 更稳定的矩阵运算
            left_side = XZ @ W @ XZ.T
            right_side = XZ @ W @ ZY
            
            # 解线性方程组
            try:
                beta = np.linalg.solve(left_side, right_side)
            except np.linalg.LinAlgError:
                # 如果矩阵奇异，使用伪逆
                beta = np.linalg.pinv(left_side) @ right_side
            
            # 计算残差
            residuals = y - X @ beta
            
            # 更新权重矩阵（基于残差的矩条件）
            moments = Z * residuals.reshape(-1, 1)
            S = moments.T @ moments / n  # 协方差矩阵
            
            # 在更新权重矩阵前进行有效性检查
            if np.isnan(S).any() or np.isinf(S).any():
                raise ValueError("矩条件协方差矩阵包含无效值")
            
            # 安全地更新权重矩阵
            W_new = _safe_inverse(S, reg_param=1e-8)  # 增加正则化参数以提高稳定性
            
            # 检查新权重矩阵的有效性
            if np.isnan(W_new).any() or np.isinf(W_new).any():
                raise ValueError("计算出的权重矩阵包含无效值")
                
            # 检查收敛性
            if np.allclose(W, W_new, rtol=1e-6, atol=1e-10):
                W = W_new
                break
            W = W_new
        
        # 计算最终的协方差矩阵和统计量
        residuals = y - X @ beta
        moments = Z * residuals.reshape(-1, 1)
        S = moments.T @ moments / n
        
        # 检查矩条件协方差矩阵
        if np.isnan(S).any() or np.isinf(S).any() or np.linalg.norm(S) == 0:
            raise ValueError("矩条件协方差矩阵无效")
        
        # 计算系数协方差矩阵
        # Var(beta) = (X'Z W Z'X)^(-1) X'Z W S W Z'X (X'Z W Z'X)^(-1)
        XZ = X.T @ Z
        
        # 检查XZ矩阵
        if np.isnan(XZ).any() or np.isinf(XZ).any():
            raise ValueError("X'Z矩阵包含无效值")
            
        left_side = XZ @ W @ XZ.T
        
        # 检查左侧矩阵
        if np.isnan(left_side).any() or np.isinf(left_side).any():
            raise ValueError("左侧矩阵(X'Z W Z'X)包含无效值")
        
        left_side_inv = _safe_inverse(left_side, reg_param=1e-8)  # 使用相同正则化参数
        
        # 检查逆矩阵
        if np.isnan(left_side_inv).any() or np.isinf(left_side_inv).any():
            raise ValueError("左侧矩阵的逆包含无效值")
        
        # 计算协方差矩阵
        cov_intermediate = XZ @ W @ S @ W @ XZ.T
        if np.isnan(cov_intermediate).any() or np.isinf(cov_intermediate).any():
            raise ValueError("中间协方差计算包含无效值")
            
        cov_beta = left_side_inv @ cov_intermediate @ left_side_inv
        std_errors = np.sqrt(np.diag(cov_beta))
        
        # 避免零标准误
        std_errors = np.maximum(std_errors, 1e-12)
        
        # 检查标准误
        if np.isnan(std_errors).any() or np.isinf(std_errors).any():
            raise ValueError("计算出的标准误包含无效值")
        
        # 计算t统计量和p值
        t_values = beta / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), n - len(beta)))
        
        # 计算置信区间
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n - len(beta))
        conf_int_lower = beta - t_critical * std_errors
        conf_int_upper = beta + t_critical * std_errors
        
        # J统计量（过度识别约束检验）
        if Z.shape[1] > len(beta):
            # 过度识别情况
            moment_conditions = Z.T @ residuals
            j_statistic = n * moment_conditions.T @ W @ moment_conditions
            j_df = Z.shape[1] - len(beta)
            j_p_value = 1 - stats.chi2.cdf(j_statistic, j_df)
        else:
            # 恰好识别情况
            j_statistic = 0.0
            j_p_value = 1.0
        
        return GMMResult(
            coefficients=beta.tolist(),
            std_errors=std_errors.tolist(),
            t_values=t_values.tolist(),
            p_values=p_values.tolist(),
            conf_int_lower=conf_int_lower.tolist(),
            conf_int_upper=conf_int_upper.tolist(),
            j_statistic=float(j_statistic),
            j_p_value=float(j_p_value),
            weight_matrix=W.tolist(),
            n_obs=n,
            n_moments=Z.shape[1],
            feature_names=feature_names
        )
    except Exception as e:
        # 如果GMM失败，抛出异常
        raise ValueError(f"GMM估计失败: {str(e)}")