"""
空间权重矩阵构建
基于 libpysal 库实现多种空间权重矩阵构建方法
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np

try:
    from libpysal.weights import Queen, Rook, KNN, DistanceBand, Kernel
    from libpysal.weights import W
    LIBPYSAL_AVAILABLE = True
except ImportError:
    LIBPYSAL_AVAILABLE = False
    W = None


class SpatialWeightsResult(BaseModel):
    """空间权重矩阵结果"""
    n_observations: int = Field(..., description="观测数量")
    weight_type: str = Field(..., description="权重类型")
    n_neighbors_mean: float = Field(..., description="平均邻居数")
    n_neighbors_min: int = Field(..., description="最小邻居数")
    n_neighbors_max: int = Field(..., description="最大邻居数")
    pct_nonzero: float = Field(..., description="非零权重百分比")
    weights_matrix: List[List[float]] = Field(..., description="权重矩阵（稀疏表示）")
    neighbors: Dict[int, List[int]] = Field(..., description="邻居字典")
    is_symmetric: bool = Field(..., description="是否对称")
    summary: str = Field(..., description="摘要信息")


def create_spatial_weights(
    coordinates: Optional[List[Tuple[float, float]]] = None,
    adjacency_matrix: Optional[List[List[int]]] = None,
    weight_type: str = "queen",
    k: int = 4,
    distance_threshold: Optional[float] = None,
    bandwidth: Optional[float] = None,
    kernel_type: str = "triangular",
    row_standardize: bool = True
) -> SpatialWeightsResult:
    """
    创建空间权重矩阵
    
    Args:
        coordinates: 坐标列表 [(x1,y1), (x2,y2), ...]
        adjacency_matrix: 邻接矩阵（用于基于邻接的权重）
        weight_type: 权重类型 - "queen"(皇后邻接), "rook"(车邻接), 
                     "knn"(K近邻), "distance"(距离带), "kernel"(核权重)
        k: K近邻中的邻居数量
        distance_threshold: 距离带阈值
        bandwidth: 核权重带宽
        kernel_type: 核函数类型
        row_standardize: 是否进行行标准化
        
    Returns:
        SpatialWeightsResult: 空间权重矩阵结果
        
    Raises:
        ImportError: libpysal库未安装
        ValueError: 输入数据无效
    """
    if not LIBPYSAL_AVAILABLE:
        raise ImportError(
            "libpysal库未安装。请运行: pip install libpysal\n"
            "或: pip install pysal"
        )
    
    # 输入验证
    if coordinates is None and adjacency_matrix is None:
        raise ValueError("必须提供coordinates或adjacency_matrix之一")
    
    # 构建空间权重对象
    w = None
    
    if weight_type == "queen":
        if adjacency_matrix is not None:
            # 基于邻接矩阵构建
            w = _create_from_adjacency(adjacency_matrix, "queen")
        else:
            raise ValueError("Queen邻接需要提供adjacency_matrix")
            
    elif weight_type == "rook":
        if adjacency_matrix is not None:
            w = _create_from_adjacency(adjacency_matrix, "rook")
        else:
            raise ValueError("Rook邻接需要提供adjacency_matrix")
            
    elif weight_type == "knn":
        if coordinates is None:
            raise ValueError("KNN需要提供coordinates")
        coords_array = np.array(coordinates)
        w = KNN.from_array(coords_array, k=k)
        
    elif weight_type == "distance":
        if coordinates is None:
            raise ValueError("距离带需要提供coordinates")
        if distance_threshold is None:
            raise ValueError("距离带需要提供distance_threshold")
        coords_array = np.array(coordinates)
        w = DistanceBand.from_array(coords_array, threshold=distance_threshold)
        
    elif weight_type == "kernel":
        if coordinates is None:
            raise ValueError("核权重需要提供coordinates")
        coords_array = np.array(coordinates)
        if bandwidth is None:
            # 使用默认带宽
            bandwidth = "auto"
        w = Kernel.from_array(coords_array, bandwidth=bandwidth, function=kernel_type)
        
    else:
        raise ValueError(
            f"不支持的权重类型: {weight_type}。"
            f"支持的类型: queen, rook, knn, distance, kernel"
        )
    
    # 行标准化
    if row_standardize and weight_type != "kernel":  # 核权重通常已经标准化
        w.transform = 'r'
    
    # 提取结果
    n = w.n
    
    # 邻居统计
    cardinalities = w.cardinalities
    n_neighbors_mean = float(np.mean(list(cardinalities.values())))
    n_neighbors_min = int(min(cardinalities.values()))
    n_neighbors_max = int(max(cardinalities.values()))
    
    # 非零权重百分比
    total_possible = n * n
    pct_nonzero = float(w.pct_nonzero)
    
    # 转换为稀疏矩阵表示（字典格式）
    weights_matrix = _convert_to_sparse_matrix(w)
    
    # 邻居字典
    neighbors = {int(i): [int(j) for j in w.neighbors[i]] for i in w.neighbors}
    
    # 检查对称性
    is_symmetric = _check_symmetry(w)
    
    # 生成摘要
    summary = f"""空间权重矩阵摘要:
- 观测数量: {n}
- 权重类型: {weight_type}
- 平均邻居数: {n_neighbors_mean:.2f}
- 邻居数范围: [{n_neighbors_min}, {n_neighbors_max}]
- 非零权重: {pct_nonzero:.2f}%
- 是否对称: {'是' if is_symmetric else '否'}
- 是否行标准化: {'是' if row_standardize else '否'}
"""
    
    return SpatialWeightsResult(
        n_observations=n,
        weight_type=weight_type,
        n_neighbors_mean=n_neighbors_mean,
        n_neighbors_min=n_neighbors_min,
        n_neighbors_max=n_neighbors_max,
        pct_nonzero=pct_nonzero,
        weights_matrix=weights_matrix,
        neighbors=neighbors,
        is_symmetric=is_symmetric,
        summary=summary
    )


def _create_from_adjacency(adjacency_matrix: List[List[int]], contiguity_type: str) -> W:
    """从邻接矩阵创建空间权重"""
    n = len(adjacency_matrix)
    
    # 转换为邻居字典
    neighbors = {}
    weights = {}
    
    for i in range(n):
        neighbors[i] = []
        weights[i] = []
        for j in range(n):
            if i != j and adjacency_matrix[i][j] > 0:
                neighbors[i].append(j)
                weights[i].append(float(adjacency_matrix[i][j]))
    
    # 创建权重对象
    w = W(neighbors, weights)
    return w


def _convert_to_sparse_matrix(w: W) -> List[List[float]]:
    """将权重对象转换为稀疏矩阵表示（用于返回）"""
    # 返回前100个非零元素（避免过大）
    sparse_repr = []
    count = 0
    max_elements = 100
    
    for i in w.neighbors:
        for j_idx, j in enumerate(w.neighbors[i]):
            if count >= max_elements:
                break
            weight = w.weights[i][j_idx]
            sparse_repr.append([int(i), int(j), float(weight)])
            count += 1
        if count >= max_elements:
            break
    
    return sparse_repr


def _check_symmetry(w: W) -> bool:
    """检查权重矩阵是否对称"""
    try:
        # 简单检查：对于每个i->j，是否存在j->i
        for i in w.neighbors:
            for j_idx, j in enumerate(w.neighbors[i]):
                # 检查j的邻居中是否有i
                if i not in w.neighbors.get(j, []):
                    return False
                # 检查权重是否相同
                j_i_idx = w.neighbors[j].index(i)
                if abs(w.weights[i][j_idx] - w.weights[j][j_i_idx]) > 1e-10:
                    return False
        return True
    except:
        return False