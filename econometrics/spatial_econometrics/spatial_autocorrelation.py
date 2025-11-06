"""
空间自相关检验
基于 esda (Exploratory Spatial Data Analysis) 库实现
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np

try:
    from esda import Moran, Moran_Local, Geary
    from libpysal.weights import W
    ESDA_AVAILABLE = True
except ImportError:
    ESDA_AVAILABLE = False
    Moran = None
    Moran_Local = None
    Geary = None
    W = None


class MoranIResult(BaseModel):
    """Moran's I 空间自相关检验结果"""
    moran_i: float = Field(..., description="Moran's I 统计量")
    expected_i: float = Field(..., description="期望值")
    variance_i: float = Field(..., description="方差")
    z_score: float = Field(..., description="Z统计量")
    p_value: float = Field(..., description="P值（双侧检验）")
    p_value_one_sided: float = Field(..., description="P值（单侧检验）")
    interpretation: str = Field(..., description="结果解释")
    n_observations: int = Field(..., description="观测数量")
    summary: str = Field(..., description="摘要信息")


class GearysCResult(BaseModel):
    """Geary's C 空间自相关检验结果"""
    geary_c: float = Field(..., description="Geary's C 统计量")
    expected_c: float = Field(..., description="期望值")
    variance_c: float = Field(..., description="方差")
    z_score: float = Field(..., description="Z统计量")
    p_value: float = Field(..., description="P值")
    interpretation: str = Field(..., description="结果解释")
    n_observations: int = Field(..., description="观测数量")
    summary: str = Field(..., description="摘要信息")


class LocalMoranResult(BaseModel):
    """局部Moran's I (LISA) 结果"""
    local_i: List[float] = Field(..., description="局部Moran's I值")
    z_scores: List[float] = Field(..., description="Z统计量")
    p_values: List[float] = Field(..., description="P值")
    quadrants: List[str] = Field(..., description="象限分类 (HH, LL, HL, LH)")
    significant_locations: List[int] = Field(..., description="显著位置索引")
    n_significant: int = Field(..., description="显著位置数量")
    summary: str = Field(..., description="摘要信息")


def morans_i_test(
    values: List[float],
    neighbors: dict,
    weights: Optional[dict] = None,
    permutations: int = 999,
    two_tailed: bool = True
) -> MoranIResult:
    """
    Moran's I 全局空间自相关检验
    
    Args:
        values: 观测值列表
        neighbors: 邻居字典 {i: [j1, j2, ...]}
        weights: 权重字典 {i: [w1, w2, ...]}，如果为None则使用均等权重
        permutations: 置换检验次数
        two_tailed: 是否双侧检验
        
    Returns:
        MoranIResult: Moran's I检验结果
        
    Raises:
        ImportError: esda库未安装
        ValueError: 输入数据无效
    """
    if not ESDA_AVAILABLE:
        raise ImportError(
            "esda库未安装。请运行: pip install esda\n"
            "或: pip install pysal"
        )
    
    # 输入验证
    if not values:
        raise ValueError("values不能为空")
    if not neighbors:
        raise ValueError("neighbors不能为空")
    
    n = len(values)
    y = np.array(values, dtype=np.float64)
    
    # 构建权重对象
    if weights is None:
        # 使用均等权重
        weights = {i: [1.0] * len(neighbors[i]) for i in neighbors}
    
    # 确保邻居字典的键是整数
    neighbors_int = {int(k): [int(n) for n in v] for k, v in neighbors.items()}
    weights_int = {int(k): v for k, v in weights.items()}
    
    w = W(neighbors_int, weights_int)
    
    # 执行Moran's I检验
    try:
        mi = Moran(y, w, permutations=permutations)
        
        # 提取结果
        moran_i = float(mi.I)
        expected_i = float(mi.EI)
        variance_i = float(mi.VI_norm)
        z_score = float(mi.z_norm)
        
        # P值
        if two_tailed:
            p_value = float(mi.p_norm)
            p_value_one_sided = float(mi.p_norm) / 2
        else:
            p_value_one_sided = float(mi.p_norm)
            p_value = float(mi.p_norm) * 2
        
        # 解释结果
        interpretation = _interpret_moran_i(moran_i, p_value, z_score)
        
        # 生成摘要
        summary = f"""Moran's I 空间自相关检验:
- Moran's I: {moran_i:.4f}
- 期望值: {expected_i:.4f}
- Z统计量: {z_score:.4f}
- P值: {p_value:.4f}
- 置换次数: {permutations}
- 结论: {interpretation}
"""
        
        return MoranIResult(
            moran_i=moran_i,
            expected_i=expected_i,
            variance_i=variance_i,
            z_score=z_score,
            p_value=p_value,
            p_value_one_sided=p_value_one_sided,
            interpretation=interpretation,
            n_observations=n,
            summary=summary
        )
    except Exception as e:
        raise ValueError(f"Moran's I检验失败: {str(e)}")


def gearys_c_test(
    values: List[float],
    neighbors: dict,
    weights: Optional[dict] = None,
    permutations: int = 999
) -> GearysCResult:
    """
    Geary's C 空间自相关检验
    
    Args:
        values: 观测值列表
        neighbors: 邻居字典
        weights: 权重字典
        permutations: 置换检验次数
        
    Returns:
        GearysCResult: Geary's C检验结果
    """
    if not ESDA_AVAILABLE:
        raise ImportError("esda库未安装")
    
    if not values or not neighbors:
        raise ValueError("输入数据不能为空")
    
    n = len(values)
    y = np.array(values, dtype=np.float64)
    
    # 构建权重对象
    if weights is None:
        weights = {i: [1.0] * len(neighbors[i]) for i in neighbors}
    
    # 确保邻居字典的键是整数
    neighbors_int = {int(k): [int(n) for n in v] for k, v in neighbors.items()}
    weights_int = {int(k): v for k, v in weights.items()}
    
    w = W(neighbors_int, weights_int)
    
    # 执行Geary's C检验
    try:
        gc = Geary(y, w, permutations=permutations)
        
        # 提取结果
        geary_c = float(gc.C)
        expected_c = float(gc.EC)
        variance_c = float(gc.VC_norm)
        z_score = float(gc.z_norm)
        p_value = float(gc.p_norm)
        
        # 解释结果
        interpretation = _interpret_geary_c(geary_c, p_value, z_score)
        
        # 生成摘要
        summary = f"""Geary's C 空间自相关检验:
- Geary's C: {geary_c:.4f}
- 期望值: {expected_c:.4f}
- Z统计量: {z_score:.4f}
- P值: {p_value:.4f}
- 结论: {interpretation}
"""
        
        return GearysCResult(
            geary_c=geary_c,
            expected_c=expected_c,
            variance_c=variance_c,
            z_score=z_score,
            p_value=p_value,
            interpretation=interpretation,
            n_observations=n,
            summary=summary
        )
    except Exception as e:
        raise ValueError(f"Geary's C检验失败: {str(e)}")


def local_morans_i(
    values: List[float],
    neighbors: dict,
    weights: Optional[dict] = None,
    permutations: int = 999,
    significance_level: float = 0.05
) -> LocalMoranResult:
    """
    局部Moran's I (LISA - Local Indicators of Spatial Association)
    
    Args:
        values: 观测值列表
        neighbors: 邻居字典
        weights: 权重字典
        permutations: 置换检验次数
        significance_level: 显著性水平
        
    Returns:
        LocalMoranResult: 局部Moran's I结果
    """
    if not ESDA_AVAILABLE:
        raise ImportError("esda库未安装")
    
    if not values or not neighbors:
        raise ValueError("输入数据不能为空")
    
    y = np.array(values, dtype=np.float64)
    
    # 构建权重对象
    if weights is None:
        weights = {i: [1.0] * len(neighbors[i]) for i in neighbors}
    
    # 确保邻居字典的键是整数
    neighbors_int = {int(k): [int(n) for n in v] for k, v in neighbors.items()}
    weights_int = {int(k): v for k, v in weights.items()}
    
    w = W(neighbors_int, weights_int)
    
    # 执行局部Moran's I分析
    lm = Moran_Local(y, w, permutations=permutations)
    
    # 提取结果
    local_i = lm.Is.tolist()
    z_scores = lm.z_sim.tolist()
    p_values = lm.p_sim.tolist()
    
    # 象限分类
    quadrants = []
    for q in lm.q:
        if q == 1:
            quadrants.append("HH")  # High-High
        elif q == 2:
            quadrants.append("LH")  # Low-High
        elif q == 3:
            quadrants.append("LL")  # Low-Low
        elif q == 4:
            quadrants.append("HL")  # High-Low
        else:
            quadrants.append("NS")  # Not Significant
    
    # 识别显著位置
    significant_locations = [
        i for i, p in enumerate(p_values) 
        if p < significance_level
    ]
    n_significant = len(significant_locations)
    
    # 生成摘要
    summary = f"""局部Moran's I (LISA) 分析:
- 观测数量: {len(values)}
- 显著位置数: {n_significant} ({n_significant/len(values)*100:.1f}%)
- 显著性水平: {significance_level}
- HH聚类: {quadrants.count('HH')} 个
- LL聚类: {quadrants.count('LL')} 个
- HL离群: {quadrants.count('HL')} 个
- LH离群: {quadrants.count('LH')} 个
"""
    
    return LocalMoranResult(
        local_i=local_i,
        z_scores=z_scores,
        p_values=p_values,
        quadrants=quadrants,
        significant_locations=significant_locations,
        n_significant=n_significant,
        summary=summary
    )


def _interpret_moran_i(moran_i: float, p_value: float, z_score: float) -> str:
    """解释Moran's I结果"""
    if p_value < 0.01:
        sig_level = "高度显著"
    elif p_value < 0.05:
        sig_level = "显著"
    elif p_value < 0.10:
        sig_level = "边际显著"
    else:
        sig_level = "不显著"
    
    if moran_i > 0:
        pattern = "正空间自相关（空间聚集）"
    elif moran_i < 0:
        pattern = "负空间自相关（空间离散）"
    else:
        pattern = "无空间自相关（随机分布）"
    
    return f"{sig_level}的{pattern}"


def _interpret_geary_c(geary_c: float, p_value: float, z_score: float) -> str:
    """解释Geary's C结果"""
    if p_value < 0.01:
        sig_level = "高度显著"
    elif p_value < 0.05:
        sig_level = "显著"
    elif p_value < 0.10:
        sig_level = "边际显著"
    else:
        sig_level = "不显著"
    
    if geary_c < 1:
        pattern = "正空间自相关"
    elif geary_c > 1:
        pattern = "负空间自相关"
    else:
        pattern = "无空间自相关"
    
    return f"{sig_level}的{pattern}"