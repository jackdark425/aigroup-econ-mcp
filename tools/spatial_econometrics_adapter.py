"""
空间计量经济学适配器
将核心算法适配为MCP工具
"""

from typing import List, Optional, Union, Dict, Tuple
import json
from pathlib import Path

from econometrics.spatial_econometrics import (
    create_spatial_weights,
    morans_i_test,
    gearys_c_test,
    local_morans_i,
    spatial_lag_model,
    spatial_error_model,
    spatial_durbin_model,
    geographically_weighted_regression,
    SpatialWeightsResult,
    MoranIResult,
    GearysCResult,
    LocalMoranResult,
    SpatialRegressionResult,
    SpatialDurbinResult,
    GWRResult
)

from .output_formatter import OutputFormatter


def spatial_weights_adapter(
    coordinates: Optional[List[Tuple[float, float]]] = None,
    adjacency_matrix: Optional[List[List[int]]] = None,
    weight_type: str = "queen",
    k: int = 4,
    distance_threshold: Optional[float] = None,
    bandwidth: Optional[float] = None,
    kernel_type: str = "triangular",
    row_standardize: bool = True,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """空间权重矩阵适配器"""
    
    # 调用核心算法
    result: SpatialWeightsResult = create_spatial_weights(
        coordinates=coordinates,
        adjacency_matrix=adjacency_matrix,
        weight_type=weight_type,
        k=k,
        distance_threshold=distance_threshold,
        bandwidth=bandwidth,
        kernel_type=kernel_type,
        row_standardize=row_standardize
    )
    
    # 格式化输出
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        # Markdown格式
        formatted = f"""# 空间权重矩阵分析结果

{result.summary}

## 详细信息
- 观测数量: {result.n_observations}
- 权重类型: {result.weight_type}
- 平均邻居数: {result.n_neighbors_mean:.2f}
- 邻居数范围: [{result.n_neighbors_min}, {result.n_neighbors_max}]
- 非零权重: {result.pct_nonzero:.2f}%
- 是否对称: {'是' if result.is_symmetric else '否'}
"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{formatted}"
        return formatted


def morans_i_adapter(
    values: List[float],
    neighbors: dict,
    weights: Optional[dict] = None,
    permutations: int = 999,
    two_tailed: bool = True,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """Moran's I检验适配器"""
    
    # 调用核心算法
    result: MoranIResult = morans_i_test(
        values=values,
        neighbors=neighbors,
        weights=weights,
        permutations=permutations,
        two_tailed=two_tailed
    )
    
    # 格式化输出
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# Moran's I 空间自相关检验结果

{result.summary}

## 统计量
- Moran's I: {result.moran_i:.4f}
- 期望值: {result.expected_i:.4f}
- Z统计量: {result.z_score:.4f}
- P值: {result.p_value:.4f}

## 解释
{result.interpretation}
"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
            return f"分析完成！\n\n{formatted}\n\n已保存到: {save_path}"
        return formatted


def gearys_c_adapter(
    values: List[float],
    neighbors: dict,
    weights: Optional[dict] = None,
    permutations: int = 999,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """Geary's C检验适配器"""
    
    result: GearysCResult = gearys_c_test(
        values=values,
        neighbors=neighbors,
        weights=weights,
        permutations=permutations
    )
    
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# Geary's C 空间自相关检验结果

{result.summary}
"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
        return formatted


def local_moran_adapter(
    values: List[float],
    neighbors: dict,
    weights: Optional[dict] = None,
    permutations: int = 999,
    significance_level: float = 0.05,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """局部Moran's I (LISA) 适配器"""
    
    result: LocalMoranResult = local_morans_i(
        values=values,
        neighbors=neighbors,
        weights=weights,
        permutations=permutations,
        significance_level=significance_level
    )
    
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# 局部Moran's I (LISA) 分析结果

{result.summary}
"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
        return formatted


def spatial_regression_adapter(
    y_data: List[float],
    x_data: List[List[float]],
    neighbors: dict,
    weights: Optional[dict] = None,
    feature_names: Optional[List[str]] = None,
    model_type: str = "sar",
    method: str = "ml",
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """空间回归模型适配器"""
    
    # 调用核心算法
    if model_type.lower() == "sar":
        result: SpatialRegressionResult = spatial_lag_model(
            y_data=y_data,
            x_data=x_data,
            neighbors=neighbors,
            weights=weights,
            feature_names=feature_names,
            method=method
        )
    elif model_type.lower() == "sem":
        result: SpatialRegressionResult = spatial_error_model(
            y_data=y_data,
            x_data=x_data,
            neighbors=neighbors,
            weights=weights,
            feature_names=feature_names,
            method=method
        )
    elif model_type.lower() == "sdm":
        result: SpatialDurbinResult = spatial_durbin_model(
            y_data=y_data,
            x_data=x_data,
            neighbors=neighbors,
            weights=weights,
            feature_names=feature_names
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 格式化输出
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# {result.model_type if hasattr(result, 'model_type') else 'SDM'} 空间回归模型结果

{result.summary}

## 系数估计
"""
        # 确保所有结果都是列表类型
        feature_names = list(result.feature_names) if hasattr(result.feature_names, '__iter__') else []
        coefficients = list(result.coefficients) if hasattr(result.coefficients, '__iter__') else []
        std_errors = list(result.std_errors) if hasattr(result.std_errors, '__iter__') else []
        z_scores = list(result.z_scores) if hasattr(result.z_scores, '__iter__') else []
        p_values = list(result.p_values) if hasattr(result.p_values, '__iter__') else []
        
        # 使用最短的长度来避免索引错误
        min_len = min(len(feature_names), len(coefficients), len(std_errors), len(z_scores), len(p_values))
        
        for i in range(min_len):
            name = feature_names[i]
            coef = coefficients[i]
            se = std_errors[i]
            z = z_scores[i]
            p = p_values[i]
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            formatted += f"- {name}: {coef:.4f} (SE: {se:.4f}, Z={z:.2f}, p={p:.4f}){sig}\n"
        
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
            return f"分析完成！\n\n{formatted}\n\n已保存到: {save_path}"
        return formatted


def gwr_adapter(
    y_data: List[float],
    x_data: List[List[float]],
    coordinates: List[Tuple[float, float]],
    feature_names: Optional[List[str]] = None,
    kernel_type: str = "bisquare",
    bandwidth: Optional[float] = None,
    fixed: bool = False,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """地理加权回归适配器"""
    
    result: GWRResult = geographically_weighted_regression(
        y_data=y_data,
        x_data=x_data,
        coordinates=coordinates,
        feature_names=feature_names,
        kernel_type=kernel_type,
        bandwidth=bandwidth,
        fixed=fixed
    )
    
    if output_format == "json":
        # 使用model_dump替代弃用的dict方法
        json_result = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# 地理加权回归 (GWR) 结果

{result.summary}
"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
        return formatted