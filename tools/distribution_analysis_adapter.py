"""
分布分析与分解方法适配器
将核心算法适配为MCP工具
"""

from typing import List, Optional
import json

from econometrics.distribution_analysis import (
    oaxaca_blinder_decomposition,
    variance_decomposition,
    time_series_decomposition,
    OaxacaResult,
    VarianceDecompositionResult,
    TimeSeriesDecompositionResult
)

from .output_formatter import OutputFormatter


def oaxaca_blinder_adapter(
    y_a: List[float],
    x_a: List[List[float]],
    y_b: List[float],
    x_b: List[List[float]],
    feature_names: Optional[List[str]] = None,
    weight_matrix: str = "pooled",
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """Oaxaca-Blinder分解适配器"""
    
    result: OaxacaResult = oaxaca_blinder_decomposition(
        y_a=y_a,
        x_a=x_a,
        y_b=y_b,
        x_b=x_b,
        feature_names=feature_names,
        weight_matrix=weight_matrix
    )
    
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# Oaxaca-Blinder分解结果

{result.summary}
"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
        return formatted


def variance_decomposition_adapter(
    values: List[float],
    groups: List[str],
    group_names: Optional[List[str]] = None,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """方差分解适配器"""
    
    result: VarianceDecompositionResult = variance_decomposition(
        values=values,
        groups=groups,
        group_names=group_names
    )
    
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# 方差分解(ANOVA)结果

{result.summary}
"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
        return formatted


def time_series_decomposition_adapter(
    data: List[float],
    period: int = 12,
    model: str = "additive",
    method: str = "classical",
    extrapolate_trend: str = "freq",
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """时间序列分解适配器"""
    
    result: TimeSeriesDecompositionResult = time_series_decomposition(
        data=data,
        period=period,
        model=model,
        method=method,
        extrapolate_trend=extrapolate_trend
    )
    
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# 时间序列分解结果

{result.summary}
"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
        return formatted