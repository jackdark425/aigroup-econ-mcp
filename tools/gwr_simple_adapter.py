"""
简化的GWR适配器
避免复杂的类型转换问题
"""

from typing import List, Optional, Tuple
import json
from pathlib import Path

from econometrics.spatial_econometrics.gwr_simple import (
    geographically_weighted_regression_simple,
    GWRSimpleResult
)

from .output_formatter import OutputFormatter


def gwr_simple_adapter(
    y_data: List[float],
    x_data: List[List[float]],
    coordinates: List[Tuple[float, float]],
    feature_names: Optional[List[str]] = None,
    kernel_type: str = "gaussian",
    bandwidth: Optional[float] = None,
    fixed: bool = False,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """简化的地理加权回归适配器"""
    
    result: GWRSimpleResult = geographically_weighted_regression_simple(
        y_data=y_data,
        x_data=x_data,
        coordinates=coordinates,
        feature_names=feature_names,
        kernel_type=kernel_type,
        bandwidth=bandwidth,
        fixed=fixed
    )
    
    if output_format == "json":
        json_result = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# 简化的地理加权回归 (GWR) 结果

{result.summary}
"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
        return formatted