"""
非参数与半参数方法适配器
将核心算法适配为MCP工具
"""

from typing import List, Optional
import json

from econometrics.nonparametric import (
    kernel_regression,
    quantile_regression,
    spline_regression,
    gam_model,
    KernelRegressionResult,
    QuantileRegressionResult,
    SplineRegressionResult,
    GAMResult
)

from .data_loader import DataLoader
from .output_formatter import OutputFormatter


def kernel_regression_adapter(
    y_data: Optional[List[float]] = None,
    x_data: Optional[List[List[float]]] = None,
    file_path: Optional[str] = None,
    kernel_type: str = "gaussian",
    bandwidth: Optional[List[float]] = None,
    bandwidth_method: str = "cv_ls",
    variable_type: Optional[str] = None,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """核回归适配器"""
    
    # 数据准备
    if file_path:
        data = DataLoader.load_from_file(file_path)
        y_data = data["y_data"]
        x_data = data["x_data"]
    elif y_data is None or x_data is None:
        raise ValueError("必须提供文件路径(file_path)或直接数据(y_data和x_data)")
    
    # 调用核心算法
    result: KernelRegressionResult = kernel_regression(
        y_data=y_data,
        x_data=x_data,
        kernel_type=kernel_type,
        bandwidth=bandwidth,
        bandwidth_method=bandwidth_method,
        variable_type=variable_type
    )
    
    # 格式化输出
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# 核回归分析结果

{result.summary}

## 模型信息
- 核函数: {result.kernel_type}
- 带宽: {', '.join([f'{b:.4f}' for b in result.bandwidth])}
- R²: {result.r_squared:.4f}
"""
        if result.aic:
            formatted += f"- AIC: {result.aic:.2f}\n"
        
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
            return f"分析完成！\n\n{formatted}\n\n已保存到: {save_path}"
        return formatted


def quantile_regression_adapter(
    y_data: Optional[List[float]] = None,
    x_data: Optional[List[List[float]]] = None,
    file_path: Optional[str] = None,
    quantile: float = 0.5,
    feature_names: Optional[List[str]] = None,
    confidence_level: float = 0.95,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """分位数回归适配器"""
    
    # 数据准备
    if file_path:
        data = DataLoader.load_from_file(file_path)
        y_data = data["y_data"]
        x_data = data["x_data"]
        feature_names = data.get("feature_names") or feature_names
    elif y_data is None or x_data is None:
        raise ValueError("必须提供文件路径(file_path)或直接数据(y_data和x_data)")
    
    # 调用核心算法
    result: QuantileRegressionResult = quantile_regression(
        y_data=y_data,
        x_data=x_data,
        quantile=quantile,
        feature_names=feature_names,
        confidence_level=confidence_level
    )
    
    # 格式化输出
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# 分位数回归分析结果

{result.summary}
"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
            return f"分析完成！\n\n{formatted}\n\n已保存到: {save_path}"
        return formatted


def spline_regression_adapter(
    y_data: List[float],
    x_data: List[float],
    n_knots: int = 5,
    degree: int = 3,
    knots: str = "uniform",
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """样条回归适配器"""
    
    result: SplineRegressionResult = spline_regression(
        y_data=y_data,
        x_data=x_data,
        n_knots=n_knots,
        degree=degree,
        knots=knots
    )
    
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# 样条回归结果\n\n{result.summary}"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
        return formatted


def gam_adapter(
    y_data: List[float],
    x_data: List[List[float]],
    problem_type: str = "regression",
    n_splines: int = 10,
    lam: float = 0.6,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """GAM模型适配器"""
    
    result: GAMResult = gam_model(
        y_data=y_data,
        x_data=x_data,
        problem_type=problem_type,
        n_splines=n_splines,
        lam=lam
    )
    
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# GAM模型结果\n\n{result.summary}"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
        return formatted