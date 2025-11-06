"""
生存分析适配器 - 简化版本
使用完全简化的生存分析模块，避免lifelines依赖
"""

from typing import List, Optional
import json

from econometrics.survival_analysis.survival_models import (
    cox_regression_simple,
    CoxRegressionResult
)

from .output_formatter import OutputFormatter


def cox_regression_adapter_simple(
    durations: List[float],
    event_observed: List[int],
    covariates: List[List[float]],
    feature_names: Optional[List[str]] = None,
    confidence_level: float = 0.95,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """Cox回归适配器 - 简化版本"""
    
    result: CoxRegressionResult = cox_regression_simple(
        durations=durations,
        event_observed=event_observed,
        covariates=covariates,
        feature_names=feature_names,
        confidence_level=confidence_level
    )
    
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# Cox比例风险模型\n\n{result.summary}"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
        return formatted