"""
生存分析适配器 - 简化版本
使用完全简化的生存分析模块，避免lifelines依赖
"""

import json

from econometrics.survival_analysis.survival_models import (
    CoxRegressionResult,
    cox_regression_simple,
)

from .output_formatter import OutputFormatter


def cox_regression_adapter_simple(
    durations: list[float],
    event_observed: list[int],
    covariates: list[list[float]],
    feature_names: list[str] | None = None,
    confidence_level: float = 0.95,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """Cox回归适配器 - 简化版本"""

    result: CoxRegressionResult = cox_regression_simple(
        durations=durations,
        event_observed=event_observed,
        covariates=covariates,
        feature_names=feature_names,
        confidence_level=confidence_level,
    )

    if output_format == "json":
        json_result = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# Cox比例风险模型\n\n{result.summary}"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
        return formatted
