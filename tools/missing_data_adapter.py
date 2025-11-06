"""
缺失数据处理适配器
"""

from typing import List, Optional
import json

from econometrics.missing_data import (
    simple_imputation,
    multiple_imputation,
    SimpleImputationResult,
    MultipleImputationResult
)

from .output_formatter import OutputFormatter


def simple_imputation_adapter(
    data: List[List[float]],
    strategy: str = "mean",
    fill_value: Optional[float] = None,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """简单插补适配器"""
    
    result: SimpleImputationResult = simple_imputation(
        data=data,
        strategy=strategy,
        fill_value=fill_value
    )
    
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# 简单插补结果\n\n{result.summary}"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
        return formatted


def multiple_imputation_adapter(
    data: List[List[float]],
    n_imputations: int = 5,
    max_iter: int = 10,
    random_state: Optional[int] = None,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """多重插补适配器"""
    
    result: MultipleImputationResult = multiple_imputation(
        data=data,
        n_imputations=n_imputations,
        max_iter=max_iter,
        random_state=random_state
    )
    
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# 多重插补结果\n\n{result.summary}"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
        return formatted