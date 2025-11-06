"""
统计推断技术适配器
将核心算法适配为MCP工具
"""

from typing import List, Optional
import json

from econometrics.statistical_inference import (
    bootstrap_inference,
    permutation_test,
    BootstrapResult,
    PermutationTestResult
)

from .output_formatter import OutputFormatter


def bootstrap_adapter(
    data: List[float],
    statistic_func: str = "mean",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    method: str = "percentile",
    random_state: Optional[int] = None,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """Bootstrap推断适配器"""
    
    result: BootstrapResult = bootstrap_inference(
        data=data,
        statistic_func=statistic_func,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        method=method,
        random_state=random_state
    )
    
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# Bootstrap推断结果

{result.summary}
"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
        return formatted


def permutation_test_adapter(
    sample_a: List[float],
    sample_b: List[float],
    test_type: str = "mean_difference",
    alternative: str = "two-sided",
    n_permutations: int = 10000,
    random_state: Optional[int] = None,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """置换检验适配器"""
    
    result: PermutationTestResult = permutation_test(
        sample_a=sample_a,
        sample_b=sample_b,
        test_type=test_type,
        alternative=alternative,
        n_permutations=n_permutations,
        random_state=random_state
    )
    
    if output_format == "json":
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    else:
        formatted = f"""# 置换检验结果

{result.summary}
"""
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
        return formatted