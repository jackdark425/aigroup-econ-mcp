"""
面板数据模型适配器
将econometrics/specific_data_modeling/time_series_panel_data中的面板数据模型适配为MCP工具
"""

import json
import sys
from pathlib import Path

# 确保可以导入econometrics模块
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入面板数据模型
from econometrics.specific_data_modeling.time_series_panel_data.dynamic_panel_models import (
    DynamicPanelResult as CoreDynamicPanelResult,
)
from econometrics.specific_data_modeling.time_series_panel_data.dynamic_panel_models import (
    diff_gmm_model as core_diff_gmm_model,
)
from econometrics.specific_data_modeling.time_series_panel_data.dynamic_panel_models import (
    sys_gmm_model as core_sys_gmm_model,
)
from econometrics.specific_data_modeling.time_series_panel_data.panel_diagnostics import (
    PanelDiagnosticResult as CorePanelDiagnosticResult,
)
from econometrics.specific_data_modeling.time_series_panel_data.panel_diagnostics import (
    hausman_test as core_hausman_test,
)
from econometrics.specific_data_modeling.time_series_panel_data.panel_diagnostics import (
    lm_test as core_lm_test,
)
from econometrics.specific_data_modeling.time_series_panel_data.panel_diagnostics import (
    pooling_f_test as core_pooling_f_test,
)
from econometrics.specific_data_modeling.time_series_panel_data.panel_diagnostics import (
    within_correlation_test as core_within_correlation_test,
)
from econometrics.specific_data_modeling.time_series_panel_data.panel_var import (
    PanelVARResult as CorePanelVARResult,
)
from econometrics.specific_data_modeling.time_series_panel_data.panel_var import (
    panel_var_model as core_panel_var_model,
)

# 导入数据加载和格式化组件
from .data_loader import DataLoader
from .output_formatter import OutputFormatter


def dynamic_panel_adapter(
    y_data: list[float] | None = None,
    x_data: list[list[float]] | None = None,
    entity_ids: list[int] | None = None,
    time_periods: list[int] | None = None,
    file_path: str | None = None,
    model_type: str = "diff_gmm",
    lags: int = 1,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    动态面板模型适配器

    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        entity_ids: 个体标识符
        time_periods: 时间标识符
        file_path: 数据文件路径
        model_type: 模型类型 ("diff_gmm", "sys_gmm")
        lags: 滞后期数
        output_format: 输出格式
        save_path: 保存路径

    Returns:
        str: 格式化的分析结果
    """
    # 1. 数据准备
    if file_path:
        data_dict = DataLoader.load_from_file(file_path)
        y_data = data_dict["y_data"]
        x_data = data_dict["x_data"]
        entity_ids = data_dict["entity_ids"]
        time_periods = data_dict["time_periods"]
    elif y_data is None or x_data is None or entity_ids is None or time_periods is None:
        raise ValueError(
            "Must provide either file_path or (y_data, x_data, entity_ids, time_periods)"
        )

    # 2. 调用核心算法（使用改进的手动实现）
    try:
        result: CoreDynamicPanelResult = None
        if model_type == "diff_gmm":
            result = core_diff_gmm_model(y_data, x_data, entity_ids, time_periods, lags=lags)
        elif model_type == "sys_gmm":
            result = core_sys_gmm_model(y_data, x_data, entity_ids, time_periods, lags=lags)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    except Exception as e:
        # Return the canonical registry error payload so MCP clients can
        # parse all tool errors uniformly. Rich diagnostic hints go under
        # "details" rather than as top-level keys.
        error_payload = {
            "ok": False,
            "error": {
                "code": "dynamic_panel_fit_failed",
                "message": f"dynamic panel model ({model_type}) failed to fit",
                "details": {
                    "cause": str(e),
                    "suggestions": [
                        "inconsistent data shapes — check that all series have equal length",
                        "insufficient sample size — add observations or reduce lag order",
                        "multicollinearity — drop regressors or apply regularization",
                        "too few instruments — ensure GMM identification is satisfied",
                        "numerical instability — standardize inputs or enlarge the sample",
                    ],
                },
            },
        }
        return json.dumps(error_payload, ensure_ascii=False, indent=2)

    # 3. 格式化输出
    if output_format == "json":
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    else:
        try:
            formatted = OutputFormatter.format_dynamic_panel_result(result, output_format)

        except Exception as e:
            formatted = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)

            formatted = f"警告: {output_format}格式化失败({str(e)})，返回JSON格式\n\n{formatted}"
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
            return (
                f"动态面板模型({model_type})分析完成!\n\n{formatted}\n\n结果已保存到: {save_path}"
            )
        return formatted


def panel_diagnostics_adapter(
    test_type: str = "hausman",
    fe_coefficients: list[float] | None = None,
    re_coefficients: list[float] | None = None,
    fe_covariance: list[list[float]] | None = None,
    re_covariance: list[list[float]] | None = None,
    pooled_ssrs: float | None = None,
    fixed_ssrs: float | None = None,
    random_ssrs: float | None = None,
    n_individuals: int | None = None,
    n_params: int | None = None,
    n_obs: int | None = None,
    n_periods: int | None = None,
    residuals: list[list[float]] | None = None,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    面板数据诊断检验适配器

    Args:
        test_type: 检验类型 ("hausman", "pooling_f", "lm", "within_correlation")
        fe_coefficients: 固定效应模型系数 (Hausman)
        re_coefficients: 随机效应模型系数 (Hausman)
        fe_covariance: 固定效应模型协方差矩阵 (Hausman)
        re_covariance: 随机效应模型协方差矩阵 (Hausman)
        pooled_ssrs: 混合OLS模型残差平方和 (Pooling F, LM)
        fixed_ssrs: 固定效应模型残差平方和 (Pooling F)
        random_ssrs: 随机效应模型残差平方和 (LM)
        n_individuals: 个体数量
        n_params: 参数数量 (Pooling F)
        n_obs: 观测数量
        n_periods: 时间期数 (LM)
        residuals: 面板数据残差 (Within Correlation)
        output_format: 输出格式
        save_path: 保存路径

    Returns:
        str: 格式化的分析结果
    """
    # 调用核心算法
    result: CorePanelDiagnosticResult = None
    if test_type == "hausman":
        if not all([fe_coefficients, re_coefficients, fe_covariance, re_covariance]):
            raise ValueError(
                "Hausman test requires fe_coefficients, re_coefficients, fe_covariance, re_covariance"
            )
        result = core_hausman_test(fe_coefficients, re_coefficients, fe_covariance, re_covariance)
    elif test_type == "pooling_f":
        if not all(
            [pooled_ssrs is not None, fixed_ssrs is not None, n_individuals, n_params, n_obs]
        ):
            raise ValueError(
                "Pooling F test requires pooled_ssrs, fixed_ssrs, n_individuals, n_params, n_obs"
            )
        result = core_pooling_f_test(pooled_ssrs, fixed_ssrs, n_individuals, n_params, n_obs)
    elif test_type == "lm":
        if not all([pooled_ssrs is not None, random_ssrs is not None, n_individuals, n_periods]):
            raise ValueError("LM test requires pooled_ssrs, random_ssrs, n_individuals, n_periods")
        result = core_lm_test(pooled_ssrs, random_ssrs, n_individuals, n_periods)
    elif test_type == "within_correlation":
        if residuals is None:
            raise ValueError("Within correlation test requires residuals")
        result = core_within_correlation_test(residuals)
    else:
        raise ValueError(f"Unsupported test_type: {test_type}")

    # 格式化输出
    if output_format == "json":
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    else:
        try:
            formatted = OutputFormatter.format_panel_diagnostic_result(result, output_format)
        except Exception as e:
            formatted = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
            formatted = f"警告: {output_format}格式化失败({str(e)})，返回JSON格式\n\n{formatted}"
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
            return f"面板数据诊断({test_type})完成!\n\n{formatted}\n\n结果已保存到: {save_path}"
        return formatted


def panel_var_adapter(
    data: list[list[float]] | None = None,
    entity_ids: list[int] | None = None,
    time_periods: list[int] | None = None,
    file_path: str | None = None,
    lags: int = 1,
    variables: list[str] | None = None,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    面板VAR模型适配器

    Args:
        data: 多元面板数据
        entity_ids: 个体标识符
        time_periods: 时间标识符
        file_path: 数据文件路径
        lags: 滞后期数
        variables: 变量名称列表
        output_format: 输出格式
        save_path: 保存路径

    Returns:
        str: 格式化的分析结果
    """
    # 数据准备
    if file_path:
        data_dict = DataLoader.load_from_file(file_path)
        data = data_dict["data"]
        entity_ids = data_dict.get("entity_ids") or entity_ids
        time_periods = data_dict.get("time_periods") or time_periods
        variables = data_dict.get("variables") or variables
    elif data is None or entity_ids is None or time_periods is None:
        raise ValueError("Must provide either file_path or (data, entity_ids, time_periods)")

    # 调用核心算法
    result: CorePanelVARResult = core_panel_var_model(
        data, entity_ids, time_periods, lags, variables
    )

    # 格式化输出
    if output_format == "json":
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    else:
        try:
            formatted = OutputFormatter.format_panel_var_result(result, output_format)
        except Exception as e:
            formatted = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
            formatted = f"警告: {output_format}格式化失败({str(e)})，返回JSON格式\n\n{formatted}"
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
            return f"面板VAR模型分析完成!\n\n{formatted}\n\n结果已保存到: {save_path}"
        return formatted
