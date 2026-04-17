"""
时间序列模型适配器
将econometrics/specific_data_modeling/time_series_panel_data中的时间序列模型适配为MCP工具
"""

import json
import sys
from pathlib import Path

# 确保可以导入econometrics模块
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入时间序列模型
from econometrics.specific_data_modeling.time_series_panel_data.arima_model import (
    ARIMAResult as CoreARIMAResult,
)
from econometrics.specific_data_modeling.time_series_panel_data.arima_model import (
    arima_model as core_arima_model,
)
from econometrics.specific_data_modeling.time_series_panel_data.cointegration_vecm import (
    CointegrationResult as CoreCointegrationResult,
)
from econometrics.specific_data_modeling.time_series_panel_data.cointegration_vecm import (
    VECMResult as CoreVECMResult,
)
from econometrics.specific_data_modeling.time_series_panel_data.cointegration_vecm import (
    engle_granger_cointegration_test as core_engle_granger_cointegration_test,
)
from econometrics.specific_data_modeling.time_series_panel_data.cointegration_vecm import (
    johansen_cointegration_test as core_johansen_cointegration_test,
)
from econometrics.specific_data_modeling.time_series_panel_data.cointegration_vecm import (
    vecm_model as core_vecm_model,
)
from econometrics.specific_data_modeling.time_series_panel_data.exponential_smoothing import (
    ExponentialSmoothingResult as CoreExponentialSmoothingResult,
)
from econometrics.specific_data_modeling.time_series_panel_data.exponential_smoothing import (
    exponential_smoothing_model as core_exponential_smoothing_model,
)
from econometrics.specific_data_modeling.time_series_panel_data.garch_model import (
    GARCHResult as CoreGARCHResult,
)
from econometrics.specific_data_modeling.time_series_panel_data.garch_model import (
    garch_model as core_garch_model,
)
from econometrics.specific_data_modeling.time_series_panel_data.structural_break_tests import (
    StructuralBreakResult as CoreStructuralBreakResult,
)
from econometrics.specific_data_modeling.time_series_panel_data.structural_break_tests import (
    bai_perron_test as core_bai_perron_test,
)
from econometrics.specific_data_modeling.time_series_panel_data.structural_break_tests import (
    chow_test as core_chow_test,
)
from econometrics.specific_data_modeling.time_series_panel_data.structural_break_tests import (
    quandt_andrews_test as core_quandt_andrews_test,
)
from econometrics.specific_data_modeling.time_series_panel_data.time_varying_parameter_models import (
    TimeVaryingParameterResult as CoreTimeVaryingParameterResult,
)
from econometrics.specific_data_modeling.time_series_panel_data.time_varying_parameter_models import (
    markov_switching_model as core_markov_switching_model,
)
from econometrics.specific_data_modeling.time_series_panel_data.time_varying_parameter_models import (
    star_model as core_star_model,
)
from econometrics.specific_data_modeling.time_series_panel_data.time_varying_parameter_models import (
    tar_model as core_tar_model,
)
from econometrics.specific_data_modeling.time_series_panel_data.unit_root_tests import (
    UnitRootTestResult as CoreUnitRootTestResult,
)
from econometrics.specific_data_modeling.time_series_panel_data.unit_root_tests import (
    adf_test as core_adf_test,
)
from econometrics.specific_data_modeling.time_series_panel_data.unit_root_tests import (
    kpss_test as core_kpss_test,
)
from econometrics.specific_data_modeling.time_series_panel_data.unit_root_tests import (
    pp_test as core_pp_test,
)
from econometrics.specific_data_modeling.time_series_panel_data.var_svar_model import (
    VARResult as CoreVARResult,
)
from econometrics.specific_data_modeling.time_series_panel_data.var_svar_model import (
    svar_model as core_svar_model,
)
from econometrics.specific_data_modeling.time_series_panel_data.var_svar_model import (
    var_model as core_var_model,
)

# 导入数据加载和格式化组件
from .data_loader import DataLoader
from .output_formatter import OutputFormatter


def arima_adapter(
    data: list[float] | None = None,
    file_path: str | None = None,
    order: tuple = (1, 1, 1),
    forecast_steps: int = 1,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    ARIMA模型适配器

    Args:
        data: 时间序列数据
        file_path: 数据文件路径
        order: (p,d,q) 参数设置
        forecast_steps: 预测步数
        output_format: 输出格式 ("json", "markdown", "html")
        save_path: 保存路径

    Returns:
        str: 格式化的分析结果
    """
    # 1. 数据准备
    if file_path:
        data_dict = DataLoader.load_from_file(file_path)
        data = data_dict["data"]
    elif data is None:
        raise ValueError("Must provide either file_path or data")

    # 2. 调用核心算法
    result: CoreARIMAResult = core_arima_model(
        data=data, order=order, forecast_steps=forecast_steps
    )

    # 3. 格式化输出
    if output_format == "json":
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    else:
        try:
            formatted = OutputFormatter.format_arima_result(result, output_format)

        except Exception as e:
            formatted = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)

            formatted = f"警告: {output_format}格式化失败({str(e)})，返回JSON格式\n\n{formatted}"
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
            return f"ARIMA分析完成!\n\n{formatted}\n\n结果已保存到: {save_path}"
        return formatted


def exp_smoothing_adapter(
    data: list[float] | None = None,
    file_path: str | None = None,
    trend: bool = True,
    seasonal: bool = False,
    seasonal_periods: int | None = None,
    forecast_steps: int = 1,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    指数平滑模型适配器

    Args:
        data: 时间序列数据
        file_path: 数据文件路径
        trend: 是否包含趋势成分
        seasonal: 是否包含季节成分
        seasonal_periods: 季节周期长度
        forecast_steps: 预测步数
        output_format: 输出格式
        save_path: 保存路径

    Returns:
        str: 格式化的分析结果
    """
    # 1. 数据准备
    if file_path:
        data_dict = DataLoader.load_from_file(file_path)
        data = data_dict["data"]
    elif data is None:
        raise ValueError("Must provide either file_path or data")

    # 2. 调用核心算法
    result: CoreExponentialSmoothingResult = core_exponential_smoothing_model(
        data=data,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        forecast_steps=forecast_steps,
    )

    # 3. 格式化输出
    if output_format == "json":
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    else:
        try:
            formatted = OutputFormatter.format_exponential_smoothing_result(result, output_format)

        except Exception as e:
            formatted = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)

            formatted = f"警告: {output_format}格式化失败({str(e)})，返回JSON格式\n\n{formatted}"
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
            return f"指数平滑分析完成!\n\n{formatted}\n\n结果已保存到: {save_path}"
        return formatted


def garch_adapter(
    data: list[float] | None = None,
    file_path: str | None = None,
    order: tuple = (1, 1),
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    GARCH模型适配器

    Args:
        data: 时间序列数据
        file_path: 数据文件路径
        order: (p, q) 参数设置
        output_format: 输出格式
        save_path: 保存路径

    Returns:
        str: 格式化的分析结果
    """
    # 1. 数据准备
    if file_path:
        data_dict = DataLoader.load_from_file(file_path)
        data = data_dict["data"]
    elif data is None:
        raise ValueError("Must provide either file_path or data")

    # 2. 调用核心算法
    result: CoreGARCHResult = core_garch_model(data=data, order=order)

    # 3. 格式化输出
    if output_format == "json":
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    else:
        try:
            formatted = OutputFormatter.format_garch_result(result, output_format)

        except Exception as e:
            formatted = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)

            formatted = f"警告: {output_format}格式化失败({str(e)})，返回JSON格式\n\n{formatted}"
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
            return f"GARCH分析完成!\n\n{formatted}\n\n结果已保存到: {save_path}"
        return formatted


def unit_root_adapter(
    data: list[float] | None = None,
    file_path: str | None = None,
    test_type: str = "adf",
    max_lags: int | None = None,
    regression_type: str = "c",
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    单位根检验适配器

    Args:
        data: 时间序列数据
        file_path: 数据文件路径
        test_type: 检验类型 ("adf", "pp", "kpss")
        max_lags: 最大滞后阶数 (仅ADF检验)
        regression_type: 回归类型 ("c"=常数, "ct"=常数和趋势, "nc"=无常数)
        output_format: 输出格式
        save_path: 保存路径

    Returns:
        str: 格式化的分析结果
    """
    # 1. 数据准备
    if file_path:
        data_dict = DataLoader.load_from_file(file_path)
        data = data_dict["data"]
    elif data is None:
        raise ValueError("Must provide either file_path or data")

    # 2. 调用核心算法
    result: CoreUnitRootTestResult = None
    if test_type == "adf":
        result = core_adf_test(data, max_lags=max_lags, regression_type=regression_type)
    elif test_type == "pp":
        result = core_pp_test(data, regression_type=regression_type)
    elif test_type == "kpss":
        result = core_kpss_test(data, regression_type=regression_type)
    else:
        raise ValueError(f"Unsupported test_type: {test_type}")

    # 3. 格式化输出
    if output_format == "json":
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    else:
        try:
            formatted = OutputFormatter.format_unit_root_test_result(result, output_format)

        except Exception as e:
            formatted = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)

            formatted = f"警告: {output_format}格式化失败({str(e)})，返回JSON格式\n\n{formatted}"
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
            return (
                f"单位根检验({test_type.upper()})完成!\n\n{formatted}\n\n结果已保存到: {save_path}"
            )
        return formatted


def var_svar_adapter(
    data: list[list[float]] | None = None,
    file_path: str | None = None,
    model_type: str = "var",
    lags: int = 1,
    variables: list[str] | None = None,
    a_matrix: list[list[float]] | None = None,
    b_matrix: list[list[float]] | None = None,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    VAR/SVAR模型适配器

    Args:
        data: 多元时间序列数据 (格式: 每个子列表代表一个时间点的所有变量值)
        file_path: 数据文件路径
        model_type: 模型类型 ("var", "svar")
        lags: 滞后期数
        variables: 变量名称列表
        a_matrix: A约束矩阵 (仅SVAR)
        b_matrix: B约束矩阵 (仅SVAR)
        output_format: 输出格式
        save_path: 保存路径

    Returns:
        str: 格式化的分析结果
    """
    # 1. 数据准备
    if file_path:
        data_dict = DataLoader.load_from_file(file_path)
        data = data_dict["data"]
        variables = data_dict.get("variables") or variables
    elif data is None:
        raise ValueError("Must provide either file_path or data")

    # 2. 数据格式转换：从时间点格式转换为变量格式
    # 输入格式: [[var1_t1, var2_t1], [var1_t2, var2_t2], ...]
    # 需要转换为: [[var1_t1, var1_t2, ...], [var2_t1, var2_t2, ...]]
    if data and len(data) > 0:
        n_vars = len(data[0])
        n_obs = len(data)

        # 转换数据格式
        var_data = []
        for var_idx in range(n_vars):
            var_series = [data[t][var_idx] for t in range(n_obs)]
            var_data.append(var_series)

        # 如果没有提供变量名，自动生成
        if variables is None:
            variables = [f"Variable_{i}" for i in range(n_vars)]

        # 检查变量数量是否与数据一致
        if len(variables) != n_vars:
            raise ValueError(f"变量名称数量({len(variables)})与数据列数({n_vars})不一致")
    else:
        raise ValueError("数据不能为空")

    # 3. 调用核心算法
    result: CoreVARResult = None
    if model_type == "var":
        result = core_var_model(var_data, lags=lags, variables=variables)
    elif model_type == "svar":
        result = core_svar_model(
            var_data, lags=lags, variables=variables, a_matrix=a_matrix, b_matrix=b_matrix
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # 4. 格式化输出
    if output_format == "json":
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    else:
        try:
            formatted = OutputFormatter.format_var_result(result, output_format)

        except Exception as e:
            formatted = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)

            formatted = f"警告: {output_format}格式化失败({str(e)})，返回JSON格式\n\n{formatted}"
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
            return f"{model_type.upper()}分析完成!\n\n{formatted}\n\n结果已保存到: {save_path}"
        return formatted


def cointegration_adapter(
    data: list[list[float]] | None = None,
    file_path: str | None = None,
    analysis_type: str = "johansen",
    variables: list[str] | None = None,
    coint_rank: int = 1,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    协整分析适配器

    Args:
        data: 多元时间序列数据 (格式: 每个子列表代表一个时间点的所有变量值)
        file_path: 数据文件路径
        analysis_type: 分析类型 ("engle-granger", "johansen", "vecm")
        variables: 变量名称列表
        coint_rank: 协整秩 (仅VECM)
        output_format: 输出格式
        save_path: 保存路径

    Returns:
        str: 格式化的分析结果
    """
    # 1. 数据准备
    if file_path:
        data_dict = DataLoader.load_from_file(file_path)
        data = data_dict["data"]
        variables = data_dict.get("variables") or variables
    elif data is None:
        raise ValueError("Must provide either file_path or data")

    # 2. 数据格式转换：从时间点格式转换为变量格式
    # 输入格式: [[var1_t1, var2_t1], [var1_t2, var2_t2], ...]
    # 需要转换为: [[var1_t1, var1_t2, ...], [var2_t1, var2_t2, ...]]
    if data and len(data) > 0:
        n_vars = len(data[0])
        n_obs = len(data)

        # 转换数据格式
        var_data = []
        for var_idx in range(n_vars):
            var_series = [data[t][var_idx] for t in range(n_obs)]
            var_data.append(var_series)

        # 如果没有提供变量名，自动生成
        if variables is None:
            variables = [f"Variable_{i}" for i in range(n_vars)]

        # 检查变量数量是否与数据一致
        if len(variables) != n_vars:
            raise ValueError(f"变量名称数量({len(variables)})与数据列数({n_vars})不一致")
    else:
        raise ValueError("数据不能为空")

    # 3. 调用核心算法
    result = None
    if analysis_type == "engle-granger":
        result: CoreCointegrationResult = core_engle_granger_cointegration_test(
            var_data, variables=variables
        )
    elif analysis_type == "johansen":
        result: CoreCointegrationResult = core_johansen_cointegration_test(
            var_data, variables=variables
        )
    elif analysis_type == "vecm":
        result: CoreVECMResult = core_vecm_model(
            var_data, coint_rank=coint_rank, variables=variables
        )
    else:
        raise ValueError(f"Unsupported analysis_type: {analysis_type}")

    # 4. 格式化输出
    if output_format == "json":
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    else:
        if analysis_type in ["engle-granger", "johansen"]:
            try:
                formatted = OutputFormatter.format_cointegration_result(result, output_format)

            except Exception as e:
                formatted = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)

                formatted = (
                    f"警告: {output_format}格式化失败({str(e)})，返回JSON格式\n\n{formatted}"
                )
        else:  # vecm
            try:
                formatted = OutputFormatter.format_vecm_result(result, output_format)

            except Exception as e:
                formatted = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)

                formatted = (
                    f"警告: {output_format}格式化失败({str(e)})，返回JSON格式\n\n{formatted}"
                )

        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
            return f"{analysis_type.upper()}分析完成!\n\n{formatted}\n\n结果已保存到: {save_path}"
        return formatted


def structural_break_adapter(
    data: list[float] | None = None,
    file_path: str | None = None,
    test_type: str = "chow",
    break_point: int | None = None,
    max_breaks: int = 5,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    结构突变检验适配器

    Args:
        data: 时间序列数据
        file_path: 数据文件路径
        test_type: 检验类型 ("chow", "quandt_andrews", "bai_perron")
        break_point: 断点位置 (仅Chow检验)
        max_breaks: 最大断点数 (仅Bai-Perron检验)
        output_format: 输出格式
        save_path: 保存路径

    Returns:
        str: 格式化的分析结果
    """
    # 数据准备
    if file_path:
        data_dict = DataLoader.load_from_file(file_path)
        data = data_dict["data"]
    elif data is None:
        raise ValueError("Must provide either file_path or data")

    # 调用核心算法
    result: CoreStructuralBreakResult = None
    if test_type == "chow":
        if break_point is None:
            break_point = len(data) // 2  # 默认中点
        result = core_chow_test(data, break_point)
    elif test_type == "quandt_andrews":
        result = core_quandt_andrews_test(data)
    elif test_type == "bai_perron":
        result = core_bai_perron_test(data, max_breaks)
    else:
        raise ValueError(f"Unsupported test_type: {test_type}")

    # 格式化输出
    if output_format == "json":
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    else:
        try:
            formatted = OutputFormatter.format_structural_break_result(result, output_format)
        except Exception as e:
            formatted = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
            formatted = f"警告: {output_format}格式化失败({str(e)})，返回JSON格式\n\n{formatted}"
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
            return f"结构突变检验({test_type})完成!\n\n{formatted}\n\n结果已保存到: {save_path}"
        return formatted


def time_varying_parameter_adapter(
    y_data: list[float] | None = None,
    x_data: list[list[float]] | None = None,
    file_path: str | None = None,
    model_type: str = "tar",
    threshold_variable: list[float] | None = None,
    n_regimes: int = 2,
    star_type: str = "logistic",
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    时变参数模型适配器

    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        file_path: 数据文件路径
        model_type: 模型类型 ("tar", "star", "markov_switching")
        threshold_variable: 门限变量 (TAR/STAR)
        n_regimes: 机制数量 (TAR/Markov)
        star_type: STAR类型 ("logistic", "exponential") (仅STAR)
        output_format: 输出格式
        save_path: 保存路径

    Returns:
        str: 格式化的分析结果
    """
    # 数据准备
    if file_path:
        data_dict = DataLoader.load_from_file(file_path)
        y_data = data_dict["y_data"]
        x_data = data_dict["x_data"]
        threshold_variable = data_dict.get("threshold_variable") or threshold_variable
    elif y_data is None or x_data is None:
        raise ValueError("Must provide either file_path or (y_data, x_data)")

    # 如果没有提供门限变量，使用y_data的滞后值
    if threshold_variable is None and model_type in ["tar", "star"]:
        threshold_variable = y_data[:-1]  # 使用y的滞后值

    # 调用核心算法
    result: CoreTimeVaryingParameterResult = None
    if model_type == "tar":
        result = core_tar_model(y_data, x_data, threshold_variable, n_regimes)
    elif model_type == "star":
        result = core_star_model(y_data, x_data, threshold_variable, star_type)
    elif model_type == "markov_switching":
        result = core_markov_switching_model(y_data, x_data, n_regimes)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # 格式化输出
    if output_format == "json":
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    else:
        try:
            formatted = OutputFormatter.format_time_varying_parameter_result(result, output_format)
        except Exception as e:
            formatted = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
            formatted = f"警告: {output_format}格式化失败({str(e)})，返回JSON格式\n\n{formatted}"
        if save_path:
            OutputFormatter.save_to_file(formatted, save_path)
            return (
                f"时变参数模型({model_type})分析完成!\n\n{formatted}\n\n结果已保存到: {save_path}"
            )
        return formatted
