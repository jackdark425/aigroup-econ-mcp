"""
因果推断方法适配器
提供统一的接口调用econometrics/causal_inference中的各种因果识别方法
"""

from typing import List, Optional, Union, Dict, Any
import json

# 导入所有因果推断方法
from econometrics.causal_inference.causal_identification_strategy.difference_in_differences import (
    difference_in_differences, DIDResult
)
from econometrics.causal_inference.causal_identification_strategy.instrumental_variables import (
    instrumental_variables_2sls, IVResult
)
from econometrics.causal_inference.causal_identification_strategy.propensity_score_matching import (
    propensity_score_matching, PSMMatchResult
)
from econometrics.causal_inference.causal_identification_strategy.fixed_effects import (
    fixed_effects_model, FixedEffectsResult
)
from econometrics.causal_inference.causal_identification_strategy.random_effects import (
    random_effects_model, RandomEffectsResult
)
from econometrics.causal_inference.causal_identification_strategy.regression_discontinuity import (
    regression_discontinuity, RDDResult
)
from econometrics.causal_inference.causal_identification_strategy.synthetic_control import (
    synthetic_control_method, SyntheticControlResult
)
from econometrics.causal_inference.causal_identification_strategy.event_study import (
    event_study, EventStudyResult
)
from econometrics.causal_inference.causal_identification_strategy.triple_difference import (
    triple_difference, TripeDifferenceResult
)
from econometrics.causal_inference.causal_identification_strategy.mediation_analysis import (
    mediation_analysis, MediationResult
)
from econometrics.causal_inference.causal_identification_strategy.moderation_analysis import (
    moderation_analysis, ModerationResult
)
from econometrics.causal_inference.causal_identification_strategy.control_function import (
    control_function_approach, ControlFunctionResult
)
from econometrics.causal_inference.causal_identification_strategy.first_difference import (
    first_difference_model, FirstDifferenceResult
)

from .data_loader import DataLoader
from .output_formatter import OutputFormatter


def did_adapter(
    treatment: Optional[List[int]] = None,
    time_period: Optional[List[int]] = None,
    outcome: Optional[List[float]] = None,
    covariates: Optional[List[List[float]]] = None,
    file_path: Optional[str] = None,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """
    双重差分法 (DID) 适配器
    """
    try:
        # 从文件加载数据
        if file_path:
            data = DataLoader.load_from_file(file_path)
            treatment = data.get("treatment", treatment)
            time_period = data.get("time_period", time_period)
            outcome = data.get("outcome", outcome)
            covariates = data.get("covariates", covariates)
        
        # 调用核心方法
        result: DIDResult = difference_in_differences(
            treatment=treatment,
            time_period=time_period,
            outcome=outcome,
            covariates=covariates
        )
        
        # 格式化输出
        if output_format == "json":
            output = result.model_dump_json(indent=2)
        else:
            output = str(result.model_dump())
        
        # 保存结果
        if save_path:
            OutputFormatter.save_to_file(output, save_path)
        
        return output
        
    except Exception as e:
        error_msg = f"DID分析错误: {str(e)}"
        return json.dumps({"error": error_msg}, indent=2)


def iv_adapter(
    y_data: Optional[List[float]] = None,
    x_data: Optional[List[List[float]]] = None,
    instruments: Optional[List[List[float]]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    instrument_names: Optional[List[str]] = None,
    constant: bool = True,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """
    工具变量法 (IV/2SLS) 适配器
    """
    try:
        # 从文件加载数据
        if file_path:
            data = DataLoader.load_from_file(file_path)
            y_data = data.get("y_data", y_data)
            x_data = data.get("x_data", x_data)
            instruments = data.get("instruments", instruments)
        
        # 调用核心方法
        result: IVResult = instrumental_variables_2sls(
            y=y_data,
            x=x_data,
            instruments=instruments,
            feature_names=feature_names,
            instrument_names=instrument_names,
            constant=constant
        )
        
        # 格式化输出
        if output_format == "json":
            output = result.model_dump_json(indent=2)
        else:
            output = str(result.model_dump())
        
        # 保存结果
        if save_path:
            OutputFormatter.save_to_file(output, save_path)
        
        return output
        
    except Exception as e:
        error_msg = f"IV/2SLS分析错误: {str(e)}"
        return json.dumps({"error": error_msg}, indent=2)


def psm_adapter(
    treatment: Optional[List[int]] = None,
    outcome: Optional[List[float]] = None,
    covariates: Optional[List[List[float]]] = None,
    file_path: Optional[str] = None,
    matching_method: str = "nearest",
    k_neighbors: int = 1,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """
    倾向得分匹配 (PSM) 适配器
    """
    try:
        # 从文件加载数据
        if file_path:
            data = DataLoader.load_from_file(file_path)
            treatment = data.get("treatment", treatment)
            outcome = data.get("outcome", outcome)
            covariates = data.get("covariates", covariates)
        
        # 调用核心方法
        result: PSMMatchResult = propensity_score_matching(
            treatment=treatment,
            outcome=outcome,
            covariates=covariates,
            matching_method=matching_method,
            k_neighbors=k_neighbors
        )
        
        # 格式化输出
        if output_format == "json":
            output = result.model_dump_json(indent=2)
        else:
            output = str(result.model_dump())
        
        # 保存结果
        if save_path:
            OutputFormatter.save_to_file(output, save_path)
        
        return output
        
    except Exception as e:
        error_msg = f"PSM分析错误: {str(e)}"
        return json.dumps({"error": error_msg}, indent=2)


def fixed_effects_adapter(
    y_data: Optional[List[float]] = None,
    x_data: Optional[List[List[float]]] = None,
    entity_ids: Optional[List[str]] = None,
    time_periods: Optional[List[str]] = None,
    file_path: Optional[str] = None,
    constant: bool = True,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """
    固定效应模型适配器
    """
    try:
        # 从文件加载数据
        if file_path:
            data = DataLoader.load_from_file(file_path)
            y_data = data.get("y_data", y_data)
            x_data = data.get("x_data", x_data)
            entity_ids = data.get("entity_ids", entity_ids)
            time_periods = data.get("time_periods", time_periods)
        
        # 调用核心方法
        result: FixedEffectsResult = fixed_effects_model(
            y=y_data,
            x=x_data,
            entity_ids=entity_ids,
            time_periods=time_periods,
            constant=constant
        )
        
        # 格式化输出
        if output_format == "json":
            output = result.model_dump_json(indent=2)
        else:
            output = str(result.model_dump())
        
        # 保存结果
        if save_path:
            OutputFormatter.save_to_file(output, save_path)
        
        return output
        
    except Exception as e:
        error_msg = f"固定效应模型分析错误: {str(e)}"
        return json.dumps({"error": error_msg}, indent=2)


def random_effects_adapter(
    y_data: Optional[List[float]] = None,
    x_data: Optional[List[List[float]]] = None,
    entity_ids: Optional[List[str]] = None,
    time_periods: Optional[List[str]] = None,
    file_path: Optional[str] = None,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """
    随机效应模型适配器
    """
    try:
        # 从文件加载数据
        if file_path:
            data = DataLoader.load_from_file(file_path)
            y_data = data.get("y_data", y_data)
            x_data = data.get("x_data", x_data)
            entity_ids = data.get("entity_ids", entity_ids)
            time_periods = data.get("time_periods", time_periods)
        
        # 调用核心方法
        result: RandomEffectsResult = random_effects_model(
            y=y_data,
            x=x_data,
            entity_ids=entity_ids,
            time_periods=time_periods
        )
        
        # 格式化输出
        if output_format == "json":
            output = result.model_dump_json(indent=2)
        else:
            output = str(result.model_dump())
        
        # 保存结果
        if save_path:
            OutputFormatter.save_to_file(output, save_path)
        
        return output
        
    except Exception as e:
        error_msg = f"随机效应模型分析错误: {str(e)}"
        return json.dumps({"error": error_msg}, indent=2)


def rdd_adapter(
    running_variable: Optional[List[float]] = None,
    outcome: Optional[List[float]] = None,
    cutoff: float = 0.0,
    file_path: Optional[str] = None,
    bandwidth: Optional[float] = None,
    polynomial_order: int = 1,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """
    回归断点设计 (RDD) 适配器
    """
    try:
        # 从文件加载数据
        if file_path:
            data = DataLoader.load_from_file(file_path)
            running_variable = data.get("running_variable", running_variable)
            outcome = data.get("outcome", outcome)
            cutoff = data.get("cutoff", cutoff)
        
        # 调用核心方法
        result: RDDResult = regression_discontinuity(
            running_variable=running_variable,
            outcome=outcome,
            cutoff=cutoff,
            bandwidth=bandwidth,
            polynomial_order=polynomial_order
        )
        
        # 格式化输出
        if output_format == "json":
            output = result.model_dump_json(indent=2)
        else:
            output = str(result.model_dump())
        
        # 保存结果
        if save_path:
            OutputFormatter.save_to_file(output, save_path)
        
        return output
        
    except Exception as e:
        error_msg = f"RDD分析错误: {str(e)}"
        return json.dumps({"error": error_msg}, indent=2)


def synthetic_control_adapter(
    outcome: Optional[List[float]] = None,
    treatment_period: int = 0,
    treated_unit: str = "unit_1",
    donor_units: Optional[List[str]] = None,
    time_periods: Optional[List[str]] = None,
    file_path: Optional[str] = None,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """
    合成控制法适配器
    """
    try:
        # 从文件加载数据
        if file_path:
            data = DataLoader.load_from_file(file_path)
            outcome = data.get("outcome", outcome)
            treatment_period = data.get("treatment_period", treatment_period)
            treated_unit = data.get("treated_unit", treated_unit)
            donor_units = data.get("donor_units", donor_units)
            time_periods = data.get("time_periods", time_periods)
        
        # 调用核心方法
        result: SyntheticControlResult = synthetic_control_method(
            outcome=outcome,
            treatment_period=treatment_period,
            treated_unit=treated_unit,
            donor_units=donor_units,
            time_periods=time_periods
        )
        
        # 格式化输出
        if output_format == "json":
            output = result.model_dump_json(indent=2)
        else:
            output = str(result.model_dump())
        
        # 保存结果
        if save_path:
            OutputFormatter.save_to_file(output, save_path)
        
        return output
        
    except Exception as e:
        error_msg = f"合成控制法分析错误: {str(e)}"
        return json.dumps({"error": error_msg}, indent=2)


def event_study_adapter(
    outcome: Optional[List[float]] = None,
    treatment: Optional[List[int]] = None,
    entity_ids: Optional[List[str]] = None,
    time_periods: Optional[List[str]] = None,
    event_time: Optional[List[int]] = None,
    file_path: Optional[str] = None,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """
    事件研究法适配器
    """
    try:
        # 从文件加载数据
        if file_path:
            data = DataLoader.load_from_file(file_path)
            outcome = data.get("outcome", outcome)
            treatment = data.get("treatment", treatment)
            entity_ids = data.get("entity_ids", entity_ids)
            time_periods = data.get("time_periods", time_periods)
            event_time = data.get("event_time", event_time)
        
        # 调用核心方法
        result: EventStudyResult = event_study(
            outcome=outcome,
            treatment=treatment,
            entity_ids=entity_ids,
            time_periods=time_periods,
            event_time=event_time
        )
        
        # 格式化输出
        if output_format == "json":
            output = result.model_dump_json(indent=2)
        else:
            output = str(result.model_dump())
        
        # 保存结果
        if save_path:
            OutputFormatter.save_to_file(output, save_path)
        
        return output
        
    except Exception as e:
        error_msg = f"事件研究法分析错误: {str(e)}"
        return json.dumps({"error": error_msg}, indent=2)


def triple_difference_adapter(
    outcome: Optional[List[float]] = None,
    treatment_group: Optional[List[int]] = None,
    time_period: Optional[List[int]] = None,
    cohort_group: Optional[List[int]] = None,
    file_path: Optional[str] = None,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """
    三重差分法 (DDD) 适配器
    """
    try:
        # 从文件加载数据
        if file_path:
            data = DataLoader.load_from_file(file_path)
            outcome = data.get("outcome", outcome)
            treatment_group = data.get("treatment_group", treatment_group)
            time_period = data.get("time_period", time_period)
            cohort_group = data.get("cohort_group", cohort_group)
        
        # 调用核心方法
        result: TripeDifferenceResult = triple_difference(
            outcome=outcome,
            treatment_group=treatment_group,
            time_period=time_period,
            cohort_group=cohort_group
        )
        
        # 格式化输出
        if output_format == "json":
            output = result.model_dump_json(indent=2)
        else:
            output = str(result.model_dump())
        
        # 保存结果
        if save_path:
            OutputFormatter.save_to_file(output, save_path)
        
        return output
        
    except Exception as e:
        error_msg = f"DDD分析错误: {str(e)}"
        return json.dumps({"error": error_msg}, indent=2)


def mediation_adapter(
    outcome: Optional[List[float]] = None,
    treatment: Optional[List[float]] = None,
    mediator: Optional[List[float]] = None,
    covariates: Optional[List[List[float]]] = None,
    file_path: Optional[str] = None,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """
    中介效应分析适配器
    """
    try:
        # 从文件加载数据
        if file_path:
            data = DataLoader.load_from_file(file_path)
            outcome = data.get("outcome", outcome)
            treatment = data.get("treatment", treatment)
            mediator = data.get("mediator", mediator)
            covariates = data.get("covariates", covariates)
        
        # 调用核心方法
        result: MediationResult = mediation_analysis(
            outcome=outcome,
            treatment=treatment,
            mediator=mediator,
            covariates=covariates
        )
        
        # 格式化输出
        if output_format == "json":
            output = result.model_dump_json(indent=2)
        else:
            output = str(result.model_dump())
        
        # 保存结果
        if save_path:
            OutputFormatter.save_to_file(output, save_path)
        
        return output
        
    except Exception as e:
        error_msg = f"中介效应分析错误: {str(e)}"
        return json.dumps({"error": error_msg}, indent=2)


def moderation_adapter(
    outcome: Optional[List[float]] = None,
    predictor: Optional[List[float]] = None,
    moderator: Optional[List[float]] = None,
    covariates: Optional[List[List[float]]] = None,
    file_path: Optional[str] = None,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """
    调节效应分析适配器
    """
    try:
        # 从文件加载数据
        if file_path:
            data = DataLoader.load_from_file(file_path)
            outcome = data.get("outcome", outcome)
            predictor = data.get("predictor", predictor)
            moderator = data.get("moderator", moderator)
            covariates = data.get("covariates", covariates)
        
        # 调用核心方法
        result: ModerationResult = moderation_analysis(
            outcome=outcome,
            predictor=predictor,
            moderator=moderator,
            covariates=covariates
        )
        
        # 格式化输出
        if output_format == "json":
            output = result.model_dump_json(indent=2)
        else:
            output = str(result.model_dump())
        
        # 保存结果
        if save_path:
            OutputFormatter.save_to_file(output, save_path)
        
        return output
        
    except Exception as e:
        error_msg = f"调节效应分析错误: {str(e)}"
        return json.dumps({"error": error_msg}, indent=2)


def control_function_adapter(
    y_data: Optional[List[float]] = None,
    x_data: Optional[List[float]] = None,
    z_data: Optional[List[List[float]]] = None,
    file_path: Optional[str] = None,
    constant: bool = True,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """
    控制函数法适配器
    """
    try:
        # 从文件加载数据
        if file_path:
            data = DataLoader.load_from_file(file_path)
            y_data = data.get("y_data", y_data)
            x_data = data.get("x_data", x_data)
            z_data = data.get("z_data", z_data)
        
        # 调用核心方法
        result: ControlFunctionResult = control_function_approach(
            y=y_data,
            x=x_data,
            z=z_data,
            constant=constant
        )
        
        # 格式化输出
        if output_format == "json":
            output = result.model_dump_json(indent=2)
        else:
            output = str(result.model_dump())
        
        # 保存结果
        if save_path:
            OutputFormatter.save_to_file(output, save_path)
        
        return output
        
    except Exception as e:
        error_msg = f"控制函数法分析错误: {str(e)}"
        return json.dumps({"error": error_msg}, indent=2)


def first_difference_adapter(
    y_data: Optional[List[float]] = None,
    x_data: Optional[List[float]] = None,
    entity_ids: Optional[List[str]] = None,
    file_path: Optional[str] = None,
    output_format: str = "json",
    save_path: Optional[str] = None
) -> str:
    """
    一阶差分模型适配器
    """
    try:
        # 从文件加载数据
        if file_path:
            data = DataLoader.load_from_file(file_path)
            y_data = data.get("y_data", y_data)
            x_data = data.get("x_data", x_data)
            entity_ids = data.get("entity_ids", entity_ids)
        
        # 调用核心方法
        result: FirstDifferenceResult = first_difference_model(
            y=y_data,
            x=x_data,
            entity_ids=entity_ids
        )
        
        # 格式化输出
        if output_format == "json":
            output = result.model_dump_json(indent=2)
        else:
            output = str(result.model_dump())
        
        # 保存结果
        if save_path:
            OutputFormatter.save_to_file(output, save_path)
        
        return output
        
    except Exception as e:
        error_msg = f"一阶差分模型分析错误: {str(e)}"
        return json.dumps({"error": error_msg}, indent=2)