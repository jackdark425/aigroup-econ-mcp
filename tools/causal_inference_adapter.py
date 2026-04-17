"""
因果推断方法适配器
提供统一的接口调用econometrics/causal_inference中的各种因果识别方法
"""

# 导入所有因果推断方法
from econometrics.causal_inference.causal_identification_strategy.control_function import (
    ControlFunctionResult,
    control_function_approach,
)
from econometrics.causal_inference.causal_identification_strategy.difference_in_differences import (
    DIDResult,
    difference_in_differences,
)
from econometrics.causal_inference.causal_identification_strategy.event_study import (
    EventStudyResult,
    event_study,
)
from econometrics.causal_inference.causal_identification_strategy.first_difference import (
    FirstDifferenceResult,
    first_difference_model,
)
from econometrics.causal_inference.causal_identification_strategy.fixed_effects import (
    FixedEffectsResult,
    fixed_effects_model,
)
from econometrics.causal_inference.causal_identification_strategy.instrumental_variables import (
    IVResult,
    instrumental_variables_2sls,
)
from econometrics.causal_inference.causal_identification_strategy.mediation_analysis import (
    MediationResult,
    mediation_analysis,
)
from econometrics.causal_inference.causal_identification_strategy.moderation_analysis import (
    ModerationResult,
    moderation_analysis,
)
from econometrics.causal_inference.causal_identification_strategy.propensity_score_matching import (
    PSMMatchResult,
    propensity_score_matching,
)
from econometrics.causal_inference.causal_identification_strategy.random_effects import (
    RandomEffectsResult,
    random_effects_model,
)
from econometrics.causal_inference.causal_identification_strategy.regression_discontinuity import (
    RDDResult,
    regression_discontinuity,
)
from econometrics.causal_inference.causal_identification_strategy.synthetic_control import (
    SyntheticControlResult,
    synthetic_control_method,
)
from econometrics.causal_inference.causal_identification_strategy.triple_difference import (
    TripeDifferenceResult,
    triple_difference,
)

from .data_loader import merge_file_data
from .output_formatter import OutputFormatter


def did_adapter(
    treatment: list[int] | None = None,
    time_period: list[int] | None = None,
    outcome: list[float] | None = None,
    covariates: list[list[float]] | None = None,
    file_path: str | None = None,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    双重差分法 (DID) 适配器
    """
    # 从文件加载数据
    _merged = merge_file_data(file_path, treatment=treatment, time_period=time_period, outcome=outcome, covariates=covariates)
    treatment, time_period, outcome, covariates = _merged["treatment"], _merged["time_period"], _merged["outcome"], _merged["covariates"]

    # 调用核心方法
    result: DIDResult = difference_in_differences(
        treatment=treatment, time_period=time_period, outcome=outcome, covariates=covariates
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


def iv_adapter(
    y_data: list[float] | None = None,
    x_data: list[list[float]] | None = None,
    instruments: list[list[float]] | None = None,
    file_path: str | None = None,
    feature_names: list[str] | None = None,
    instrument_names: list[str] | None = None,
    constant: bool = True,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    工具变量法 (IV/2SLS) 适配器
    """
    # 从文件加载数据
    _merged = merge_file_data(file_path, y_data=y_data, x_data=x_data, instruments=instruments)
    y_data, x_data, instruments = _merged["y_data"], _merged["x_data"], _merged["instruments"]

    # 调用核心方法
    result: IVResult = instrumental_variables_2sls(
        y=y_data,
        x=x_data,
        instruments=instruments,
        feature_names=feature_names,
        instrument_names=instrument_names,
        constant=constant,
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


def psm_adapter(
    treatment: list[int] | None = None,
    outcome: list[float] | None = None,
    covariates: list[list[float]] | None = None,
    file_path: str | None = None,
    matching_method: str = "nearest",
    k_neighbors: int = 1,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    倾向得分匹配 (PSM) 适配器
    """
    # 从文件加载数据
    _merged = merge_file_data(file_path, treatment=treatment, outcome=outcome, covariates=covariates)
    treatment, outcome, covariates = _merged["treatment"], _merged["outcome"], _merged["covariates"]

    # 调用核心方法
    result: PSMMatchResult = propensity_score_matching(
        treatment=treatment,
        outcome=outcome,
        covariates=covariates,
        matching_method=matching_method,
        k_neighbors=k_neighbors,
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


def fixed_effects_adapter(
    y_data: list[float] | None = None,
    x_data: list[list[float]] | None = None,
    entity_ids: list[str] | None = None,
    time_periods: list[str] | None = None,
    file_path: str | None = None,
    constant: bool = True,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    固定效应模型适配器
    """
    # 从文件加载数据
    _merged = merge_file_data(file_path, y_data=y_data, x_data=x_data, entity_ids=entity_ids, time_periods=time_periods)
    y_data, x_data, entity_ids, time_periods = _merged["y_data"], _merged["x_data"], _merged["entity_ids"], _merged["time_periods"]

    # 调用核心方法
    result: FixedEffectsResult = fixed_effects_model(
        y=y_data, x=x_data, entity_ids=entity_ids, time_periods=time_periods, constant=constant
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


def random_effects_adapter(
    y_data: list[float] | None = None,
    x_data: list[list[float]] | None = None,
    entity_ids: list[str] | None = None,
    time_periods: list[str] | None = None,
    file_path: str | None = None,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    随机效应模型适配器
    """
    # 从文件加载数据
    _merged = merge_file_data(file_path, y_data=y_data, x_data=x_data, entity_ids=entity_ids, time_periods=time_periods)
    y_data, x_data, entity_ids, time_periods = _merged["y_data"], _merged["x_data"], _merged["entity_ids"], _merged["time_periods"]

    # 调用核心方法
    result: RandomEffectsResult = random_effects_model(
        y=y_data, x=x_data, entity_ids=entity_ids, time_periods=time_periods
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


def rdd_adapter(
    running_variable: list[float] | None = None,
    outcome: list[float] | None = None,
    cutoff: float = 0.0,
    file_path: str | None = None,
    bandwidth: float | None = None,
    polynomial_order: int = 1,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    回归断点设计 (RDD) 适配器
    """
    # 从文件加载数据
    _merged = merge_file_data(file_path, running_variable=running_variable, outcome=outcome, cutoff=cutoff)
    running_variable, outcome, cutoff = _merged["running_variable"], _merged["outcome"], _merged["cutoff"]

    # 调用核心方法
    result: RDDResult = regression_discontinuity(
        running_variable=running_variable,
        outcome=outcome,
        cutoff=cutoff,
        bandwidth=bandwidth,
        polynomial_order=polynomial_order,
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


def synthetic_control_adapter(
    outcome: list[float] | None = None,
    treatment_period: int = 0,
    treated_unit: str = "unit_1",
    donor_units: list[str] | None = None,
    time_periods: list[str] | None = None,
    file_path: str | None = None,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    合成控制法适配器
    """
    # 从文件加载数据
    _merged = merge_file_data(file_path, outcome=outcome, treatment_period=treatment_period, treated_unit=treated_unit, donor_units=donor_units, time_periods=time_periods)
    outcome, treatment_period, treated_unit, donor_units, time_periods = _merged["outcome"], _merged["treatment_period"], _merged["treated_unit"], _merged["donor_units"], _merged["time_periods"]

    # 调用核心方法
    result: SyntheticControlResult = synthetic_control_method(
        outcome=outcome,
        treatment_period=treatment_period,
        treated_unit=treated_unit,
        donor_units=donor_units,
        time_periods=time_periods,
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


def event_study_adapter(
    outcome: list[float] | None = None,
    treatment: list[int] | None = None,
    entity_ids: list[str] | None = None,
    time_periods: list[str] | None = None,
    event_time: list[int] | None = None,
    file_path: str | None = None,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    事件研究法适配器
    """
    # 从文件加载数据
    _merged = merge_file_data(file_path, outcome=outcome, treatment=treatment, entity_ids=entity_ids, time_periods=time_periods, event_time=event_time)
    outcome, treatment, entity_ids, time_periods, event_time = _merged["outcome"], _merged["treatment"], _merged["entity_ids"], _merged["time_periods"], _merged["event_time"]

    # 调用核心方法
    result: EventStudyResult = event_study(
        outcome=outcome,
        treatment=treatment,
        entity_ids=entity_ids,
        time_periods=time_periods,
        event_time=event_time,
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


def triple_difference_adapter(
    outcome: list[float] | None = None,
    treatment_group: list[int] | None = None,
    time_period: list[int] | None = None,
    cohort_group: list[int] | None = None,
    file_path: str | None = None,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    三重差分法 (DDD) 适配器
    """
    # 从文件加载数据
    _merged = merge_file_data(file_path, outcome=outcome, treatment_group=treatment_group, time_period=time_period, cohort_group=cohort_group)
    outcome, treatment_group, time_period, cohort_group = _merged["outcome"], _merged["treatment_group"], _merged["time_period"], _merged["cohort_group"]

    # 调用核心方法
    result: TripeDifferenceResult = triple_difference(
        outcome=outcome,
        treatment_group=treatment_group,
        time_period=time_period,
        cohort_group=cohort_group,
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


def mediation_adapter(
    outcome: list[float] | None = None,
    treatment: list[float] | None = None,
    mediator: list[float] | None = None,
    covariates: list[list[float]] | None = None,
    file_path: str | None = None,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    中介效应分析适配器
    """
    # 从文件加载数据
    _merged = merge_file_data(file_path, outcome=outcome, treatment=treatment, mediator=mediator, covariates=covariates)
    outcome, treatment, mediator, covariates = _merged["outcome"], _merged["treatment"], _merged["mediator"], _merged["covariates"]

    # 调用核心方法
    result: MediationResult = mediation_analysis(
        outcome=outcome, treatment=treatment, mediator=mediator, covariates=covariates
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


def moderation_adapter(
    outcome: list[float] | None = None,
    predictor: list[float] | None = None,
    moderator: list[float] | None = None,
    covariates: list[list[float]] | None = None,
    file_path: str | None = None,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    调节效应分析适配器
    """
    # 从文件加载数据
    _merged = merge_file_data(file_path, outcome=outcome, predictor=predictor, moderator=moderator, covariates=covariates)
    outcome, predictor, moderator, covariates = _merged["outcome"], _merged["predictor"], _merged["moderator"], _merged["covariates"]

    # 调用核心方法
    result: ModerationResult = moderation_analysis(
        outcome=outcome, predictor=predictor, moderator=moderator, covariates=covariates
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


def control_function_adapter(
    y_data: list[float] | None = None,
    x_data: list[float] | None = None,
    z_data: list[list[float]] | None = None,
    file_path: str | None = None,
    constant: bool = True,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    控制函数法适配器
    """
    # 从文件加载数据
    _merged = merge_file_data(file_path, y_data=y_data, x_data=x_data, z_data=z_data)
    y_data, x_data, z_data = _merged["y_data"], _merged["x_data"], _merged["z_data"]

    # 调用核心方法
    result: ControlFunctionResult = control_function_approach(
        y=y_data, x=x_data, z=z_data, constant=constant
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


def first_difference_adapter(
    y_data: list[float] | None = None,
    x_data: list[float] | None = None,
    entity_ids: list[str] | None = None,
    file_path: str | None = None,
    output_format: str = "json",
    save_path: str | None = None,
) -> str:
    """
    一阶差分模型适配器
    """
    # 从文件加载数据
    _merged = merge_file_data(file_path, y_data=y_data, x_data=x_data, entity_ids=entity_ids)
    y_data, x_data, entity_ids = _merged["y_data"], _merged["x_data"], _merged["entity_ids"]

    # 调用核心方法
    result: FirstDifferenceResult = first_difference_model(
        y=y_data, x=x_data, entity_ids=entity_ids
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
