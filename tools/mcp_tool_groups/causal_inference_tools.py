"""
因果推断方法工具组
包含13种主要因果识别策略的MCP工具
"""

from typing import List, Optional, Union, Dict, Any
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from ..mcp_tools_registry import ToolGroup
from ..causal_inference_adapter import (
    did_adapter,
    iv_adapter,
    psm_adapter,
    fixed_effects_adapter,
    random_effects_adapter,
    rdd_adapter,
    synthetic_control_adapter,
    event_study_adapter,
    triple_difference_adapter,
    mediation_adapter,
    moderation_adapter,
    control_function_adapter,
    first_difference_adapter
)


class CausalInferenceTools(ToolGroup):
    """因果推断方法工具组"""
    
    name = "CAUSAL INFERENCE"
    description = "因果推断和识别策略工具"
    version = "1.0.0"
    
    @classmethod
    def get_tools(cls) -> List[Dict[str, Any]]:
        """返回工具列表"""
        return [
            {
                "name": "causal_difference_in_differences",
                "handler": cls.did_tool,
                "description": "Difference-in-Differences (DID) Analysis"
            },
            {
                "name": "causal_instrumental_variables",
                "handler": cls.iv_tool,
                "description": "Instrumental Variables (IV/2SLS) Analysis"
            },
            {
                "name": "causal_propensity_score_matching",
                "handler": cls.psm_tool,
                "description": "Propensity Score Matching (PSM) Analysis"
            },
            {
                "name": "causal_fixed_effects",
                "handler": cls.fixed_effects_tool,
                "description": "Fixed Effects Model"
            },
            {
                "name": "causal_random_effects",
                "handler": cls.random_effects_tool,
                "description": "Random Effects Model"
            },
            {
                "name": "causal_regression_discontinuity",
                "handler": cls.rdd_tool,
                "description": "Regression Discontinuity Design (RDD)"
            },
            {
                "name": "causal_synthetic_control",
                "handler": cls.synthetic_control_tool,
                "description": "Synthetic Control Method"
            },
            {
                "name": "causal_event_study",
                "handler": cls.event_study_tool,
                "description": "Event Study Analysis"
            },
            {
                "name": "causal_triple_difference",
                "handler": cls.triple_difference_tool,
                "description": "Triple Difference (DDD) Analysis"
            },
            {
                "name": "causal_mediation_analysis",
                "handler": cls.mediation_tool,
                "description": "Mediation Effect Analysis"
            },
            {
                "name": "causal_moderation_analysis",
                "handler": cls.moderation_tool,
                "description": "Moderation Effect Analysis"
            },
            {
                "name": "causal_control_function",
                "handler": cls.control_function_tool,
                "description": "Control Function Approach"
            },
            {
                "name": "causal_first_difference",
                "handler": cls.first_difference_tool,
                "description": "First Difference Model"
            }
        ]
    
    @classmethod
    def get_help_text(cls) -> str:
        """返回帮助文档"""
        return """
因果推断方法工具组 - 13种主要因果识别策略

1. Difference-in-Differences (DID) - causal_difference_in_differences
   - 双重差分法，用于评估政策干预效果
   - 需要数据：处理组、时间期、结果变量、协变量（可选）
   - 关键假设：平行趋势假设

2. Instrumental Variables (IV/2SLS) - causal_instrumental_variables
   - 工具变量法，解决内生性问题
   - 需要数据：因变量、内生自变量、工具变量
   - 关键假设：工具变量相关性和外生性

3. Propensity Score Matching (PSM) - causal_propensity_score_matching
   - 倾向得分匹配，控制混杂因素
   - 需要数据：处理状态、结果变量、协变量
   - 方法：最近邻匹配等

4. Fixed Effects Model - causal_fixed_effects
   - 固定效应模型，控制不随时间变化的个体异质性
   - 需要数据：面板数据（个体-时间）
   - 应用：面板数据分析

5. Random Effects Model - causal_random_effects
   - 随机效应模型，假设个体效应随机
   - 需要数据：面板数据（个体-时间）
   - 应用：面板数据分析

6. Regression Discontinuity Design (RDD) - causal_regression_discontinuity
   - 回归断点设计，利用连续变量的断点
   - 需要数据：运行变量、结果变量、断点值
   - 关键假设：断点处的连续性

7. Synthetic Control Method - causal_synthetic_control
   - 合成控制法，构造反事实对照组
   - 需要数据：多单元时间序列数据
   - 应用：政策评估、比较案例研究

8. Event Study - causal_event_study
   - 事件研究法，分析处理前后的动态效应
   - 需要数据：面板数据、事件时间
   - 应用：验证平行趋势假设

9. Triple Difference (DDD) - causal_triple_difference
   - 三重差分法，进一步控制混杂因素
   - 需要数据：处理组、时间、队列组
   - 应用：复杂政策评估

10. Mediation Analysis - causal_mediation_analysis
    - 中介效应分析，识别因果机制
    - 需要数据：结果、处理、中介变量
    - 方法：Baron-Kenny方法

11. Moderation Analysis - causal_moderation_analysis
    - 调节效应分析，检验条件效应
    - 需要数据：结果、预测变量、调节变量
    - 方法：交互项回归

12. Control Function Approach - causal_control_function
    - 控制函数法，解决内生性问题
    - 需要数据：因变量、内生变量、外生变量
    - 应用：非线性模型的内生性处理

13. First Difference Model - causal_first_difference
    - 一阶差分模型，消除固定效应
    - 需要数据：面板数据
    - 应用：短面板数据分析
"""

    @staticmethod
    async def did_tool(
        treatment: Optional[List[int]] = None,
        time_period: Optional[List[int]] = None,
        outcome: Optional[List[float]] = None,
        covariates: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """双重差分法(DID)分析"""
        try:
            if ctx:
                await ctx.info("Starting Difference-in-Differences analysis...")
            
            result = did_adapter(
                treatment=treatment,
                time_period=time_period,
                outcome=outcome,
                covariates=covariates,
                file_path=file_path,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("DID analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def iv_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[List[float]]] = None,
        instruments: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        instrument_names: Optional[List[str]] = None,
        constant: bool = True,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """工具变量法(IV/2SLS)分析"""
        try:
            if ctx:
                await ctx.info("Starting Instrumental Variables analysis...")
            
            result = iv_adapter(
                y_data=y_data,
                x_data=x_data,
                instruments=instruments,
                file_path=file_path,
                feature_names=feature_names,
                instrument_names=instrument_names,
                constant=constant,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("IV/2SLS analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def psm_tool(
        treatment: Optional[List[int]] = None,
        outcome: Optional[List[float]] = None,
        covariates: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        matching_method: str = "nearest",
        k_neighbors: int = 1,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """倾向得分匹配(PSM)分析"""
        try:
            if ctx:
                await ctx.info("Starting Propensity Score Matching analysis...")
            
            result = psm_adapter(
                treatment=treatment,
                outcome=outcome,
                covariates=covariates,
                file_path=file_path,
                matching_method=matching_method,
                k_neighbors=k_neighbors,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("PSM analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def fixed_effects_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[List[float]]] = None,
        entity_ids: Optional[List[str]] = None,
        time_periods: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        constant: bool = True,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """固定效应模型分析"""
        try:
            if ctx:
                await ctx.info("Starting Fixed Effects Model analysis...")
            
            result = fixed_effects_adapter(
                y_data=y_data,
                x_data=x_data,
                entity_ids=entity_ids,
                time_periods=time_periods,
                file_path=file_path,
                constant=constant,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Fixed Effects Model analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def random_effects_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[List[float]]] = None,
        entity_ids: Optional[List[str]] = None,
        time_periods: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """随机效应模型分析"""
        try:
            if ctx:
                await ctx.info("Starting Random Effects Model analysis...")
            
            result = random_effects_adapter(
                y_data=y_data,
                x_data=x_data,
                entity_ids=entity_ids,
                time_periods=time_periods,
                file_path=file_path,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Random Effects Model analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def rdd_tool(
        running_variable: Optional[List[float]] = None,
        outcome: Optional[List[float]] = None,
        cutoff: float = 0.0,
        file_path: Optional[str] = None,
        bandwidth: Optional[float] = None,
        polynomial_order: int = 1,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """回归断点设计(RDD)分析"""
        try:
            if ctx:
                await ctx.info("Starting Regression Discontinuity Design analysis...")
            
            result = rdd_adapter(
                running_variable=running_variable,
                outcome=outcome,
                cutoff=cutoff,
                file_path=file_path,
                bandwidth=bandwidth,
                polynomial_order=polynomial_order,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("RDD analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def synthetic_control_tool(
        outcome: Optional[List[float]] = None,
        treatment_period: int = 0,
        treated_unit: str = "unit_1",
        donor_units: Optional[List[str]] = None,
        time_periods: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """合成控制法分析"""
        try:
            if ctx:
                await ctx.info("Starting Synthetic Control Method analysis...")
            
            result = synthetic_control_adapter(
                outcome=outcome,
                treatment_period=treatment_period,
                treated_unit=treated_unit,
                donor_units=donor_units,
                time_periods=time_periods,
                file_path=file_path,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Synthetic Control Method analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def event_study_tool(
        outcome: Optional[List[float]] = None,
        treatment: Optional[List[int]] = None,
        entity_ids: Optional[List[str]] = None,
        time_periods: Optional[List[str]] = None,
        event_time: Optional[List[int]] = None,
        file_path: Optional[str] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """事件研究法分析"""
        try:
            if ctx:
                await ctx.info("Starting Event Study analysis...")
            
            result = event_study_adapter(
                outcome=outcome,
                treatment=treatment,
                entity_ids=entity_ids,
                time_periods=time_periods,
                event_time=event_time,
                file_path=file_path,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Event Study analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def triple_difference_tool(
        outcome: Optional[List[float]] = None,
        treatment_group: Optional[List[int]] = None,
        time_period: Optional[List[int]] = None,
        cohort_group: Optional[List[int]] = None,
        file_path: Optional[str] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """三重差分法(DDD)分析"""
        try:
            if ctx:
                await ctx.info("Starting Triple Difference analysis...")
            
            result = triple_difference_adapter(
                outcome=outcome,
                treatment_group=treatment_group,
                time_period=time_period,
                cohort_group=cohort_group,
                file_path=file_path,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Triple Difference analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def mediation_tool(
        outcome: Optional[List[float]] = None,
        treatment: Optional[List[float]] = None,
        mediator: Optional[List[float]] = None,
        covariates: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """中介效应分析"""
        try:
            if ctx:
                await ctx.info("Starting Mediation Analysis...")
            
            result = mediation_adapter(
                outcome=outcome,
                treatment=treatment,
                mediator=mediator,
                covariates=covariates,
                file_path=file_path,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Mediation Analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def moderation_tool(
        outcome: Optional[List[float]] = None,
        predictor: Optional[List[float]] = None,
        moderator: Optional[List[float]] = None,
        covariates: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """调节效应分析"""
        try:
            if ctx:
                await ctx.info("Starting Moderation Analysis...")
            
            result = moderation_adapter(
                outcome=outcome,
                predictor=predictor,
                moderator=moderator,
                covariates=covariates,
                file_path=file_path,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Moderation Analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def control_function_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[float]] = None,
        z_data: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        constant: bool = True,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """控制函数法分析"""
        try:
            if ctx:
                await ctx.info("Starting Control Function analysis...")
            
            result = control_function_adapter(
                y_data=y_data,
                x_data=x_data,
                z_data=z_data,
                file_path=file_path,
                constant=constant,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Control Function analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def first_difference_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[float]] = None,
        entity_ids: Optional[List[str]] = None,
        file_path: Optional[str] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """一阶差分模型分析"""
        try:
            if ctx:
                await ctx.info("Starting First Difference Model analysis...")
            
            result = first_difference_adapter(
                y_data=y_data,
                x_data=x_data,
                entity_ids=entity_ids,
                file_path=file_path,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("First Difference Model analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise