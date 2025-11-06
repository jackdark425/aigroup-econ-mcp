"""
微观计量模型工具组
包含离散选择、计数数据和受限因变量模型的MCP工具
"""

from typing import List, Optional, Union, Dict, Any
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from ..mcp_tools_registry import ToolGroup
from ..microecon_adapter import (
    logit_adapter,
    probit_adapter,
    multinomial_logit_adapter,
    poisson_adapter,
    negative_binomial_adapter,
    tobit_adapter,
    heckman_adapter
)


class MicroeconometricsTools(ToolGroup):
    """微观计量模型工具组"""
    
    name = "MICROECONOMETRICS"
    description = "微观计量模型工具"
    version = "1.0.0"
    
    @classmethod
    def get_tools(cls) -> List[Dict[str, Any]]:
        """返回工具列表"""
        return [
            {
                "name": "micro_logit",
                "handler": cls.logit_tool,
                "description": "Logistic Regression Model"
            },
            {
                "name": "micro_probit",
                "handler": cls.probit_tool,
                "description": "Probit Regression Model"
            },
            {
                "name": "micro_multinomial_logit",
                "handler": cls.multinomial_logit_tool,
                "description": "Multinomial Logit Model"
            },
            {
                "name": "micro_poisson",
                "handler": cls.poisson_tool,
                "description": "Poisson Regression Model"
            },
            {
                "name": "micro_negative_binomial",
                "handler": cls.negative_binomial_tool,
                "description": "Negative Binomial Regression Model"
            },
            {
                "name": "micro_tobit",
                "handler": cls.tobit_tool,
                "description": "Tobit Model (Censored Regression)"
            },
            {
                "name": "micro_heckman",
                "handler": cls.heckman_tool,
                "description": "Heckman Selection Model"
            }
        ]
    
    @classmethod
    def get_help_text(cls) -> str:
        """返回帮助文档"""
        return """
微观计量模型工具组 - 7种模型

离散选择模型:
1. Logit Model - micro_logit
   - 二元Logistic回归
   - 适用于二元因变量
   
2. Probit Model - micro_probit
   - Probit回归
   - 基于正态分布假设

3. Multinomial Logit - micro_multinomial_logit
   - 多项Logit模型
   - 适用于多分类问题

计数数据模型:
4. Poisson Model - micro_poisson
   - 泊松回归
   - 适用于计数数据
   
5. Negative Binomial - micro_negative_binomial
   - 负二项回归
   - 处理过度离散问题

受限因变量模型:
6. Tobit Model - micro_tobit
   - Tobit模型（截断回归）
   - 适用于受限因变量
   
7. Heckman Model - micro_heckman
   - Heckman样本选择模型
   - 处理样本选择偏差
"""
    
    @staticmethod
    async def logit_tool(
        X_data: Optional[List] = None,
        y_data: Optional[List[int]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        output_format: str = 'json',
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Logit回归分析"""
        try:
            if ctx:
                await ctx.info("Starting Logit regression analysis...")
            
            result = logit_adapter(
                X_data=X_data, y_data=y_data, file_path=file_path,
                feature_names=feature_names, output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Logit analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def probit_tool(
        X_data: Optional[List] = None,
        y_data: Optional[List[int]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        output_format: str = 'json',
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Probit回归分析"""
        try:
            if ctx:
                await ctx.info("Starting Probit regression analysis...")
            
            result = probit_adapter(
                X_data=X_data, y_data=y_data, file_path=file_path,
                feature_names=feature_names, output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Probit analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def multinomial_logit_tool(
        X_data: Optional[List] = None,
        y_data: Optional[List[int]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        output_format: str = 'json',
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """多项Logit分析"""
        try:
            if ctx:
                await ctx.info("Starting Multinomial Logit analysis...")
            
            result = multinomial_logit_adapter(
                X_data=X_data, y_data=y_data, file_path=file_path,
                feature_names=feature_names, output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Multinomial Logit analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def poisson_tool(
        X_data: Optional[List] = None,
        y_data: Optional[List[int]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        output_format: str = 'json',
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """泊松回归分析"""
        try:
            if ctx:
                await ctx.info("Starting Poisson regression analysis...")
            
            result = poisson_adapter(
                X_data=X_data, y_data=y_data, file_path=file_path,
                feature_names=feature_names, output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Poisson analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def negative_binomial_tool(
        X_data: Optional[List] = None,
        y_data: Optional[List[int]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        distr: str = 'nb2',
        output_format: str = 'json',
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """负二项回归分析"""
        try:
            if ctx:
                await ctx.info("Starting Negative Binomial regression analysis...")
            
            result = negative_binomial_adapter(
                X_data=X_data, y_data=y_data, file_path=file_path,
                feature_names=feature_names, distr=distr,
                output_format=output_format, save_path=save_path
            )
            
            if ctx:
                await ctx.info("Negative Binomial analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def tobit_tool(
        X_data: Optional[List] = None,
        y_data: Optional[List[float]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        lower_bound: float = 0.0,
        upper_bound: Optional[float] = None,
        output_format: str = 'json',
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Tobit模型分析"""
        try:
            if ctx:
                await ctx.info("Starting Tobit model analysis...")
            
            result = tobit_adapter(
                X_data=X_data, y_data=y_data, file_path=file_path,
                feature_names=feature_names, lower_bound=lower_bound,
                upper_bound=upper_bound, output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Tobit analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def heckman_tool(
        X_select_data: Optional[List] = None,
        Z_data: Optional[List] = None,
        y_data: Optional[List[float]] = None,
        s_data: Optional[List[int]] = None,
        file_path: Optional[str] = None,
        selection_feature_names: Optional[List[str]] = None,
        outcome_feature_names: Optional[List[str]] = None,
        output_format: str = 'json',
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Heckman样本选择模型分析"""
        try:
            if ctx:
                await ctx.info("Starting Heckman selection model analysis...")
            
            result = heckman_adapter(
                X_select_data=X_select_data, Z_data=Z_data,
                y_data=y_data, s_data=s_data, file_path=file_path,
                selection_feature_names=selection_feature_names,
                outcome_feature_names=outcome_feature_names,
                output_format=output_format, save_path=save_path
            )
            
            if ctx:
                await ctx.info("Heckman analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise