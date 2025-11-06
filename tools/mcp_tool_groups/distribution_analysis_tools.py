"""
分布分析与分解方法工具组
包含Oaxaca-Blinder分解、方差分解、时间序列分解
"""

from typing import List, Optional, Dict, Any
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from ..mcp_tools_registry import ToolGroup
from ..distribution_analysis_adapter import (
    oaxaca_blinder_adapter,
    variance_decomposition_adapter,
    time_series_decomposition_adapter
)


class DistributionAnalysisTools(ToolGroup):
    """分布分析与分解方法工具组"""
    
    name = "DISTRIBUTION ANALYSIS & DECOMPOSITION"
    description = "分布分析和分解方法工具"
    version = "1.0.0"
    
    @classmethod
    def get_tools(cls) -> List[Dict[str, Any]]:
        """返回工具列表"""
        return [
            {
                "name": "decomposition_oaxaca_blinder",
                "handler": cls.oaxaca_blinder_tool,
                "description": "Oaxaca-Blinder Decomposition"
            },
            {
                "name": "decomposition_variance_anova",
                "handler": cls.variance_decomposition_tool,
                "description": "Variance Decomposition (ANOVA)"
            },
            {
                "name": "decomposition_time_series",
                "handler": cls.time_series_decomposition_tool,
                "description": "Time Series Decomposition (Trend-Seasonal-Random)"
            }
        ]
    
    @classmethod
    def get_help_text(cls) -> str:
        """返回帮助文档"""
        return """
分布分析与分解方法工具组 - 3个工具

1. Oaxaca-Blinder Decomposition (decomposition_oaxaca_blinder)
   - 分解两组之间的平均差异
   - 禀赋效应 vs 系数效应
   - 应用: 工资差距、就业差异分析
   
2. Variance Decomposition (decomposition_variance_anova)
   - 单因素方差分析
   - 组间方差 vs 组内方差
   - F检验和效应量估计
   
3. Time Series Decomposition (decomposition_time_series)
   - 趋势-季节-随机分解
   - 加法/乘法模型
   - 经典分解或STL分解
"""
    
    @staticmethod
    async def oaxaca_blinder_tool(
        y_a: List[float],
        x_a: List[List[float]],
        y_b: List[float],
        x_b: List[List[float]],
        feature_names: Optional[List[str]] = None,
        weight_matrix: str = "pooled",
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Oaxaca-Blinder分解"""
        try:
            if ctx:
                await ctx.info("Starting Oaxaca-Blinder decomposition...")
            
            result = oaxaca_blinder_adapter(
                y_a=y_a,
                x_a=x_a,
                y_b=y_b,
                x_b=x_b,
                feature_names=feature_names,
                weight_matrix=weight_matrix,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Oaxaca-Blinder decomposition complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def variance_decomposition_tool(
        values: List[float],
        groups: List[str],
        group_names: Optional[List[str]] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """方差分解(ANOVA)"""
        try:
            if ctx:
                await ctx.info("Starting variance decomposition (ANOVA)...")
            
            result = variance_decomposition_adapter(
                values=values,
                groups=groups,
                group_names=group_names,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Variance decomposition complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def time_series_decomposition_tool(
        data: List[float],
        period: int = 12,
        model: str = "additive",
        method: str = "classical",
        extrapolate_trend: str = "freq",
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """时间序列分解"""
        try:
            if ctx:
                await ctx.info(f"Starting time series decomposition ({method})...")
            
            result = time_series_decomposition_adapter(
                data=data,
                period=period,
                model=model,
                method=method,
                extrapolate_trend=extrapolate_trend,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Time series decomposition complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise