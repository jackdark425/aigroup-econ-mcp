"""
统计推断技术工具组
包含Bootstrap和置换检验
"""

from typing import List, Optional, Dict, Any
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from ..mcp_tools_registry import ToolGroup
from ..statistical_inference_adapter import (
    bootstrap_adapter,
    permutation_test_adapter
)


class StatisticalInferenceTools(ToolGroup):
    """统计推断技术工具组"""
    
    name = "STATISTICAL INFERENCE TECHNIQUES"
    description = "统计推断和重采样方法工具"
    version = "1.0.0"
    
    @classmethod
    def get_tools(cls) -> List[Dict[str, Any]]:
        """返回工具列表"""
        return [
            {
                "name": "inference_bootstrap",
                "handler": cls.bootstrap_tool,
                "description": "Bootstrap Resampling Inference"
            },
            {
                "name": "inference_permutation_test",
                "handler": cls.permutation_test_tool,
                "description": "Permutation Test (Nonparametric)"
            }
        ]
    
    @classmethod
    def get_help_text(cls) -> str:
        """返回帮助文档"""
        return """
统计推断技术工具组 - 2个工具

1. Bootstrap Inference (inference_bootstrap)
   - Bootstrap重采样推断
   - 置信区间估计
   - 支持多种统计量和置信区间方法
   - 基于: scipy.stats
   
2. Permutation Test (inference_permutation_test)
   - 置换检验（非参数）
   - 两样本比较
   - 均值/中位数/方差比检验
   - 基于: scipy.stats
"""
    
    @staticmethod
    async def bootstrap_tool(
        data: List[float],
        statistic_func: str = "mean",
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        method: str = "percentile",
        random_state: Optional[int] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Bootstrap重采样推断"""
        try:
            if ctx:
                await ctx.info(f"Starting Bootstrap inference ({statistic_func})...")
            
            result = bootstrap_adapter(
                data=data,
                statistic_func=statistic_func,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                method=method,
                random_state=random_state,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Bootstrap inference complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def permutation_test_tool(
        sample_a: List[float],
        sample_b: List[float],
        test_type: str = "mean_difference",
        alternative: str = "two-sided",
        n_permutations: int = 10000,
        random_state: Optional[int] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """置换检验"""
        try:
            if ctx:
                await ctx.info(f"Starting permutation test ({test_type})...")
            
            result = permutation_test_adapter(
                sample_a=sample_a,
                sample_b=sample_b,
                test_type=test_type,
                alternative=alternative,
                n_permutations=n_permutations,
                random_state=random_state,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Permutation test complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise