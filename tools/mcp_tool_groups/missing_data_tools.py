"""
缺失数据处理工具组
"""

from typing import List, Optional, Dict, Any
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from ..mcp_tools_registry import ToolGroup
from ..missing_data_adapter import (
    simple_imputation_adapter,
    multiple_imputation_adapter
)


class MissingDataTools(ToolGroup):
    """缺失数据处理工具组"""
    
    name = "MISSING DATA HANDLING"
    description = "缺失数据插补和处理工具"
    version = "1.0.0"
    
    @classmethod
    def get_tools(cls) -> List[Dict[str, Any]]:
        """返回工具列表"""
        return [
            {
                "name": "missing_data_simple_imputation",
                "handler": cls.simple_imputation_tool,
                "description": "Simple Imputation (Mean/Median/Mode/Constant)"
            },
            {
                "name": "missing_data_multiple_imputation",
                "handler": cls.multiple_imputation_tool,
                "description": "Multiple Imputation (MICE)"
            }
        ]
    
    @classmethod
    def get_help_text(cls) -> str:
        return """
缺失数据处理工具组 - 2个工具

1. Simple Imputation (missing_data_simple_imputation)
   - 简单插补方法
   - 均值/中位数/众数/常数填充
   - 基于: sklearn.impute
   
2. Multiple Imputation (missing_data_multiple_imputation)
   - 多重插补 (MICE)
   - 迭代插补算法
   - 基于: sklearn.impute
"""
    
    @staticmethod
    async def simple_imputation_tool(
        data: List[List[float]],
        strategy: str = "mean",
        fill_value: Optional[float] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """简单插补"""
        try:
            if ctx:
                await ctx.info(f"Starting simple imputation ({strategy})...")
            
            result = simple_imputation_adapter(
                data=data,
                strategy=strategy,
                fill_value=fill_value,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Simple imputation complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def multiple_imputation_tool(
        data: List[List[float]],
        n_imputations: int = 5,
        max_iter: int = 10,
        random_state: Optional[int] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """多重插补(MICE)"""
        try:
            if ctx:
                await ctx.info(f"Starting multiple imputation (n={n_imputations})...")
            
            result = multiple_imputation_adapter(
                data=data,
                n_imputations=n_imputations,
                max_iter=max_iter,
                random_state=random_state,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Multiple imputation complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise