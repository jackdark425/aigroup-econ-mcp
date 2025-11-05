"""
基础参数估计工具组
包含 OLS、MLE、GMM 三个核心工具
"""

from typing import List, Optional, Union, Dict, Any
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from ..mcp_tools_registry import ToolGroup
from ..econometrics_adapter import ols_adapter, mle_adapter, gmm_adapter


class BasicParametricTools(ToolGroup):
    """基础参数估计工具组"""
    
    name = "BASIC PARAMETRIC ESTIMATION"
    description = "使用 econometrics/ 核心算法的基础参数估计工具"
    version = "2.0.0"
    
    @classmethod
    def get_tools(cls) -> List[Dict[str, Any]]:
        """返回工具列表"""
        return [
            {
                "name": "basic_parametric_estimation_ols",
                "handler": cls.ols_tool,
                "description": "OLS Regression Analysis"
            },
            {
                "name": "basic_parametric_estimation_mle",
                "handler": cls.mle_tool,
                "description": "Maximum Likelihood Estimation"
            },
            {
                "name": "basic_parametric_estimation_gmm",
                "handler": cls.gmm_tool,
                "description": "Generalized Method of Moments"
            }
        ]
    
    @classmethod
    def get_help_text(cls) -> str:
        """返回帮助文档"""
        return """
1. OLS Regression (basic_parametric_estimation_ols)
   - Reuses: econometrics/basic_parametric_estimation/ols/ols_model.py
   - Input: Direct (y_data + x_data) or File (file_path)
   - Formats: txt/json/csv/excel
    
2. Maximum Likelihood Estimation (basic_parametric_estimation_mle)
   - Reuses: econometrics/basic_parametric_estimation/mle/mle_model.py
   - Input: Direct (data) or File (file_path)
   - Distributions: normal, poisson, exponential
   - Formats: txt/json/csv/excel
    
3. Generalized Method of Moments (basic_parametric_estimation_gmm)
   - Reuses: econometrics/basic_parametric_estimation/gmm/gmm_model.py
   - Input: Direct (y_data + x_data) or File (file_path)
   - Fixed: j_p_value bug
   - Formats: txt/json/csv/excel
"""
    
    @staticmethod
    async def ols_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[Union[List[float], List[List[float]]]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        constant: bool = True,
        confidence_level: float = 0.95,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """OLS Regression Analysis"""
        try:
            if ctx:
                await ctx.info("Starting OLS regression...")
            
            result = ols_adapter(
                y_data=y_data,
                x_data=x_data,
                file_path=file_path,
                feature_names=feature_names,
                constant=constant,
                confidence_level=confidence_level,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("OLS regression complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def mle_tool(
        data: Optional[List[float]] = None,
        file_path: Optional[str] = None,
        distribution: str = "normal",
        initial_params: Optional[List[float]] = None,
        confidence_level: float = 0.95,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Maximum Likelihood Estimation"""
        try:
            if ctx:
                await ctx.info("Starting MLE estimation...")
            
            result = mle_adapter(
                data=data,
                file_path=file_path,
                distribution=distribution,
                initial_params=initial_params,
                confidence_level=confidence_level,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("MLE estimation complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def gmm_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[Union[List[float], List[List[float]]]] = None,
        file_path: Optional[str] = None,
        instruments: Optional[Union[List[float], List[List[float]]]] = None,
        feature_names: Optional[List[str]] = None,
        constant: bool = True,
        confidence_level: float = 0.95,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Generalized Method of Moments"""
        try:
            if ctx:
                await ctx.info("Starting GMM estimation...")
            
            result = gmm_adapter(
                y_data=y_data,
                x_data=x_data,
                file_path=file_path,
                instruments=instruments,
                feature_names=feature_names,
                constant=constant,
                confidence_level=confidence_level,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("GMM estimation complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise