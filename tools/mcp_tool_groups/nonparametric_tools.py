"""
非参数与半参数方法工具组
包含核回归、分位数回归、样条回归和GAM
"""

from typing import List, Optional, Dict, Any
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from ..mcp_tools_registry import ToolGroup
from ..nonparametric_adapter import (
    kernel_regression_adapter,
    quantile_regression_adapter,
    spline_regression_adapter,
    gam_adapter
)


class NonparametricTools(ToolGroup):
    """非参数与半参数方法工具组"""
    
    name = "NONPARAMETRIC & SEMIPARAMETRIC METHODS"
    description = "非参数与半参数分析工具"
    version = "1.0.0"
    
    @classmethod
    def get_tools(cls) -> List[Dict[str, Any]]:
        """返回工具列表"""
        return [
            {
                "name": "nonparametric_kernel_regression",
                "handler": cls.kernel_regression_tool,
                "description": "Kernel Regression (Nonparametric)"
            },
            {
                "name": "nonparametric_quantile_regression",
                "handler": cls.quantile_regression_tool,
                "description": "Quantile Regression"
            },
            {
                "name": "nonparametric_spline_regression",
                "handler": cls.spline_regression_tool,
                "description": "Spline Regression"
            },
            {
                "name": "nonparametric_gam_model",
                "handler": cls.gam_tool,
                "description": "Generalized Additive Model (GAM)"
            }
        ]
    
    @classmethod
    def get_help_text(cls) -> str:
        """返回帮助文档"""
        return """
非参数与半参数方法工具组 - 4个工具

1. Kernel Regression (nonparametric_kernel_regression)
   - 核回归估计
   - 支持核函数: Gaussian, Epanechnikov, Uniform, Triangular, Biweight
   - 带宽选择: 交叉验证, AIC, 正态参考
   - 基于: statsmodels.nonparametric
   
2. Quantile Regression (nonparametric_quantile_regression)
   - 分位数回归
   - 分析条件分位数
   - 稳健于异常值
   - 基于: statsmodels.regression.quantile_regression

3. Spline Regression (nonparametric_spline_regression)
   - 样条回归
   - 灵活的非线性拟合
   - 基于: sklearn.preprocessing

4. Generalized Additive Model (nonparametric_gam_model)
   - 广义可加模型
   - 多个平滑函数的加和
   - 基于: pygam
"""
    
    @staticmethod
    async def kernel_regression_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        kernel_type: str = "gaussian",
        bandwidth: Optional[List[float]] = None,
        bandwidth_method: str = "cv_ls",
        variable_type: Optional[str] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """核回归分析"""
        try:
            if ctx:
                await ctx.info(f"Starting kernel regression ({kernel_type} kernel)...")
            
            result = kernel_regression_adapter(
                y_data=y_data,
                x_data=x_data,
                file_path=file_path,
                kernel_type=kernel_type,
                bandwidth=bandwidth,
                bandwidth_method=bandwidth_method,
                variable_type=variable_type,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Kernel regression complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def quantile_regression_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        quantile: float = 0.5,
        feature_names: Optional[List[str]] = None,
        confidence_level: float = 0.95,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """分位数回归分析"""
        try:
            if ctx:
                await ctx.info(f"Starting quantile regression (τ={quantile})...")
            
            result = quantile_regression_adapter(
                y_data=y_data,
                x_data=x_data,
                file_path=file_path,
                quantile=quantile,
                feature_names=feature_names,
                confidence_level=confidence_level,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Quantile regression complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def spline_regression_tool(
        y_data: List[float],
        x_data: List[float],
        n_knots: int = 5,
        degree: int = 3,
        knots: str = "uniform",
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """样条回归"""
        try:
            if ctx:
                await ctx.info(f"Starting spline regression (degree={degree}, knots={n_knots})...")
            
            result = spline_regression_adapter(
                y_data=y_data,
                x_data=x_data,
                n_knots=n_knots,
                degree=degree,
                knots=knots,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Spline regression complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def gam_tool(
        y_data: List[float],
        x_data: List[List[float]],
        problem_type: str = "regression",
        n_splines: int = 10,
        lam: float = 0.6,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """广义可加模型(GAM)"""
        try:
            if ctx:
                await ctx.info(f"Starting GAM model ({problem_type})...")
            
            result = gam_adapter(
                y_data=y_data,
                x_data=x_data,
                problem_type=problem_type,
                n_splines=n_splines,
                lam=lam,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("GAM model complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise