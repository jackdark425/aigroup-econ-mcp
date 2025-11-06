"""
空间计量经济学工具组
包含空间权重、空间自相关检验和空间回归模型
"""

from typing import List, Optional, Dict, Any, Tuple
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from ..mcp_tools_registry import ToolGroup
from ..spatial_econometrics_adapter import (
    spatial_weights_adapter,
    morans_i_adapter,
    gearys_c_adapter,
    local_moran_adapter,
    spatial_regression_adapter,
    gwr_adapter
)


class SpatialEconometricsTools(ToolGroup):
    """空间计量经济学工具组"""
    
    name = "SPATIAL ECONOMETRICS"
    description = "空间计量经济学分析工具"
    version = "1.0.0"
    
    @classmethod
    def get_tools(cls) -> List[Dict[str, Any]]:
        """返回工具列表"""
        return [
            {
                "name": "spatial_weights_matrix",
                "handler": cls.spatial_weights_tool,
                "description": "Spatial Weights Matrix Construction"
            },
            {
                "name": "spatial_morans_i_test",
                "handler": cls.morans_i_tool,
                "description": "Moran's I Spatial Autocorrelation Test"
            },
            {
                "name": "spatial_gearys_c_test",
                "handler": cls.gearys_c_tool,
                "description": "Geary's C Spatial Autocorrelation Test"
            },
            {
                "name": "spatial_local_moran_lisa",
                "handler": cls.local_moran_tool,
                "description": "Local Moran's I (LISA) Analysis"
            },
            {
                "name": "spatial_regression_model",
                "handler": cls.spatial_regression_tool,
                "description": "Spatial Regression Models (SAR/SEM/SDM)"
            },
            {
                "name": "spatial_gwr_model",
                "handler": cls.gwr_tool,
                "description": "Geographically Weighted Regression (GWR)"
            }
        ]
    
    @classmethod
    def get_help_text(cls) -> str:
        """返回帮助文档"""
        return """
空间计量经济学工具组 - 6个工具

1. Spatial Weights Matrix (spatial_weights_matrix)
   - 空间权重矩阵构建
   - 支持类型: Queen, Rook, KNN, Distance Band, Kernel
   - 基于: libpysal库
   
2. Moran's I Test (spatial_morans_i_test)
   - Moran's I空间自相关检验
   - 全局空间聚集性分析
   - 基于: esda库

3. Geary's C Test (spatial_gearys_c_test)
   - Geary's C空间自相关检验
   - 另一种全局空间相关性度量
   - 基于: esda库

4. Local Moran's I - LISA (spatial_local_moran_lisa)
   - 局部空间自相关分析
   - 识别HH, LL, HL, LH聚类
   - 基于: esda库
   
5. Spatial Regression (spatial_regression_model)
   - 空间滞后模型 (SAR)
   - 空间误差模型 (SEM)
   - 空间杜宾模型 (SDM)
   - 基于: spreg库

6. Geographically Weighted Regression (spatial_gwr_model)
   - 地理加权回归
   - 局部回归系数估计
   - 捕捉空间异质性
   - 基于: mgwr库
"""
    
    @staticmethod
    async def spatial_weights_tool(
        coordinates: Optional[List[Tuple[float, float]]] = None,
        adjacency_matrix: Optional[List[List[int]]] = None,
        weight_type: str = "queen",
        k: int = 4,
        distance_threshold: Optional[float] = None,
        bandwidth: Optional[float] = None,
        kernel_type: str = "triangular",
        row_standardize: bool = True,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """空间权重矩阵构建"""
        try:
            if ctx:
                await ctx.info(f"Starting spatial weights construction ({weight_type})...")
            
            result = spatial_weights_adapter(
                coordinates=coordinates,
                adjacency_matrix=adjacency_matrix,
                weight_type=weight_type,
                k=k,
                distance_threshold=distance_threshold,
                bandwidth=bandwidth,
                kernel_type=kernel_type,
                row_standardize=row_standardize,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Spatial weights construction complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def morans_i_tool(
        values: List[float],
        neighbors: dict,
        weights: Optional[dict] = None,
        permutations: int = 999,
        two_tailed: bool = True,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Moran's I空间自相关检验"""
        try:
            if ctx:
                await ctx.info("Starting Moran's I test...")
            
            result = morans_i_adapter(
                values=values,
                neighbors=neighbors,
                weights=weights,
                permutations=permutations,
                two_tailed=two_tailed,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Moran's I test complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def gearys_c_tool(
        values: List[float],
        neighbors: dict,
        weights: Optional[dict] = None,
        permutations: int = 999,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Geary's C空间自相关检验"""
        try:
            if ctx:
                await ctx.info("Starting Geary's C test...")
            
            result = gearys_c_adapter(
                values=values,
                neighbors=neighbors,
                weights=weights,
                permutations=permutations,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Geary's C test complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def local_moran_tool(
        values: List[float],
        neighbors: dict,
        weights: Optional[dict] = None,
        permutations: int = 999,
        significance_level: float = 0.05,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """局部Moran's I (LISA) 分析"""
        try:
            if ctx:
                await ctx.info("Starting Local Moran's I (LISA) analysis...")
            
            result = local_moran_adapter(
                values=values,
                neighbors=neighbors,
                weights=weights,
                permutations=permutations,
                significance_level=significance_level,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Local Moran's I analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def spatial_regression_tool(
        y_data: List[float],
        x_data: List[List[float]],
        neighbors: dict,
        weights: Optional[dict] = None,
        feature_names: Optional[List[str]] = None,
        model_type: str = "sar",
        method: str = "ml",
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """空间回归模型（SAR/SEM/SDM）"""
        try:
            if ctx:
                await ctx.info(f"Starting {model_type.upper()} spatial regression ({method.upper()})...")
            
            result = spatial_regression_adapter(
                y_data=y_data,
                x_data=x_data,
                neighbors=neighbors,
                weights=weights,
                feature_names=feature_names,
                model_type=model_type,
                method=method,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info(f"{model_type.upper()} spatial regression complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def gwr_tool(
        y_data: List[float],
        x_data: List[List[float]],
        coordinates: List[Tuple[float, float]],
        feature_names: Optional[List[str]] = None,
        kernel_type: str = "bisquare",
        bandwidth: Optional[float] = None,
        fixed: bool = False,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """地理加权回归 (GWR)"""
        try:
            if ctx:
                await ctx.info(f"Starting GWR analysis ({kernel_type} kernel)...")
            
            result = gwr_adapter(
                y_data=y_data,
                x_data=x_data,
                coordinates=coordinates,
                feature_names=feature_names,
                kernel_type=kernel_type,
                bandwidth=bandwidth,
                fixed=fixed,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("GWR analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise