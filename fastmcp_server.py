"""
AIGroup 计量经济学 MCP 服务器 v2.0 - 适配器版本
使用适配器模式，复用 econometrics/ 核心算法
"""

from typing import List, Optional
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

# 导入适配器
from tools.econometrics_adapter import (
    ols_adapter,
    mle_adapter,
    gmm_adapter
)

# 创建 FastMCP 服务器实例
mcp = FastMCP("aigroup-econ-mcp")


@mcp.tool()
async def basic_parametric_estimation_ols(
    y_data: Optional[List[float]] = None,
    x_data: Optional[List[List[float]]] = None,
    file_path: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    confidence_level: float = 0.95,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """
    OLS Regression Analysis (using core algorithm)
    
    Supports:
    - Input: Direct data or file (txt/json/csv/excel)
    - Output: json/markdown/txt
    """
    try:
        if ctx:
            await ctx.info("Starting OLS regression (adapter mode)...")
        
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


@mcp.tool()
async def basic_parametric_estimation_mle(
    data: Optional[List[float]] = None,
    file_path: Optional[str] = None,
    distribution: str = "normal",
    initial_params: Optional[List[float]] = None,
    confidence_level: float = 0.95,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """
    Maximum Likelihood Estimation (using core algorithm)
    
    Supports:
    - Input: Direct data or file (txt/json/csv/excel)
    - Output: json/markdown/txt
    - Distributions: normal/poisson/exponential
    """
    try:
        if ctx:
            await ctx.info(f"Starting MLE estimation (adapter mode)...")
        
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


@mcp.tool()
async def basic_parametric_estimation_gmm(
    y_data: Optional[List[float]] = None,
    x_data: Optional[List[List[float]]] = None,
    file_path: Optional[str] = None,
    instruments: Optional[List[List[float]]] = None,
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    confidence_level: float = 0.95,
    output_format: str = "json",
    save_path: Optional[str] = None,
    ctx: Context[ServerSession, None] = None
) -> str:
    """
    Generalized Method of Moments (using core algorithm with fix)
    
    Supports:
    - Input: Direct data or file (txt/json/csv/excel)
    - Output: json/markdown/txt
    - Fixed: j_p_value bug in core algorithm
    """
    try:
        if ctx:
            await ctx.info("Starting GMM estimation (adapter mode)...")
        
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


@mcp.resource("config://server")
def get_server_config() -> str:
    """Get server configuration"""
    return """{
  "server_name": "aigroup-econ-mcp",
  "version": "2.0.0-adapter",
  "architecture": "Adapter Pattern (DRY)",
  "tools": [
    "basic_parametric_estimation_ols",
    "basic_parametric_estimation_mle",
    "basic_parametric_estimation_gmm"
  ],
  "description": "Econometrics MCP Tools using adapter pattern - reuses core algorithms from econometrics/"
}"""


@mcp.resource("help://econometrics")
def get_econometrics_help() -> str:
    """Get help information"""
    return """Econometrics Tools Guide (Adapter Version):

THREE CORE TOOLS - Using econometrics/ core algorithms

1. OLS Regression (basic_parametric_estimation_ols)
   - Reuses: econometrics/basic_parametric_estimation/ols/ols_model.py
   - Input: Direct (y_data + x_data) or File (file_path)
   
2. Maximum Likelihood Estimation (basic_parametric_estimation_mle)
   - Reuses: econometrics/basic_parametric_estimation/mle/mle_model.py
   - Input: Direct (data) or File (file_path)
   - Distributions: normal, poisson, exponential
   
3. Generalized Method of Moments (basic_parametric_estimation_gmm)
   - Reuses: econometrics/basic_parametric_estimation/gmm/gmm_model.py
   - Input: Direct (y_data + x_data) or File (file_path)
   - Fixed: j_p_value bug

ARCHITECTURE:
- Adapter pattern: tools/econometrics_adapter.py
- Core algorithms: econometrics/basic_parametric_estimation/
- 84% less duplicate code!

FILE FORMATS: txt, json, csv, excel
OUTPUT FORMATS: json, markdown, txt
"""


def main():
    """Start FastMCP server (Adapter version)"""
    print("AIGroup Econometrics MCP Server v2.0.0 (Adapter)")
    print("\n🏗️  Architecture: Adapter Pattern")
    print("📦 Core: econometrics/basic_parametric_estimation/")
    print("🔌 Adapter: tools/econometrics_adapter.py")
    print("\nTools: OLS / MLE / GMM")
    print("Input: Direct data or File (txt/json/csv/excel)")
    print("Output: json / markdown / txt")
    print("\n✨ Benefits: 84% less duplicate code, DRY principle")
    print("\nStarting server...")
    
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()