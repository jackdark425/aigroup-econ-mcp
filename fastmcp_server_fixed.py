"""
AIGroup 计量经济学 MCP 服务器 v2.0 - 修复版
提供 OLS、MLE、GMM 三个基础参数估计工具
支持文件输入和直接传参，多种输出格式
"""

from typing import List, Optional
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

# 导入工具实现
from tools.ols_tool import ols_regression, ols_regression_from_file, OLSResult
from tools.mle_tool import mle_estimation, mle_estimation_from_file, MLEResult
from tools.gmm_tool import gmm_estimation, gmm_estimation_from_file, GMMResult

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
    OLS Regression Analysis
    
    Supports two input methods:
    1. Direct input: provide y_data and x_data
    2. File input: provide file_path (supports txt/json/csv/excel)
    
    Args:
        y_data: Dependent variable data
        x_data: Independent variable data matrix
        file_path: Data file path
        feature_names: Feature names
        constant: Include constant term
        confidence_level: Confidence level (0.5-0.99)
        output_format: Output format (json/markdown/txt)
        save_path: Optional save path
        
    Returns:
        Formatted result string
    """
    try:
        if ctx:
            await ctx.info("Starting OLS regression...")
        
        # File input
        if file_path:
            if ctx:
                await ctx.info(f"Loading data from file: {file_path}")
            
            if output_format == "json":
                from tools.data_loader import DataLoader
                data = DataLoader.load_from_file(file_path)
                result = ols_regression(
                    y_data=data["y_data"],
                    x_data=data["x_data"],
                    feature_names=data.get("feature_names") or feature_names,
                    constant=constant,
                    confidence_level=confidence_level
                )
                import json
                return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
            else:
                return ols_regression_from_file(
                    file_path=file_path,
                    constant=constant,
                    confidence_level=confidence_level,
                    output_format=output_format,
                    save_path=save_path
                )
        
        # Direct input
        elif y_data is not None and x_data is not None:
            if ctx:
                await ctx.info("Using direct input mode")
            
            result = ols_regression(
                y_data=y_data,
                x_data=x_data,
                feature_names=feature_names,
                constant=constant,
                confidence_level=confidence_level
            )
            
            if output_format == "json":
                import json
                return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
            else:
                from tools.output_formatter import OutputFormatter
                formatted_result = OutputFormatter.format_ols_result(result, output_format)
                if save_path:
                    OutputFormatter.save_to_file(formatted_result, save_path)
                    return f"Analysis complete!\n\n{formatted_result}\n\nSaved to: {save_path}"
                return formatted_result
        
        else:
            raise ValueError("Must provide either file_path or (y_data and x_data)")
            
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
    Maximum Likelihood Estimation (MLE)
    
    Supports two input methods:
    1. Direct input: provide data
    2. File input: provide file_path (supports txt/json/csv/excel)
    
    Args:
        data: Data for estimation
        file_path: Data file path
        distribution: Distribution type (normal/poisson/exponential)
        initial_params: Initial parameter values
        confidence_level: Confidence level (0.5-0.99)
        output_format: Output format (json/markdown/txt)
        save_path: Optional save path
        
    Returns:
        Formatted result string
    """
    try:
        if ctx:
            await ctx.info(f"Starting MLE estimation ({distribution} distribution)...")
        
        # File input
        if file_path:
            if ctx:
                await ctx.info(f"Loading data from file: {file_path}")
            
            if output_format == "json":
                from tools.data_loader import MLEDataLoader
                data_dict = MLEDataLoader.load_from_file(file_path)
                result = mle_estimation(
                    data=data_dict["data"],
                    distribution=distribution,
                    initial_params=initial_params,
                    confidence_level=confidence_level
                )
                import json
                return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
            else:
                return mle_estimation_from_file(
                    file_path=file_path,
                    distribution=distribution,
                    initial_params=initial_params,
                    confidence_level=confidence_level,
                    output_format=output_format,
                    save_path=save_path
                )
        
        # Direct input
        elif data is not None:
            if ctx:
                await ctx.info("Using direct input mode")
            
            result = mle_estimation(
                data=data,
                distribution=distribution,
                initial_params=initial_params,
                confidence_level=confidence_level
            )
            
            if output_format == "json":
                import json
                return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
            else:
                from tools.output_formatter import OutputFormatter
                formatted_result = OutputFormatter.format_mle_result(result, output_format)
                if save_path:
                    OutputFormatter.save_to_file(formatted_result, save_path)
                    return f"Analysis complete!\n\n{formatted_result}\n\nSaved to: {save_path}"
                return formatted_result
        
        else:
            raise ValueError("Must provide either file_path or data")
            
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
    Generalized Method of Moments (GMM)
    
    Supports two input methods:
    1. Direct input: provide y_data and x_data
    2. File input: provide file_path (supports txt/json/csv/excel)
    
    Args:
        y_data: Dependent variable data
        x_data: Independent variable data matrix
        file_path: Data file path
        instruments: Instrumental variables
        feature_names: Feature names
        constant: Include constant term
        confidence_level: Confidence level (0.5-0.99)
        output_format: Output format (json/markdown/txt)
        save_path: Optional save path
        
    Returns:
        Formatted result string
    """
    try:
        if ctx:
            await ctx.info("Starting GMM estimation...")
        
        # File input
        if file_path:
            if ctx:
                await ctx.info(f"Loading data from file: {file_path}")
            
            if output_format == "json":
                from tools.data_loader import DataLoader
                data = DataLoader.load_from_file(file_path)
                result = gmm_estimation(
                    y_data=data["y_data"],
                    x_data=data["x_data"],
                    instruments=instruments,
                    feature_names=data.get("feature_names") or feature_names,
                    constant=constant,
                    confidence_level=confidence_level
                )
                import json
                return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
            else:
                return gmm_estimation_from_file(
                    file_path=file_path,
                    instruments=instruments,
                    constant=constant,
                    confidence_level=confidence_level,
                    output_format=output_format,
                    save_path=save_path
                )
        
        # Direct input
        elif y_data is not None and x_data is not None:
            if ctx:
                await ctx.info("Using direct input mode")
            
            result = gmm_estimation(
                y_data=y_data,
                x_data=x_data,
                instruments=instruments,
                feature_names=feature_names,
                constant=constant,
                confidence_level=confidence_level
            )
            
            if output_format == "json":
                import json
                return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
            else:
                from tools.output_formatter import OutputFormatter
                formatted_result = OutputFormatter.format_gmm_result(result, output_format)
                if save_path:
                    OutputFormatter.save_to_file(formatted_result, save_path)
                    return f"Analysis complete!\n\n{formatted_result}\n\nSaved to: {save_path}"
                return formatted_result
        
        else:
            raise ValueError("Must provide either file_path or (y_data and x_data)")
            
    except Exception as e:
        if ctx:
            await ctx.error(f"Error: {str(e)}")
        raise


@mcp.resource("config://server")
def get_server_config() -> str:
    """Get server configuration"""
    return """{
  "server_name": "aigroup-econ-mcp",
  "version": "2.0.0",
  "tools": [
    "basic_parametric_estimation_ols",
    "basic_parametric_estimation_mle",
    "basic_parametric_estimation_gmm"
  ],
  "description": "Econometrics MCP Tools - OLS/MLE/GMM with file input (txt/json/csv/excel) and multiple output formats (json/markdown/txt)"
}"""


@mcp.resource("help://econometrics")
def get_econometrics_help() -> str:
    """Get help information"""
    return """Econometrics Tools Guide:

THREE CORE TOOLS - Support file input and direct parameters

1. OLS Regression (basic_parametric_estimation_ols)
   - Input method 1: Direct (y_data + x_data)
   - Input method 2: File (file_path)
   
2. Maximum Likelihood Estimation (basic_parametric_estimation_mle)
   - Input method 1: Direct (data)
   - Input method 2: File (file_path)
   - Distributions: normal, poisson, exponential
   
3. Generalized Method of Moments (basic_parametric_estimation_gmm)
   - Input method 1: Direct (y_data + x_data)
   - Input method 2: File (file_path)
   - Supports instrumental variables

SUPPORTED FILE FORMATS
- txt, json, csv, excel (.xlsx, .xls)

OUTPUT FORMAT OPTIONS
- json: Structured JSON data (default)
- markdown: Markdown report
- txt: Plain text report

OPTIONAL FEATURES
- save_path: Save result to file
"""


def main():
    """Start FastMCP server"""
    print("AIGroup Econometrics MCP Server v2.0.0")
    print("\nTools: OLS / MLE / GMM")
    print("Input: Direct data or File (txt/json/csv/excel)")
    print("Output: json / markdown / txt")
    print("\nStarting server...")
    
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()