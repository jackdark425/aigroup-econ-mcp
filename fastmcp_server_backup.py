"""
AIGroup 计量经济学 MCP 服务器 - 基于 FastMCP 重构
提供 OLS、MLE、GMM 三个基础参数估计工具
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

# 导入工具实现
from tools.ols_tool import ols_regression, ols_regression_from_file, OLSResult
from tools.mle_tool import mle_estimation, mle_estimation_from_file, MLEResult
from tools.gmm_tool import gmm_estimation, gmm_estimation_from_file, GMMResult

# 创建 FastMCP 服务器实例
mcp = FastMCP("aigroup-econ-mcp")


# ============================================================================
# 输入模型定义
# ============================================================================

class OLSInput(BaseModel):
    """OLS回归输入参数"""
    y_data: List[float] = Field(description="因变量数据")
    x_data: List[List[float]] = Field(description="自变量数据矩阵")
    feature_names: Optional[List[str]] = Field(default=None, description="特征名称列表")
    constant: bool = Field(default=True, description="是否包含常数项")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="置信水平")


class MLEInput(BaseModel):
    """MLE估计输入参数"""
    data: List[float] = Field(description="用于估计的数据")
    distribution: str = Field(default="normal", description="分布类型: normal(正态分布), poisson(泊松分布), exponential(指数分布)")
    initial_params: Optional[List[float]] = Field(default=None, description="初始参数值")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="置信水平")


class GMMInput(BaseModel):
    """GMM估计输入参数"""
    y_data: List[float] = Field(description="因变量数据")
    x_data: List[List[float]] = Field(description="自变量数据矩阵")
    instruments: Optional[List[List[float]]] = Field(default=None, description="工具变量数据矩阵")
    feature_names: Optional[List[str]] = Field(default=None, description="特征名称列表")
    constant: bool = Field(default=True, description="是否包含常数项")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="置信水平")


# ============================================================================
# 输出模型定义
# ============================================================================

class OLSOutput(BaseModel):
    """OLS回归输出结果"""
    coefficients: List[float] = Field(description="系数估计值")
    std_errors: List[float] = Field(description="标准误")
    t_values: List[float] = Field(description="t统计量")
    p_values: List[float] = Field(description="p值")
    conf_int_lower: List[float] = Field(description="置信区间下限")
    conf_int_upper: List[float] = Field(description="置信区间上限")
    r_squared: float = Field(description="R方")
    adj_r_squared: float = Field(description="调整R方")
    f_statistic: float = Field(description="F统计量")
    f_p_value: float = Field(description="F统计量的p值")
    n_obs: int = Field(description="观测数量")
    feature_names: List[str] = Field(description="特征名称")


class MLEOutput(BaseModel):
    """MLE估计输出结果"""
    parameters: List[float] = Field(description="参数估计值")
    std_errors: List[float] = Field(description="标准误")
    conf_int_lower: List[float] = Field(description="置信区间下限")
    conf_int_upper: List[float] = Field(description="置信区间上限")
    log_likelihood: float = Field(description="对数似然值")
    aic: float = Field(description="AIC信息准则")
    bic: float = Field(description="BIC信息准则")
    convergence: bool = Field(description="收敛状态")
    n_obs: int = Field(description="观测数量")
    param_names: List[str] = Field(description="参数名称")


class GMMOutput(BaseModel):
    """GMM估计输出结果"""
    coefficients: List[float] = Field(description="系数估计值")
    std_errors: List[float] = Field(description="标准误")
    t_values: List[float] = Field(description="t统计量")
    p_values: List[float] = Field(description="p值")
    conf_int_lower: List[float] = Field(description="置信区间下限")
    conf_int_upper: List[float] = Field(description="置信区间上限")
    j_statistic: float = Field(description="J统计量")
    j_p_value: float = Field(description="J统计量的p值")
    n_obs: int = Field(description="观测数量")
    n_moments: int = Field(description="矩条件数量")
    feature_names: List[str] = Field(description="特征名称")


# ============================================================================
# 工具定义
# ============================================================================

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
    普通最小二乘法(OLS)参数估计
    
    支持两种输入方式：
    1. 直接传参：提供y_data和x_data
    2. 文件输入：提供file_path（支持txt/json/csv/excel）
    
    Args:
        y_data: 因变量数据（直接输入时使用）
        x_data: 自变量数据矩阵（直接输入时使用）
        file_path: 数据文件路径（文件输入时使用）
        feature_names: 特征名称列表
        constant: 是否包含常数项
        confidence_level: 置信水平（0.5-0.99）
        output_format: 输出格式 ("json"=返回结构化数据, "markdown"=Markdown格式, "txt"=文本格式)
        save_path: 可选的结果保存路径（仅当output_format为markdown或txt时有效）
        
    Returns:
        根据output_format返回相应格式的结果
    """
    if ctx:
        await ctx.info("开始执行OLS回归分析...")
    
    # 判断输入方式
    if file_path:
        # 文件输入
        if ctx:
            await ctx.info(f"从文件加载数据: {file_path}")
        
        if output_format == "json":
            # 如果是JSON输出，使用原有逻辑
            from tools.data_loader import DataLoader
            data = DataLoader.load_from_file(file_path)
            result: OLSResult = ols_regression(
                y_data=data["y_data"],
                x_data=data["x_data"],
                feature_names=data.get("feature_names") or feature_names,
                constant=constant,
                confidence_level=confidence_level
            )
            
            if ctx:
                await ctx.info(f"OLS回归完成: R方 = {result.r_squared:.4f}")
            
            # 返回JSON格式
            import json
            return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        else:
            # Markdown或TXT格式
            formatted_result = ols_regression_from_file(
                file_path=file_path,
                constant=constant,
                confidence_level=confidence_level,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("OLS回归分析完成")
            
            return formatted_result
    
    elif y_data is not None and x_data is not None:
        # 直接传参
        if ctx:
            await ctx.info("使用直接传参模式")
        
        result: OLSResult = ols_regression(
            y_data=y_data,
            x_data=x_data,
            feature_names=feature_names,
            constant=constant,
            confidence_level=confidence_level
        )
        
        if ctx:
            await ctx.info(f"OLS回归完成: R方 = {result.r_squared:.4f}")
        
        # 根据输出格式返回
        if output_format == "json":
            import json
            return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        else:
            from tools.output_formatter import OutputFormatter
            formatted_result = OutputFormatter.format_ols_result(result, output_format)
            if save_path:
                OutputFormatter.save_to_file(formatted_result, save_path)
                return f"分析完成！\n\n{formatted_result}\n\n结果已保存到: {save_path}"
            return formatted_result
    
    else:
        raise ValueError("必须提供file_path或(y_data和x_data)")


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
    最大似然估计(MLE)
    
    支持两种输入方式：
    1. 直接传参：提供data
    2. 文件输入：提供file_path（支持txt/json/csv/excel）
    
    Args:
        data: 用于估计的数据（直接输入时使用）
        file_path: 数据文件路径（文件输入时使用）
        distribution: 分布类型 ("normal"=正态分布, "poisson"=泊松分布, "exponential"=指数分布)
        initial_params: 初始参数值
        confidence_level: 置信水平（0.5-0.99）
        output_format: 输出格式 ("json"=返回结构化数据, "markdown"=Markdown格式, "txt"=文本格式)
        save_path: 可选的结果保存路径（仅当output_format为markdown或txt时有效）
        
    Returns:
        根据output_format返回相应格式的结果
    """
    if ctx:
        await ctx.info(f"开始执行{distribution}分布的MLE估计...")
    
    # 判断输入方式
    if file_path:
        # 文件输入
        if ctx:
            await ctx.info(f"从文件加载数据: {file_path}")
        
        if output_format == "json":
            from tools.data_loader import MLEDataLoader
            data_dict = MLEDataLoader.load_from_file(file_path)
            result: MLEResult = mle_estimation(
                data=data_dict["data"],
                distribution=distribution,
                initial_params=initial_params,
                confidence_level=confidence_level
            )
            
            if ctx:
                await ctx.info(f"MLE估计完成: 对数似然值 = {result.log_likelihood:.4f}")
            
            import json
            return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        else:
            formatted_result = mle_estimation_from_file(
                file_path=file_path,
                distribution=distribution,
                initial_params=initial_params,
                confidence_level=confidence_level,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("MLE估计完成")
            
            return formatted_result
    
    elif data is not None:
        # 直接传参
        if ctx:
            await ctx.info("使用直接传参模式")
        
        result: MLEResult = mle_estimation(
            data=data,
            distribution=distribution,
            initial_params=initial_params,
            confidence_level=confidence_level
        )
        
        if ctx:
            await ctx.info(f"MLE估计完成: 对数似然值 = {result.log_likelihood:.4f}")
        
        if output_format == "json":
            import json
            return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        else:
            from tools.output_formatter import OutputFormatter
            formatted_result = OutputFormatter.format_mle_result(result, output_format)
            if save_path:
                OutputFormatter.save_to_file(formatted_result, save_path)
                return f"分析完成！\n\n{formatted_result}\n\n结果已保存到: {save_path}"
            return formatted_result
    
    else:
        raise ValueError("必须提供file_path或data")


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
    广义矩估计(GMM)
    
    支持两种输入方式：
    1. 直接传参：提供y_data和x_data
    2. 文件输入：提供file_path（支持txt/json/csv/excel）
    
    Args:
        y_data: 因变量数据（直接输入时使用）
        x_data: 自变量数据矩阵（直接输入时使用）
        file_path: 数据文件路径（文件输入时使用）
        instruments: 工具变量数据矩阵（可选）
        feature_names: 特征名称列表
        constant: 是否包含常数项
        confidence_level: 置信水平（0.5-0.99）
        output_format: 输出格式 ("json"=返回结构化数据, "markdown"=Markdown格式, "txt"=文本格式)
        save_path: 可选的结果保存路径（仅当output_format为markdown或txt时有效）
        
    Returns:
        根据output_format返回相应格式的结果
    """
    if ctx:
        await ctx.info("开始执行GMM估计...")
    
    # 判断输入方式
    if file_path:
        # 文件输入
        if ctx:
            await ctx.info(f"从文件加载数据: {file_path}")
        
        if output_format == "json":
            from tools.data_loader import DataLoader
            data = DataLoader.load_from_file(file_path)
            result: GMMResult = gmm_estimation(
                y_data=data["y_data"],
                x_data=data["x_data"],
                instruments=instruments,
                feature_names=data.get("feature_names") or feature_names,
                constant=constant,
                confidence_level=confidence_level
            )
            
            if ctx:
                await ctx.info(f"GMM估计完成: J统计量 = {result.j_statistic:.4f}")
            
            import json
            return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        else:
            formatted_result = gmm_estimation_from_file(
                file_path=file_path,
                instruments=instruments,
                constant=constant,
                confidence_level=confidence_level,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("GMM估计完成")
            
            return formatted_result
    
    elif y_data is not None and x_data is not None:
        # 直接传参
        if ctx:
            await ctx.info("使用直接传参模式")
        
        result: GMMResult = gmm_estimation(
            y_data=y_data,
            x_data=x_data,
            instruments=instruments,
            feature_names=feature_names,
            constant=constant,
            confidence_level=confidence_level
        )
        
        if ctx:
            await ctx.info(f"GMM估计完成: J统计量 = {result.j_statistic:.4f}")
        
        if output_format == "json":
            import json
            return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        else:
            from tools.output_formatter import OutputFormatter
            formatted_result = OutputFormatter.format_gmm_result(result, output_format)
            if save_path:
                OutputFormatter.save_to_file(formatted_result, save_path)
                return f"分析完成！\n\n{formatted_result}\n\n结果已保存到: {save_path}"
            return formatted_result
    
    else:
        raise ValueError("必须提供file_path或(y_data和x_data)")


# ============================================================================
# 资源定义
# ============================================================================

@mcp.resource("config://server")
def get_server_config() -> str:
    """获取服务器配置信息"""
    return """{
  "server_name": "aigroup-econ-mcp",
  "version": "2.0.0",
  "tools": [
    "basic_parametric_estimation_ols",
    "basic_parametric_estimation_mle",
    "basic_parametric_estimation_gmm"
  ],
  "description": "专业计量经济学MCP工具 - 提供OLS、MLE、GMM三种基础参数估计方法，每个工具支持文件输入（txt/json/csv/excel）或直接传参，输出格式可选（json/markdown/txt）"
}"""


@mcp.resource("help://econometrics")
def get_econometrics_help() -> str:
    """获取计量经济学帮助信息"""
    return """计量经济学工具使用指南：

【三大核心工具 - 支持文件输入或直接传参】

1. OLS回归 (basic_parametric_estimation_ols)
   - 用于线性回归分析
   - 输入方式1：直接传参 (y_data + x_data)
   - 输入方式2：文件输入 (file_path)
   - 支持常数项和置信区间

2. 最大似然估计 (basic_parametric_estimation_mle)
   - 用于参数分布估计
   - 输入方式1：直接传参 (data)
   - 输入方式2：文件输入 (file_path)
   - 支持正态分布、泊松分布、指数分布
   - 提供AIC、BIC信息准则

3. 广义矩估计 (basic_parametric_estimation_gmm)
   - 用于处理内生性问题
   - 输入方式1：直接传参 (y_data + x_data)
   - 输入方式2：文件输入 (file_path)
   - 支持工具变量
   - 提供J统计量进行过度识别检验

【支持的文件格式】
- txt (文本文件，空格或制表符分隔)
- json (JSON格式数据)
- csv (逗号分隔值文件)
- excel (.xlsx, .xls)

【输出格式选项】
- json: 返回结构化JSON数据（默认）
- markdown: Markdown格式的分析报告
- txt: 纯文本格式的分析报告

【数据格式要求】
- 所有数据应为数值类型
- 自变量矩阵应为二维数组
- 样本量建议大于30
- 文件格式：第一列为因变量，其余列为自变量

【可选功能】
- save_path: 保存结果到指定文件（仅当output_format为markdown或txt时有效）
"""


# ============================================================================
# 提示定义
# ============================================================================

@mcp.prompt()
def regression_analysis_guide() -> str:
    """回归分析指导提示"""
    return """请帮助用户进行回归分析：

1. 首先了解用户的数据类型和研究问题
2. 推荐合适的回归方法：
   - 线性关系：OLS回归
   - 计数数据：泊松回归(MLE)
   - 持续时间数据：指数回归(MLE)
   - 内生性问题：GMM估计
3. 解释回归结果的含义
4. 提供统计显著性判断
5. 给出实际应用建议

请根据用户的具体需求提供专业指导。"""


@mcp.prompt()
def econometric_method_selection() -> str:
    """计量经济学方法选择指导"""
    return """请帮助用户选择合适的计量经济学方法：

考虑以下因素：
1. 数据类型：横截面、时间序列、面板数据
2. 因变量类型：连续、二元、计数、持续时间
3. 研究问题：因果关系、预测、描述性分析
4. 数据特征：样本量、变量分布、异常值
5. 模型假设：线性、正态性、同方差性

请根据这些因素推荐最合适的方法并解释原因。"""


# ============================================================================
# 主程序
# ============================================================================

def main():
    """启动 FastMCP 服务器"""
    print("启动 AIGroup 计量经济学 MCP 服务器 v2.0.0...")
    print("\n【三大核心工具】")
    print("  1. basic_parametric_estimation_ols - OLS回归")
    print("  2. basic_parametric_estimation_mle - 最大似然估计")
    print("  3. basic_parametric_estimation_gmm - 广义矩估计")
    print("\n【输入方式】每个工具支持两种输入方式：")
    print("  ✓ 直接传参: 提供数据数组 (y_data/x_data 或 data)")
    print("  ✓ 文件输入: 提供文件路径 (file_path)")
    print("\n【支持的文件格式】")
    print("  ✓ TXT  - 文本文件")
    print("  ✓ JSON - JSON格式")
    print("  ✓ CSV  - 逗号分隔")
    print("  ✓ Excel - .xlsx/.xls")
    print("\n【输出格式选项】(output_format参数)")
    print("  ✓ json - 结构化JSON数据（默认）")
    print("  ✓ markdown - Markdown格式报告")
    print("  ✓ txt - 纯文本格式报告")
    print("\n【可选功能】")
    print("  ✓ save_path - 保存结果到文件")
    print("\n使用方式:")
    print("  python fastmcp_server.py")
    print("  或")
    print("  uv run mcp dev fastmcp_server.py")
    
    # 运行服务器
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()