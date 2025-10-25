"""
AIGroup 计量经济学 MCP 服务器
使用最新的MCP特性提供专业计量经济学分析工具
"""

from typing import List, Dict, Any, Optional, Annotated
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP, Context, Icon
from mcp.server.session import ServerSession
from mcp.types import CallToolResult, TextContent


# 数据模型定义 - 使用Pydantic实现结构化输出
class DescriptiveStatsResult(BaseModel):
    """描述性统计结果"""
    count: int = Field(description="样本数量")
    mean: float = Field(description="均值")
    std: float = Field(description="标准差")
    min: float = Field(description="最小值")
    max: float = Field(description="最大值")
    median: float = Field(description="中位数")
    skewness: float = Field(description="偏度")
    kurtosis: float = Field(description="峰度")


class OLSRegressionResult(BaseModel):
    """OLS回归分析结果"""
    rsquared: float = Field(description="R²")
    rsquared_adj: float = Field(description="调整R²")
    f_statistic: float = Field(description="F统计量")
    f_pvalue: float = Field(description="F检验p值")
    aic: float = Field(description="AIC信息准则")
    bic: float = Field(description="BIC信息准则")
    coefficients: Dict[str, Dict[str, float]] = Field(description="回归系数详情")


class HypothesisTestResult(BaseModel):
    """假设检验结果"""
    test_type: str = Field(description="检验类型")
    statistic: float = Field(description="检验统计量")
    p_value: float = Field(description="p值")
    significant: bool = Field(description="是否显著(5%水平)")
    confidence_interval: Optional[List[float]] = Field(default=None, description="置信区间")


class TimeSeriesStatsResult(BaseModel):
    """时间序列统计结果"""
    adf_statistic: float = Field(description="ADF检验统计量")
    adf_pvalue: float = Field(description="ADF检验p值")
    stationary: bool = Field(description="是否平稳")
    acf: List[float] = Field(description="自相关函数")
    pacf: List[float] = Field(description="偏自相关函数")


# 应用上下文
@dataclass
class AppContext:
    """应用上下文，包含共享资源"""
    config: Dict[str, Any]
    version: str = "0.1.0"


# 服务器图标
server_icon = Icon(
    src="https://img.icons8.com/fluency/48/bar-chart.png",
    mimeType="image/png",
    sizes=["48x48"]
)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """服务器生命周期管理"""
    # 启动时初始化资源
    config = {
        "max_sample_size": 10000,
        "default_significance_level": 0.05,
        "supported_tests": ["t_test", "f_test", "chi_square", "adf"],
        "data_types": ["cross_section", "time_series", "panel"]
    }

    try:
        yield AppContext(config=config, version="0.1.0")
    finally:
        # 清理资源
        pass


# 创建MCP服务器实例
mcp = FastMCP(
    name="aigroup-econ-mcp",
    instructions="专业计量经济学MCP工具，提供数据分析、回归分析、假设检验等功能",
    lifespan=lifespan
)


@mcp.resource("dataset://sample/{dataset_name}")
def get_sample_dataset(dataset_name: str) -> str:
    """获取示例数据集"""
    datasets = {
        "economic_growth": """
        GDP增长率,通货膨胀率,失业率,投资率
        3.2,2.1,4.5,15.2
        2.8,2.3,4.2,14.8
        3.5,1.9,4.0,16.1
        2.9,2.4,4.3,15.5
        """,
        "stock_returns": """
        股票A,股票B,股票C
        0.02,-0.01,0.015
        -0.015,0.025,-0.008
        0.018,-0.005,0.012
        """,
        "time_series": """
        时间,销售额,广告支出
        2023-01,12000,800
        2023-02,13500,900
        2023-03,11800,750
        2023-04,14200,1000
        """
    }

    if dataset_name not in datasets:
        return f"可用数据集: {', '.join(datasets.keys())}"

    return datasets[dataset_name]


@mcp.prompt(title="经济数据分析")
def economic_analysis_prompt(data_description: str, analysis_type: str = "descriptive") -> str:
    """经济数据分析提示模板"""
    prompts = {
        "descriptive": "请对以下经济数据进行描述性统计分析：",
        "regression": "请对以下经济数据进行回归分析，识别关键影响因素：",
        "hypothesis": "请对以下经济数据进行假设检验，验证研究假设：",
        "time_series": "请对以下时间序列数据进行分析，检查平稳性和相关性："
    }

    return f"{prompts.get(analysis_type, prompts['descriptive'])}\n\n数据描述：{data_description}"


@mcp.tool()
async def descriptive_statistics(
    ctx: Context[ServerSession, AppContext],
    data: Dict[str, List[float]]
) -> Annotated[CallToolResult, DescriptiveStatsResult]:
    """计算描述性统计量

    Args:
        data: 字典格式的数据，键为变量名，值为数值列表
        ctx: MCP上下文对象
    """
    await ctx.info(f"开始计算描述性统计，处理 {len(data)} 个变量")

    try:
        df = pd.DataFrame(data)

        # 基础统计量
        result = DescriptiveStatsResult(
            count=len(df),
            mean=df.mean().iloc[0],  # 简化示例，实际应返回所有变量
            std=df.std().iloc[0],
            min=df.min().iloc[0],
            max=df.max().iloc[0],
            median=df.median().iloc[0],
            skewness=df.skew().iloc[0],
            kurtosis=df.kurtosis().iloc[0]
        )

        # 计算相关系数矩阵
        correlation_matrix = df.corr().round(4)

        await ctx.info(f"描述性统计计算完成，样本大小: {len(df)}")

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"描述性统计结果：\n"
                         f"均值: {result.mean:.4f}\n"
                         f"标准差: {result.std:.4f}\n"
                         f"最小值: {result.min:.4f}\n"
                         f"最大值: {result.max:.4f}\n"
                         f"中位数: {result.median:.4f}\n"
                         f"偏度: {result.skewness:.4f}\n"
                         f"峰度: {result.kurtosis:.4f}\n\n"
                         f"相关系数矩阵：\n{correlation_matrix.to_string()}"
                )
            ],
            structuredContent=result.model_dump()
        )

    except Exception as e:
        await ctx.error(f"计算描述性统计时出错: {str(e)}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"错误: {str(e)}")],
            isError=True
        )


@mcp.tool()
async def ols_regression(
    ctx: Context[ServerSession, AppContext],
    y_data: List[float],
    x_data: List[List[float]],
    feature_names: Optional[List[str]] = None
) -> Annotated[CallToolResult, OLSRegressionResult]:
    """执行OLS回归分析

    Args:
        y_data: 因变量数据
        x_data: 自变量数据，每行一个观测
        feature_names: 自变量名称
        ctx: MCP上下文对象
    """
    await ctx.info(f"开始OLS回归分析，样本大小: {len(y_data)}，自变量数量: {len(x_data[0]) if x_data else 0}")

    try:
        # 准备数据
        X = np.column_stack(x_data) if x_data else np.ones((len(y_data), 1))
        if x_data:  # 只有当有自变量时才添加常数项
            X = sm.add_constant(X)

        # 拟合模型
        model = sm.OLS(y_data, X).fit()

        # 构建结果
        result = OLSRegressionResult(
            rsquared=model.rsquared,
            rsquared_adj=model.rsquared_adj,
            f_statistic=model.fvalue,
            f_pvalue=model.f_pvalue,
            aic=model.aic,
            bic=model.bic,
            coefficients={}
        )

        # 添加系数详情
        conf_int = model.conf_int()
        for i, coef in enumerate(model.params):
            var_name = "const" if i == 0 else feature_names[i-1] if feature_names else f"x{i}"
            result.coefficients[var_name] = {
                "coef": coef,
                "std_err": model.bse[i],
                "t_value": model.tvalues[i],
                "p_value": model.pvalues[i],
                "ci_lower": conf_int[i][0],
                "ci_upper": conf_int[i][1]
            }

        await ctx.info("OLS回归分析完成")

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"OLS回归分析结果：\n"
                         f"R² = {result.rsquared:.4f}\n"
                         f"调整R² = {result.rsquared_adj:.4f}\n"
                         f"F统计量 = {result.f_statistic:.4f} (p = {result.f_pvalue:.4f})\n"
                         f"AIC = {result.aic:.2f}, BIC = {result.bic:.2f}\n\n"
                         f"回归系数：\n{model.summary().tables[1]}"
                )
            ],
            structuredContent=result.model_dump()
        )

    except Exception as e:
        await ctx.error(f"OLS回归分析出错: {str(e)}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"错误: {str(e)}")],
            isError=True
        )


@mcp.tool()
async def hypothesis_testing(
    ctx: Context[ServerSession, AppContext],
    data1: List[float],
    data2: Optional[List[float]] = None,
    test_type: str = "t_test"
) -> Annotated[CallToolResult, HypothesisTestResult]:
    """执行假设检验

    Args:
        data1: 第一组数据
        data2: 第二组数据（可选）
        test_type: 检验类型
        ctx: MCP上下文对象
    """
    await ctx.info(f"开始假设检验: {test_type}")

    try:
        if test_type == "t_test":
            if data2 is None:
                # 单样本t检验
                result = stats.ttest_1samp(data1, 0)
                ci = stats.t.interval(0.95, len(data1)-1, loc=np.mean(data1), scale=stats.sem(data1))
            else:
                # 双样本t检验
                result = stats.ttest_ind(data1, data2)
                ci = None  # 双样本t检验不计算置信区间

            test_result = HypothesisTestResult(
                test_type=test_type,
                statistic=result.statistic,
                p_value=result.pvalue,
                significant=result.pvalue < 0.05,
                confidence_interval=list(ci) if ci else None
            )

        elif test_type == "adf":
            # ADF单位根检验
            result = statsmodels.tsa.stattools.adfuller(data1)
            test_result = HypothesisTestResult(
                test_type="adf",
                statistic=result[0],
                p_value=result[1],
                significant=result[1] < 0.05,
                confidence_interval=None
            )
        else:
            raise ValueError(f"不支持的检验类型: {test_type}")

        await ctx.info(f"假设检验完成: {test_type}")

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"{test_type.upper()}检验结果：\n"
                         f"检验统计量 = {test_result.statistic:.4f}\n"
                         f"p值 = {test_result.p_value:.4f}\n"
                         f"{'显著' if test_result.significant else '不显著'} (5%水平)\n"
                         f"{f'95%置信区间: [{test_result.confidence_interval[0]:.4f}, {test_result.confidence_interval[1]:.4f}]' if test_result.confidence_interval else ''}"
                )
            ],
            structuredContent=test_result.model_dump()
        )

    except Exception as e:
        await ctx.error(f"假设检验出错: {str(e)}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"错误: {str(e)}")],
            isError=True
        )


@mcp.tool()
async def time_series_analysis(
    ctx: Context[ServerSession, AppContext],
    data: List[float]
) -> Annotated[CallToolResult, TimeSeriesStatsResult]:
    """时间序列分析

    Args:
        data: 时间序列数据
        ctx: MCP上下文对象
    """
    await ctx.info(f"开始时间序列分析，数据点数量: {len(data)}")

    try:
        # ADF单位根检验
        adf_result = statsmodels.tsa.stattools.adfuller(data)

        # 自相关和偏自相关函数
        acf_values = statsmodels.tsa.stattools.acf(data, nlags=min(20, len(data)-1))
        pacf_values = statsmodels.tsa.stattools.pacf(data, nlags=min(20, len(data)-1))

        result = TimeSeriesStatsResult(
            adf_statistic=adf_result[0],
            adf_pvalue=adf_result[1],
            stationary=adf_result[1] < 0.05,
            acf=acf_values.tolist(),
            pacf=pacf_values.tolist()
        )

        await ctx.info("时间序列分析完成")

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"时间序列分析结果：\n"
                         f"ADF检验统计量 = {result.adf_statistic:.4f}\n"
                         f"ADF检验p值 = {result.adf_pvalue:.4f}\n"
                         f"{'平稳' if result.stationary else '非平稳'}序列\n"
                         f"ACF前5阶: {result.acf[:5]}\n"
                         f"PACF前5阶: {result.pacf[:5]}"
                )
            ],
            structuredContent=result.model_dump()
        )

    except Exception as e:
        await ctx.error(f"时间序列分析出错: {str(e)}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"错误: {str(e)}")],
            isError=True
        )


@mcp.tool()
async def correlation_analysis(
    ctx: Context[ServerSession, AppContext],
    data: Dict[str, List[float]],
    method: str = "pearson"
) -> str:
    """相关性分析

    Args:
        data: 变量数据
        method: 相关系数类型
        ctx: MCP上下文对象
    """
    await ctx.info(f"开始相关性分析: {method}")

    try:
        df = pd.DataFrame(data)
        correlation_matrix = df.corr(method=method)

        await ctx.info("相关性分析完成")

        return f"{method.title()}相关系数矩阵：\n{correlation_matrix.round(4).to_string()}"

    except Exception as e:
        await ctx.error(f"相关性分析出错: {str(e)}")
        return f"错误: {str(e)}"


def create_mcp_server() -> FastMCP:
    """创建并返回MCP服务器实例"""
    return mcp