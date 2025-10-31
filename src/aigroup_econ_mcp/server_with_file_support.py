"""
AIGroup 计量经济学 MCP 服务器 - 支持CSV文件路径输入
使用最新的MCP特性提供专业计量经济学分析工具
"""

from typing import List, Dict, Any, Optional, Annotated, Union
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa import stattools
from scipy import stats
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from mcp.types import CallToolResult, TextContent


# 数据模型定义
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


# 应用上下文
@dataclass
class AppContext:
    """应用上下文，包含共享资源"""
    config: Dict[str, Any]
    version: str = "0.2.0"


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """服务器生命周期管理"""
    config = {
        "max_sample_size": 10000,
        "default_significance_level": 0.05,
        "supported_tests": ["t_test", "f_test", "chi_square", "adf"],
        "data_types": ["cross_section", "time_series", "panel"]
    }
    try:
        yield AppContext(config=config, version="0.2.0")
    finally:
        pass


# 创建MCP服务器实例
mcp = FastMCP(
    name="aigroup-econ-mcp",
    instructions="Econometrics MCP Server with CSV file path support",
    lifespan=lifespan
)


# 辅助函数：加载CSV文件
async def load_data_from_path(file_path: str, ctx: Context) -> Dict[str, List[float]]:
    """从CSV文件路径加载数据"""
    await ctx.info(f"正在从文件加载数据: {file_path}")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 转换为字典格式
        data = {col: df[col].tolist() for col in df.columns}
        
        await ctx.info(f"✅ 文件加载成功：{len(df.columns)}个变量，{len(df)}个观测")
        return data
        
    except FileNotFoundError:
        raise ValueError(f"文件不存在: {file_path}")
    except Exception as e:
        raise ValueError(f"文件读取失败: {str(e)}")


@mcp.tool()
async def descriptive_statistics(
    ctx: Context[ServerSession, AppContext],
    data: Annotated[
        Union[Dict[str, List[float]], str],
        Field(
            description="""数据输入，支持两种格式：

📊 格式1：数据字典
{
    "GDP增长率": [3.2, 2.8, 3.5, 2.9],
    "通货膨胀率": [2.1, 2.3, 1.9, 2.4],
    "失业率": [4.5, 4.2, 4.0, 4.3]
}

📁 格式2：CSV文件路径（推荐）
"d:/data/economic_data.csv"
或
"./test_data.csv"

CSV文件要求：
- 第一行为变量名（表头）
- 后续行为数值数据
- 所有列必须为数值类型"""
        )
    ]
) -> CallToolResult:
    """计算描述性统计量
    
    📊 功能说明：
    对输入数据进行全面的描述性统计分析，包括集中趋势、离散程度、分布形状等指标。
    
    💡 使用场景：
    - 初步了解数据的分布特征
    - 检查数据质量和异常值
    - 为后续建模提供基础信息
    """
    try:
        # 检测输入类型并加载数据
        if isinstance(data, str):
            # 文件路径输入
            data_dict = await load_data_from_path(data, ctx)
        else:
            # 字典输入
            data_dict = data
        
        await ctx.info(f"开始计算描述性统计，处理 {len(data_dict)} 个变量")
        
        # 数据验证
        if not data_dict:
            raise ValueError("数据不能为空")
        
        df = pd.DataFrame(data_dict)
        
        # 基础统计量
        result = DescriptiveStatsResult(
            count=len(df),
            mean=df.mean().mean(),
            std=df.std().mean(),
            min=df.min().min(),
            max=df.max().max(),
            median=df.median().mean(),
            skewness=df.skew().mean(),
            kurtosis=df.kurtosis().mean()
        )
        
        # 计算相关系数矩阵
        correlation_matrix = df.corr().round(4)
        
        await ctx.info(f"描述性统计计算完成，样本大小: {len(df)}")
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"描述性统计结果：\n"
                         f"样本数: {result.count}\n"
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
async def correlation_analysis(
    ctx: Context[ServerSession, AppContext],
    data: Annotated[
        Union[Dict[str, List[float]], str],
        Field(
            description="""数据输入，支持两种格式：

📊 格式1：数据字典
{
    "销售额": [12000, 13500, 11800, 14200],
    "广告支出": [800, 900, 750, 1000],
    "价格": [99, 95, 102, 98]
}

📁 格式2：CSV文件路径
"d:/data/marketing_data.csv"

要求：
- 至少包含2个变量
- 所有变量的数据点数量必须相同"""
        )
    ],
    method: Annotated[
        str,
        Field(
            default="pearson",
            description="相关系数类型：pearson/spearman/kendall"
        )
    ] = "pearson"
) -> CallToolResult:
    """变量间相关性分析"""
    try:
        # 检测输入类型并加载数据
        if isinstance(data, str):
            data_dict = await load_data_from_path(data, ctx)
        else:
            data_dict = data
        
        await ctx.info(f"开始相关性分析: {method}")
        
        # 数据验证
        if not data_dict:
            raise ValueError("数据不能为空")
        if len(data_dict) < 2:
            raise ValueError("至少需要2个变量进行相关性分析")
        
        df = pd.DataFrame(data_dict)
        correlation_matrix = df.corr(method=method)
        
        await ctx.info("相关性分析完成")
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"{method.title()}相关系数矩阵：\n{correlation_matrix.round(4).to_string()}"
                )
            ]
        )
        
    except Exception as e:
        await ctx.error(f"相关性分析出错: {str(e)}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"错误: {str(e)}")],
            isError=True
        )


def create_mcp_server() -> FastMCP:
    """创建并返回MCP服务器实例"""
    return mcp