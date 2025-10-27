
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
from statsmodels.tsa import stattools
from scipy import stats
from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP, Context
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


# 服务器图标（已移除，因为MCP库不再支持Icon类）


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
    instructions="Econometrics MCP Server - Provides data analysis, regression analysis, hypothesis testing and more",
    lifespan=lifespan
)


@mcp.resource("dataset://sample/{dataset_name}")
def get_sample_dataset(dataset_name: str) -> str:
    """Get sample dataset"""
    datasets = {
        "economic_growth": """
        GDP Growth,Inflation Rate,Unemployment Rate,Investment Rate
        3.2,2.1,4.5,15.2
        2.8,2.3,4.2,14.8
        3.5,1.9,4.0,16.1
        2.9,2.4,4.3,15.5
        """,
        "stock_returns": """
        Stock A,Stock B,Stock C
        0.02,-0.01,0.015
        -0.015,0.025,-0.008
        0.018,-0.005,0.012
        """,
        "time_series": """
        Date,Sales,Advertising Expense
        2023-01,12000,800
        2023-02,13500,900
        2023-03,11800,750
        2023-04,14200,1000
        """
    }

    if dataset_name not in datasets:
        return f"Available datasets: {', '.join(datasets.keys())}"

    return datasets[dataset_name]


@mcp.prompt(title="Economic Data Analysis")
def economic_analysis_prompt(data_description: str, analysis_type: str = "descriptive") -> str:
    """Economic data analysis prompt template"""
    prompts = {
        "descriptive": "Please perform descriptive statistical analysis on the following economic data:",
        "regression": "Please perform regression analysis on the following economic data to identify key factors:",
        "hypothesis": "Please perform hypothesis testing on the following economic data to verify research assumptions:",
        "time_series": "Please analyze the following time series data to check stationarity and correlation:"
    }

    return f"{prompts.get(analysis_type, prompts['descriptive'])}\n\nData description: {data_description}"


@mcp.tool()
async def descriptive_statistics(
    ctx: Context[ServerSession, AppContext],
    data: Annotated[
        Dict[str, List[float]],
        Field(
            description="""数据字典，格式为 {变量名: [数值列表]}
            
示例格式：
{
    "GDP增长率": [3.2, 2.8, 3.5, 2.9],
    "通货膨胀率": [2.1, 2.3, 1.9, 2.4],
    "失业率": [4.5, 4.2, 4.0, 4.3]
}

要求：
- 至少包含一个变量
- 每个变量的数据点数量应相同
- 数值必须为浮点数或整数
- 建议样本量 >= 30 以获得可靠的统计推断"""
        )
    ]
) -> CallToolResult:
    """计算描述性统计量
    
    📊 功能说明：
    对输入数据进行全面的描述性统计分析，包括集中趋势、离散程度、分布形状等指标。
    
    📈 输出指标：
    - 样本数量 (count)
    - 均值 (mean)：数据的平均水平
    - 标准差 (std)：数据的离散程度
    - 最小值/最大值 (min/max)：数据的取值范围
    - 中位数 (median)：数据的中间值，对异常值不敏感
    - 偏度 (skewness)：分布的对称性，0表示对称，>0右偏，<0左偏
    - 峰度 (kurtosis)：分布的尖峭程度，0表示正态分布
    - 相关系数矩阵：变量间的线性相关关系
    
    💡 使用场景：
    - 初步了解数据的分布特征
    - 检查数据质量和异常值
    - 为后续建模提供基础信息
    - 比较不同变量的统计特征
    
    ⚠️ 注意事项：
    - 偏度绝对值 > 1 表示数据明显偏斜，可能需要转换
    - 峰度绝对值 > 3 表示尖峭或扁平分布
    - 相关系数 > 0.8 表示强相关，可能存在多重共线性

    Args:
        data: 数据字典，键为变量名，值为数值列表
        ctx: MCP上下文对象
    """
    await ctx.info(f"开始计算描述性统计，处理 {len(data)} 个变量")

    try:
        # 数据验证
        if not data:
            raise ValueError("数据不能为空")
        
        df = pd.DataFrame(data)
        
        # 检查数据一致性
        if len(df.columns) == 0:
            raise ValueError("至少需要一个变量")
        
        # 基础统计量 - 修复：返回所有变量的综合统计
        result = DescriptiveStatsResult(
            count=len(df),
            mean=df.mean().mean(),  # 所有变量的均值
            std=df.std().mean(),    # 所有变量的标准差均值
            min=df.min().min(),     # 所有变量的最小值
            max=df.max().max(),     # 所有变量的最大值
            median=df.median().mean(),  # 所有变量的中位数均值
            skewness=df.skew().mean(),  # 所有变量的偏度均值
            kurtosis=df.kurtosis().mean()  # 所有变量的峰度均值
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
    y_data: Annotated[
        List[float],
        Field(
            description="""因变量数据（被解释变量/响应变量）
            
示例：[12000, 13500, 11800, 14200, 15100]

要求：
- 必须为数值列表
- 长度必须与自变量观测数量一致
- 不能包含缺失值（NaN）
- 建议样本量 >= 30"""
        )
    ],
    x_data: Annotated[
        List[List[float]],
        Field(
            description="""自变量数据（解释变量/预测变量），二维列表格式
            
示例格式（3个观测，2个自变量）：
[
    [800, 5.2],    # 第1个观测的自变量值
    [900, 5.8],    # 第2个观测的自变量值
    [750, 4.9]     # 第3个观测的自变量值
]

要求：
- 外层列表：每个元素代表一个观测
- 内层列表：该观测的所有自变量值
- 所有观测的自变量数量必须相同
- 观测数量必须与y_data长度一致
- 自变量数量建议 < 观测数量/10（避免过拟合）"""
        )
    ],
    feature_names: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            description="""自变量名称列表（可选）
            
示例：["广告支出", "价格指数"]

说明：
- 如果不提供，将自动命名为 x1, x2, x3...
- 名称数量必须与自变量数量一致
- 建议使用有意义的名称以便解释结果"""
        )
    ] = None
) -> CallToolResult:
    """执行普通最小二乘法(OLS)回归分析
    
    📊 功能说明：
    使用最小二乘法估计线性回归模型，分析因变量与自变量之间的线性关系。
    模型形式：Y = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ + ε
    
    📈 输出指标：
    - R²：决定系数，取值0-1，衡量模型拟合优度
    - 调整R²：考虑自变量数量的修正R²
    - F统计量及p值：检验模型整体显著性
    - AIC/BIC：信息准则，用于模型比较，越小越好
    - 回归系数：每个自变量的估计值、标准误、t统计量、p值、置信区间
    
    💡 使用场景：
    - 因果关系分析（如广告支出对销售额的影响）
    - 预测建模（如根据经济指标预测GDP）
    - 控制变量分析（如研究教育回报率时控制工作经验）
    - 假设检验（如检验某变量是否对结果有显著影响）
    
    ⚠️ 注意事项：
    - R² > 0.7 表示拟合良好，但需警惕过拟合
    - p值 < 0.05 表示该系数在5%水平显著
    - 需检查残差的正态性、同方差性和独立性假设
    - 自变量间高度相关（相关系数>0.8）可能导致多重共线性问题
    - 样本量过小可能导致不可靠的估计结果

    Args:
        y_data: 因变量数据
        x_data: 自变量数据，每行一个观测
        feature_names: 自变量名称（可选）
        ctx: MCP上下文对象
    """
    await ctx.info(f"开始OLS回归分析，样本大小: {len(y_data)}，自变量数量: {len(x_data[0]) if x_data else 0}")

    try:
        # 数据验证
        if not y_data:
            raise ValueError("因变量数据不能为空")
        if not x_data:
            raise ValueError("自变量数据不能为空")
        if len(y_data) != len(x_data):
            raise ValueError(f"因变量和自变量的观测数量不一致: y_data={len(y_data)}, x_data={len(x_data)}")
        
        # 准备数据
        X = np.array(x_data)
        y = np.array(y_data)
        
        # 添加常数项
        X_with_const = sm.add_constant(X)

        # 拟合模型
        model = sm.OLS(y, X_with_const).fit()

        # 构建系数字典
        conf_int = model.conf_int()
        coefficients = {}
        
        # 修复：正确处理feature_names为None的情况
        if feature_names is None:
            feature_names = [f"x{i+1}" for i in range(X.shape[1])]
        elif len(feature_names) != X.shape[1]:
            await ctx.warning(f"提供的feature_names数量({len(feature_names)})与自变量数量({X.shape[1]})不匹配，使用默认命名")
            feature_names = [f"x{i+1}" for i in range(X.shape[1])]

        for i, coef in enumerate(model.params):
            if i == 0:
                var_name = "const"
            else:
                var_name = feature_names[i-1]
            
            coefficients[var_name] = {
                "coef": float(coef),  # 转换numpy.float64为float
                "std_err": float(model.bse[i]),
                "t_value": float(model.tvalues[i]),
                "p_value": float(model.pvalues[i]),
                "ci_lower": float(conf_int[i][0]),
                "ci_upper": float(conf_int[i][1])
            }
        
        # 构建结果
        result = OLSRegressionResult(
            rsquared=float(model.rsquared),
            rsquared_adj=float(model.rsquared_adj),
            f_statistic=float(model.fvalue),
            f_pvalue=float(model.f_pvalue),
            aic=float(model.aic),
            bic=float(model.bic),
            coefficients=coefficients
        )

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
    data1: Annotated[
        List[float],
        Field(
            description="""第一组数据或单一样本数据
            
示例：[3.2, 2.8, 3.5, 2.9, 3.1, 2.7, 3.3]

要求：
- 必须为数值列表
- 不能包含缺失值
- 建议样本量 >= 30（大样本）
- t检验要求数据近似正态分布"""
        )
    ],
    data2: Annotated[
        Optional[List[float]],
        Field(
            default=None,
            description="""第二组数据（可选，用于双样本检验）
            
示例：[2.5, 2.9, 2.3, 2.6, 2.8]

说明：
- 仅在双样本t检验时需要提供
- 单样本t检验时保持为None
- 两组数据可以
有不同的样本量
- ADF检验不需要第二组数据"""
        )
    ] = None,
    test_type: Annotated[
        str,
        Field(
            default="t_test",
            description="""假设检验类型

可选值：
- "t_test": t检验（默认）
  * 单样本t检验：检验样本均值是否等于0（data2=None）
  * 双样本t检验：检验两组样本均值是否相等（提供data2）
  
- "adf": 增强迪基-富勒检验（Augmented Dickey-Fuller Test）
  * 用于检验时间序列的平稳性
  * 原假设：存在单位根（非平稳）
  * p<0.05 拒绝原假设，序列平稳

使用建议：
- 比较均值差异 → 使用 t_test
- 检验时间序列平稳性 → 使用 adf"""
        )
    ] = "t_test"
) -> CallToolResult:
    """执行统计假设检验
    
    📊 功能说明：
    对数据进行统计假设检验，判断样本是否支持某个统计假设。
    
    📈 检验类型详解：
    
    1️⃣ t检验 (t_test)：
       - 单样本：H₀: μ = 0 vs H₁: μ ≠ 0
       - 双样本：H₀: μ₁ = μ₂ vs H₁: μ₁ ≠ μ₂
       - 适用于小样本（n<30）且数据近似正态分布
    
    2️⃣ ADF检验 (adf)：
       - H₀: 序列存在单位根（非平稳）
       - H₁: 序列不存在单位根（平稳）
       - 用于时间序列分析前的平稳性检验
    
    📊 输出指标：
    - 检验统计量：用于判断是否拒绝原假设
    - p值：显著性水平，<0.05表示在5%水平显著
    - 是否显著：基于5%显著性水平的判断
    - 置信区间：参数的可能取值范围（仅t检验）
    
    💡 使用场景：
    - 检验新药是否有效（单样本t检验）
    - 比较两种教学方法的效果差异（双样本t检验）
    - 检验股价序列是否平稳（ADF检验）
    - 验证经济理论假说（如购买力平价理论）
    
    ⚠️ 注意事项：
    - p值 < 0.05：拒绝原假设（结果显著）
    - p值 >= 0.05：不能拒绝原假设（结果不显著）
    - t检验要求数据近似正态分布
    - 小样本(<30)时t检验结果可能不可靠
    - ADF检验中p<0.05表示序列平稳（拒绝非平稳假设）

    Args:
        data1: 第一组数据
        data2: 第二组数据（可选，用于双样本检验）
        test_type: 检验类型（t_test或adf）
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
            result = stattools.adfuller(data1)
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
    data: Annotated[
        List[float],
        Field(
            description="""时间序列数据（按时间顺序排列）
            
示例格式：
[12000, 13500, 11800, 14200, 15100, 14800, 16200, 15900]
# 表示连续8期的观测值，如月度销售额

要求：
- 必须按时间顺序排列（从早到晚）
- 建议至少30个观测点以获得可靠结果
- 数据应等间隔采样（如日度、月度、季度）
- 不能包含缺失值
- 数据量越大，ACF/PACF分析越准确

应用示例：
- 股票价格序列
- GDP季度数据
- 月度销售额
- 日均气温数据"""
        )
    ]
) -> CallToolResult:
    """时间序列统计分析
    
    📊 功能说明：
    对时间序列数据进行全面的统计分析，包括平稳性检验和自相关分析。
    
    📈 分析内容：
    
    1️⃣ ADF单位根检验（Augmented Dickey-Fuller Test）：
       - 检验序列是否平稳
       - H₀: 存在单位根（序列非平稳）
       - p < 0.05：拒绝原假设，序列平稳
       - 平稳性是时间序列建模的基础
    
    2️⃣ 自相关函数（ACF）：
       - 衡量序列与其滞后值之间的相关性
       - 用于识别MA模型的阶数
       - 指数衰减→AR过程；q阶截尾→MA(q)过程
    
    3️⃣ 偏自相关函数（PACF）：
       - 剔除中间滞后项影响后的相关性
       - 用于识别AR模型的阶数
       - p阶截尾→AR(p)过程
    
    📊 输出指标：
    - ADF统计量：越负越可能平稳
    - ADF p值：<0.05表示序列平稳
    - 平稳性判断：基于5%显著性水平
    - ACF值：前20阶（或更少）的自相关系数
    - PACF值：前20阶（或更少）的偏自相关系数
    
    💡 使用场景：
    - ARIMA建模前的平稳性检验
    - 识别合适的时间序列模型（AR、MA、ARMA）
    - 检测季节性和趋势
    - 评估序列的记忆性和持续性
    
    ⚠️ 注意事项：
    - 非平稳序列需要差分或变换后才能建模
    - ACF和PACF应结合使用以识别模型类型
    - 数据点太少（<30）可能导致不可靠的结果
    - 强烈的季节性可能影响ACF/PACF的解读
    - 建议同时观察ACF/PACF图形以获得更好的直观理解
    
    📖 结果解读：
    - ADF p值 < 0.05 + ACF快速衰减 → 平稳序列，可直接建模
    - ADF p值 >= 0.05 → 非平稳序列，需要差分处理
    - PACF在p阶截尾 → 考虑AR(p)模型
    - ACF在q阶截尾 → 考虑MA(q)模型
    - ACF和PACF都衰减 → 考虑ARMA模型

    Args:
        data: 时间序列数据（按时间顺序）
        ctx: MCP上下文对象
    """
    await ctx.info(f"开始时间序列分析，数据点数量: {len(data)}")

    try:
        # 数据验证
        if not data:
            raise ValueError("时间序列数据不能为空")
        if len(data) < 5:
            raise ValueError("时间序列数据至少需要5个观测点")
        
        # ADF单位根检验
        adf_result = stattools.adfuller(data)

        # 自相关和偏自相关函数
        # 修复：安全计算nlags，避免PACF计算失败
        max_nlags = min(20, len(data) - 1, len(data) // 2)
        if max_nlags < 1:
            max_nlags = 1  # 确保至少计算1阶
        
        # 修复：使用try-except处理ACF/PACF计算可能失败的情况
        try:
            acf_values = stattools.acf(data, nlags=max_nlags)
            pacf_values = stattools.pacf(data, nlags=max_nlags)
        except Exception as acf_error:
            await ctx.warning(f"ACF/PACF计算遇到问题: {str(acf_error)}，使用简化计算")
            # 使用更简单的计算方法
            acf_values = np.zeros(max_nlags + 1)
            pacf_values = np.zeros(max_nlags + 1)
            acf_values[0] = 1.0  # 0阶自相关总是1
            pacf_values[0] = 1.0  # 0阶偏自相关总是1

        # 转换numpy类型为Python原生类型
        result = TimeSeriesStatsResult(
            adf_statistic=float(adf_result[0]),
            adf_pvalue=float(adf_result[1]),
            stationary=bool(adf_result[1] < 0.05),
            acf=[float(x) for x in acf_values.tolist()],
            pacf=[float(x) for x in pacf_values.tolist()]
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
    data: Annotated[
        Dict[str, List[float]],
        Field(
            description="""多变量数据字典
            
示例格式：
{
    "销售额": [12000, 13500, 11800, 14200],
    "广告支出": [800, 900, 750, 1000],
    "价格": [99, 95, 102, 98],
    "竞争对手数量": [3, 3, 4, 3]
}

要求：
- 至少包含2个变量
- 所有变量的数据点数量必须相同
- 建议样本量 >= 30
- 数值不能包含缺失值

应用：
- 探索变量间的关联关系
- 识别潜在的多重共线性
- 为回归分析筛选变量"""
        )
    ],
    method: Annotated[
        str,
        Field(
            default="pearson",
            description="""相关系数计算方法

可选值：
- "pearson": 皮尔逊相关系数（默认）
  * 衡量线性相关关系
  * 取值范围：-1到1
  * 要求：数据近似正态分布
  * 对异常值敏感
  
- "spearman": 斯皮尔曼秩相关系数
  * 衡量单调相关关系（不要求线性）
  * 基于数据的秩次
  * 对异常值不敏感
  * 适用于非正态分布
  
- "kendall": 肯德尔τ相关系数
  * 衡量一致性程度
  * 更稳健但计算较慢
  * 适用于小样本和有序数据

选择建议：
- 数据正态分布 + 关注线性关系 → pearson
- 数据有异常值或非正态 → spearman
- 有序分类数据 → kendall"""
        )
    ] = "pearson"
) -> CallToolResult:
    """变量间相关性分析
    
    📊 功能说明：
    计算多个变量之间的相关系数矩阵，揭示变量间的关联关系强度和方向。
    
    📈 相关系数解读：
    - 相关系数范围：-1 到 +1
    - |r| = 0.0-0.3：弱相关或无相关
    - |r| = 0.3-0.7：中等程度相关
    - |r| = 0.7-1.0：强相关
    - r > 0：正相关（同向变化）
    - r < 0：负相关（反向变化）
    - r = 0：无线性相关
    
    💡 使用场景：
    - 探索性数据分析（EDA）
    - 回归分析前的变量筛选
    - 识别多重共线性问题
    - 构建投资组合（寻找低相关资产）
    - 因子分析和主成分分析的前置步骤
    
    ⚠️ 注意事项：
    - 相关≠因果：高相关不代表因果关系
    - 皮尔逊相关仅衡量线性关系，可能错过非线性关系
    - 异常值会显著影响皮尔逊相关系数
    - 回归分析中，自变量间相关系数>0.8可能导致多重共线性
    - 小样本(<30)的相关系数可能不稳定
    
    📖 实际应用示例：
    - 营销分析：广告支出与销售额的相关性
    - 金融分析：不同股票收益率之间的相关性
    - 经济研究：GDP增长与失业率的关系
    - 多重共线性检测：回归模型中自变量间的相关性

    Args:
        data: 变量数据字典
        method: 相关系数类型（pearson/spearman/kendall）
        ctx: MCP上下文对象
    """
    await ctx.info(f"开始相关性分析: {method}")

    try:
        # 数据验证
        if not data:
            raise ValueError("数据不能为空")
        if len(data) < 2:
            raise ValueError("至少需要2个变量进行相关性分析")
        
        df = pd.DataFrame(data)
        correlation_matrix = df.corr(method=method)

        await ctx.info("相关性分析完成")

        # 修复：返回正确的CallToolResult类型
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