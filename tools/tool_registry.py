"""
工具注册表模块
提供MCP工具注册和参数管理功能
"""

from typing import Dict, Any, Optional, List, Callable
from pydantic import Field
from typing import Annotated

from econometrics.decorators import with_file_support_decorator as econometric_tool


# 文件输入参数定义
FILE_INPUT_PARAMS = {
    "file_content": Annotated[
        Optional[str],
        Field(
            default=None,
            description="""文件内容字符串，支持CSV/JSON/TXT格式

使用说明:
- CSV: 逗号分隔值格式
- JSON: {"列名": [值1, 值2, ...]} 或 [{"列名1": 值1, ...}, ...]
- TXT: 多种文本格式
  * 单列数据: 每行一个数值
  * 多列数据: 空格或制表符分隔
  * 键值对: 列名: 值1 值2 值3

注意事项:
- 不支持base64编码
- 文件内容优先于文件路径
- 如果同时提供file_content和file_path，使用file_content
- 文件格式自动检测，也可手动指定.csv/.json/.txt

示例:
CSV: "1,2\\n1.2,3.4\\n2.3,4.5"
JSON: "{\\"x\\": [1,2,3], \\"y\\": [4,5,6]}"
TXT: "1.5\\n2.3\\n3.1"
TXT: "x y\\n1 2\\n3 4"
TXT: "x: 1 2 3\\ny: 4 5 6"
"""
        )
    ],
    "file_format": Annotated[
        str,
        Field(
            default="auto",
            description="""
文件格式类型

选项:
- "auto": 自动检测 - 根据内容自动判断格式
- "csv": CSV格式 - 逗号分隔值
- "json": JSON格式 - JavaScript对象表示法
- "txt": TXT格式 - 纯文本格式

检测顺序:
1. JSON格式优先
2. CSV格式
3. TXT格式
4. 根据文件扩展名(.csv/.json/.txt)

注意事项:
- 推荐使用"auto"
- 如果自动检测失败，请手动指定格式
- TXT格式支持多种分隔符"""
        )
    ]
}


class ToolConfig:
    """工具配置类"""
    
    def __init__(
        self,
        name: str,
        impl_func: Callable,
        tool_type: str,
        description: str = "",
        extra_params: Dict[str, Any] = None
    ):
        self.name = name
        self.impl_func = impl_func
        self.tool_type = tool_type
        self.description = description
        self.extra_params = extra_params or {}


def create_tool_wrapper(config: ToolConfig):
    """
    创建工具包装器
    
    Args:
        config: 工具配置
    
    Returns:
        包装后的工具函数
    """
    @econometric_tool(config.tool_type)
    async def tool_wrapper(ctx, **kwargs):
        """工具包装器实现"""
        # 调用原始实现函数
        return await config.impl_func(ctx, **kwargs)
    
    # 设置函数属性
    tool_wrapper.__name__ = config.name
    tool_wrapper.__doc__ = config.description
    
    return tool_wrapper


# 工具类型参数定义
TOOL_TYPE_PARAMS = {
    "multi_var_dict": {
        "data": Annotated[
            Optional[Dict[str, List[float]]],
            Field(
                default=None, 
                description="""
多变量数据字典

使用说明:
- 字典格式，键为变量名，值为数值列表
- 所有变量必须有相同长度
- 格式: {"变量1": [值1, 值2, ...], "变量2": [值1, 值2, ...]}

注意事项:
- 变量名不能重复
- 数据长度必须一致
- 支持缺失值处理
- 至少需要1个变量

示例:
{"GDP": [100.5, 102.3, 104.1], "CPI": [2.1, 2.3, 2.2]}

适用场景:
- 多变量统计分析
- 相关性分析
- 主成分分析"""
            )
        ]
    },
    "regression": {
        "y_data": Annotated[
            Optional[List[float]],
            Field(
                default=None, 
                description="""
因变量数据

使用说明:
- 数值列表格式
- 必须与x_data有相同长度
- 支持缺失值

注意事项:
- 不能包含NaN值
- 长度必须与x_data一致
- 至少需要n+2个观测值(n为自变量个数)

示例:
[12.5, 13.2, 14.1, 13.8, 15.0]  # 因变量数据

适用场景:
- OLS回归分析
- 多元回归
- 预测建模"""
            )
        ],
        "x_data": Annotated[
            Optional[List[List[float]]],
            Field(
                default=None, 
                description="""
自变量数据矩阵

使用说明:
- 二维列表格式
- 每行代表一个观测值，每列代表一个自变量
- 格式: [[x1_1, x2_1, ...], [x1_2, x2_2, ...], ...]

注意事项:
- 必须与y_data长度一致
- 支持多个自变量
- 可以包含常数项
- 至少需要2个观测值

示例:
[[100, 50, 3],    # 观测值1: 自变量1=100, 自变量2=50, 自变量3=3
 [120, 48, 3],    # 观测值2: 自变量1=120, 自变量2=48, 自变量3=3
 [110, 52, 4]]    # 观测值3: 自变量1=110, 自变量2=52, 自变量3=4

适用场景:
- OLS回归分析
- 多元回归
- 特征工程"""
            )
        ],
        "feature_names": Annotated[
            Optional[List[str]],
            Field(
                default=None, 
                description="""
自变量名称列表

使用说明:
- 字符串列表格式
- 必须与x_data列数一致
- 用于结果展示和解释

注意事项:
["价格", "收入", "广告支出", "季节因子"]

适用场景:
- 结果解释
- 特征重要性分析
- 模型报告
- 如果不提供，自动生成x1, x2, x3...

示例:
- 提高结果可读性
- 便于模型解释
- 特征选择"""
            )
        ]
    },
    "single_var": {
        "data": Annotated[
            Optional[List[float]],
            Field(
                default=None, 
                description="""
单变量时间序列数据

使用说明:
- 数值列表格式
- 时间顺序排列
- 支持缺失值处理

注意事项:
- 数据不能为空
- 至少需要5个观测值
- ARIMA模型需要30+个观测值

示例:
[100.5, 102.3, 101.8, 103.5, 104.2, 103.8, 105.1]  # 时间序列数据

适用场景:
- 描述性统计
- ADF/KPSS平稳性检验
- ACF/PACF自相关分析
- ARIMA/GARCH建模
- 时间序列预测"""
            )
        ]
    },
    "panel": {
        "y_data": Annotated[
            Optional[List[float]],
            Field(
                default=None, 
                description="""
面板数据因变量

使用说明:
- 数值列表格式
- 数据长度 = 实体数 × 时间期数
- 按实体-时间顺序排列

注意事项:
1. [实体1-时间1, 实体1-时间2, ...]
2. [实体2-时间1, 实体2-时间2, ...]
3. entity_ids和time_periods必须对应

示例 3实体2时期:
[1000, 1100, 800, 900, 1200, 1300]  # 因变量数据
实体1: 2020年=1000, 2021年=1100
实体2: 2020年=800,  2021年=900
实体3: 2020年=1200, 2021年=1300

适用场景:
- 面板回归分析
- 固定效应模型
- 随机效应模型"""
            )
        ],
        "x_data": Annotated[
            Optional[List[List[float]]],
            Field(
                default=None, 
                description="""
面板数据自变量矩阵

使用说明:
- 二维列表格式
- 数据长度 = 实体数 × 时间期数
- 每行代表一个实体-时间观测值
- 必须与y_data长度一致

注意事项:
- y_data长度必须一致
- 支持多个自变量
- 可以包含常数项

示例 3实体2时期2自变量:
[[50, 100],   # 实体1-2020: 自变量1=50, 自变量2=100
 [52, 110],   # 实体1-2021: 自变量1=52, 自变量2=110
 [40, 80],    # 实体2-2020
 [42, 90],    # 实体2-2021
 [60, 150],   # 实体3-2020
 [62, 160]]   # 实体3-2021

适用场景:
- 面板回归分析
- 固定效应模型"""
            )
        ],
        "entity_ids": Annotated[
            Optional[List[str]],
            Field(
                default=None, 
                description="""
实体标识符列表

使用说明:
- 字符串列表格式
- 标识每个观测值所属的实体
- 长度必须等于y_data长度

注意事项:
- 必须提供
- 必须与y_data和x_data对应
- 支持重复标识符

示例 3实体2时期:
["A", "A", "B", "B", "C", "C"]

["1", "1", "2", "2", "3", "3"]

适用场景:
- 实体效应建模
- 固定效应检验
- Hausman检验

注意事项:
- 必须提供ID标识
- 支持字符串和数字ID"""
            )
        ],
        "time_periods": Annotated[
            Optional[List[str]],
            Field(
                default=None, 
                description="""
时间期数标识符列表

使用说明:
- 字符串列表格式
- 标识每个观测值所属的时间期
- 长度必须等于y_data长度

注意事项:
- 必须提供
- 必须与y_data和x_data对应
- 支持重复标识符

示例 3实体2时期:
["2020", "2021", "2020", "2021", "2020", "2021"]

["2020-01", "2020-02", "2020-01", "2020-02", "2020-01", "2020-02"]

适用场景:
- 时间效应建模
- 时间固定效应
- 面板数据平衡性检验

注意事项:
- 必须提供时间标识
- 支持多种时间格式"""
            )
        ],
        "feature_names": Annotated[
            Optional[List[str]],
            Field(
                default=None, 
                description="""与regression类型相同

使用说明:
- 字符串列表格式
- 必须与x_data列数一致

注意事项:
["价格", "收入", "广告支出", "季节因子"]

适用场景:
- 结果解释
- 特征重要性分析"""
            )
        ]
    },
    "time_series": {
        "data": Annotated[
            Optional[Dict[str, List[float]]],
            Field(
                default=None, 
                description="""
多变量时间序列数据

使用说明:
- 字典格式，键为变量名，值为时间序列
- 所有变量必须有相同长度
- 支持多个时间序列变量

注意事项:
- 至少需要2个变量用于VAR/VECM
- 与single_var类型不同
- 数据必须按时间顺序排列
- VAR模型需要40+个观测值
- VECM模型需要60+个观测值

示例:
{
    "GDP": [100.5, 102.3, 104.1, 105.8],     # GDP时间序列
    "CPI": [2.1, 2.3, 2.2, 2.4],             # 消费者价格指数
    "失业率": [3.5, 3.6, 3.4, 3.7]             # 失业率
}

适用场景:
- VAR模型分析
- VECM协整分析
- 多变量时间序列建模
- Granger因果检验
- 脉冲响应分析

注意事项:
- 变量间可能存在协整关系
- 需要足够的时间序列长度
- VAR模型要求平稳性
- VECM模型处理非平稳序列"""
            )
        ]
    }
}


def get_tool_params(tool_type: str, extra_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    获取工具参数配置
    
    Args:
        tool_type: 工具类型
        extra_params: 额外参数
    
    Returns:
        参数配置字典
    """
    params = {}
    
    # 基础参数
    if tool_type in TOOL_TYPE_PARAMS:
        params.update(TOOL_TYPE_PARAMS[tool_type])
    
    # 文件输入参数
    params.update(FILE_INPUT_PARAMS)
    
    # 额外参数
    if extra_params:
        params.update(extra_params)
    
    return params