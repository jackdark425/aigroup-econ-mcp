"""
工具注册器模块
简化MCP工具的注册和管理
"""

from typing import Dict, Any, Optional, List, Callable
from pydantic import Field
from typing import Annotated

from .base import with_file_support_decorator as econometric_tool


# 标准文件输入参数定义
FILE_INPUT_PARAMS = {
    "file_content": Annotated[
        Optional[str],
        Field(
            default=None,
            description="""CSV或JSON文件内容

📁 支持格式：
- CSV: 带表头的列数据，自动检测分隔符
- JSON: {"变量名": [数据], ...} 或 [{"变量1": 值, ...}, ...]

💡 使用方式：
- 提供文件内容字符串
- 系统会自动解析并识别变量
- 优先使用file_content，如果提供则忽略其他数据参数"""
        )
    ],
    "file_format": Annotated[
        str,
        Field(
            default="auto",
            description="""文件格式

可选值：
- "auto": 自动检测（默认）
- "csv": CSV格式
- "json": JSON格式"""
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
    创建工具包装器，自动添加文件输入支持
    
    Args:
        config: 工具配置对象
    
    Returns:
        包装后的工具函数
    """
    @econometric_tool(config.tool_type)
    async def tool_wrapper(ctx, **kwargs):
        """动态生成的工具包装器"""
        # 调用实际的实现函数
        return await config.impl_func(ctx, **kwargs)
    
    # 设置函数名和文档
    tool_wrapper.__name__ = config.name
    tool_wrapper.__doc__ = config.description
    
    return tool_wrapper


# 工具类型到参数映射
TOOL_TYPE_PARAMS = {
    "multi_var_dict": {
        "data": Annotated[
            Optional[Dict[str, List[float]]],
            Field(default=None, description="数据字典，格式为 {变量名: [数值列表]}")
        ]
    },
    "regression": {
        "y_data": Annotated[
            Optional[List[float]],
            Field(default=None, description="因变量数据")
        ],
        "x_data": Annotated[
            Optional[List[List[float]]],
            Field(default=None, description="自变量数据")
        ],
        "feature_names": Annotated[
            Optional[List[str]],
            Field(default=None, description="特征名称")
        ]
    },
    "single_var": {
        "data": Annotated[
            Optional[List[float]],
            Field(default=None, description="时间序列数据")
        ]
    },
    "panel": {
        "y_data": Annotated[
            Optional[List[float]],
            Field(default=None, description="因变量数据")
        ],
        "x_data": Annotated[
            Optional[List[List[float]]],
            Field(default=None, description="自变量数据")
        ],
        "entity_ids": Annotated[
            Optional[List[str]],
            Field(default=None, description="个体标识符")
        ],
        "time_periods": Annotated[
            Optional[List[str]],
            Field(default=None, description="时间标识符")
        ],
        "feature_names": Annotated[
            Optional[List[str]],
            Field(default=None, description="特征名称")
        ]
    },
    "time_series": {
        "data": Annotated[
            Optional[Dict[str, List[float]]],
            Field(default=None, description="多变量时间序列数据")
        ]
    }
}


def get_tool_params(tool_type: str, extra_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    获取工具的完整参数定义
    
    Args:
        tool_type: 工具类型
        extra_params: 额外的参数定义
    
    Returns:
        完整的参数字典
    """
    params = {}
    
    # 添加基础参数
    if tool_type in TOOL_TYPE_PARAMS:
        params.update(TOOL_TYPE_PARAMS[tool_type])
    
    # 添加文件输入参数
    params.update(FILE_INPUT_PARAMS)
    
    # 添加额外参数
    if extra_params:
        params.update(extra_params)
    
    return params