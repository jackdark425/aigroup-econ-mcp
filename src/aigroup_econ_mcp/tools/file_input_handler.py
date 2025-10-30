"""
文件输入处理组件
提供统一的文件输入处理接口，支持所有工具
"""

from typing import Dict, List, Any, Optional, Callable
from functools import wraps
from .file_parser import FileParser


class FileInputHandler:
    """
    文件输入处理组件
    
    使用组件模式，为任何工具函数添加文件输入支持
    """
    
    @staticmethod
    def process_input(
        file_content: Optional[str],
        file_format: str,
        tool_type: str,
        data_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理文件输入并转换为工具参数
        
        Args:
            file_content: 文件内容
            file_format: 文件格式
            tool_type: 工具类型
            data_params: 当前数据参数
        
        Returns:
            更新后的参数字典
        """
        # 如果没有文件输入，直接返回原参数
        if file_content is None:
            return data_params
        
        # 解析文件
        parsed = FileParser.parse_file_content(file_content, file_format)
        
        # 转换为工具格式
        converted = FileParser.convert_to_tool_format(parsed, tool_type)
        
        # 合并参数（文件数据优先）
        result = {**data_params, **converted}
        
        return result
    
    @staticmethod
    def with_file_support(tool_type: str):
        """
        装饰器：为工具函数添加文件输入支持
        
        Args:
            tool_type: 工具类型（single_var, multi_var_dict, regression, panel等）
        
        Returns:
            装饰后的函数
        
        使用示例：
            @FileInputHandler.with_file_support('regression')
            async def my_regression_tool(y_data, x_data, file_content=None, file_format='auto'):
                # 函数会自动处理file_content并填充y_data和x_data
                pass
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 提取文件相关参数
                file_content = kwargs.get('file_content')
                file_format = kwargs.get('file_format', 'auto')
                
                if file_content is not None:
                    # 处理文件输入
                    processed = FileInputHandler.process_input(
                        file_content=file_content,
                        file_format=file_format,
                        tool_type=tool_type,
                        data_params=kwargs
                    )
                    
                    # 更新kwargs
                    kwargs.update(processed)
                
                # 调用原函数
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator


class FileInputMixin:
    """
    文件输入混入类
    
    为类提供文件输入处理能力
    """
    
    def parse_file_input(
        self,
        file_content: Optional[str],
        file_format: str = "auto"
    ) -> Optional[Dict[str, Any]]:
        """解析文件输入"""
        if file_content is None:
            return None
        return FileParser.parse_file_content(file_content, file_format)
    
    def convert_for_tool(
        self,
        parsed_data: Dict[str, Any],
        tool_type: str
    ) -> Dict[str, Any]:
        """转换为工具格式"""
        return FileParser.convert_to_tool_format(parsed_data, tool_type)
    
    def get_recommendations(
        self,
        parsed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """获取工具推荐"""
        return FileParser.auto_detect_tool_params(parsed_data)


def create_file_params(
    description: str = "CSV或JSON文件内容"
) -> Dict[str, Any]:
    """
    创建标准的文件输入参数定义
    
    Args:
        description: 参数描述
    
    Returns:
        参数定义字典，可直接用于Field()
    """
    return {
        "file_content": {
            "default": None,
            "description": f"""{description}
            
📁 支持格式：
- CSV: 带表头的列数据，自动检测分隔符
- JSON: {{"变量名": [数据], ...}} 或 [{{"变量1": 值, ...}}, ...]

💡 使用方式：
- 提供文件内容字符串（可以是base64编码）
- 系统会自动解析并识别变量
- 优先使用file_content，如果提供则忽略其他数据参数"""
        },
        "file_format": {
            "default": "auto",
            "description": """文件格式
            
可选值：
- "auto": 自动检测（默认）
- "csv": CSV格式
- "json": JSON格式"""
        }
    }


class UnifiedFileInput:
    """
    统一文件输入接口
    
    所有工具通过此类统一处理文件输入
    """
    
    @staticmethod
    async def handle(
        ctx: Any,
        file_content: Optional[str],
        file_format: str,
        tool_type: str,
        original_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        统一处理文件输入
        
        Args:
            ctx: MCP上下文
            file_content: 文件内容
            file_format: 文件格式
            tool_type: 工具类型
            original_params: 原始参数
        
        Returns:
            处理后的参数
        """
        if file_content is None:
            return original_params
        
        try:
            # 记录日志
            await ctx.info("检测到文件输入，开始解析...")
            
            # 解析文件
            parsed = FileParser.parse_file_content(file_content, file_format)
            
            # 记录解析结果
            await ctx.info(
                f"文件解析成功：{parsed['n_variables']}个变量，"
                f"{parsed['n_observations']}个观测，"
                f"数据类型={parsed['data_type']}"
            )
            
            # 转换为工具格式
            converted = FileParser.convert_to_tool_format(parsed, tool_type)
            
            # 合并参数
            result = {**original_params}
            result.update(converted)
            
            # 记录转换结果
            if tool_type == 'regression':
                await ctx.info(
                    f"数据已转换：因变量={converted.get('y_variable')}，"
                    f"自变量={converted.get('feature_names')}"
                )
            elif tool_type == 'panel':
                await ctx.info(
                    f"面板数据已识别：{len(set(converted.get('entity_ids', [])))}个实体，"
                    f"{len(set(converted.get('time_periods', [])))}个时间点"
                )
            else:
                await ctx.info(f"数据已转换为{tool_type}格式")
            
            return result
            
        except Exception as e:
            await ctx.error(f"文件解析失败: {str(e)}")
            raise ValueError(f"文件解析失败: {str(e)}")


# 便捷函数
async def process_file_for_tool(
    ctx: Any,
    file_content: Optional[str],
    file_format: str,
    tool_type: str,
    **kwargs
) -> Dict[str, Any]:
    """
    为工具处理文件输入的便捷函数
    
    使用示例：
        params = await process_file_for_tool(
            ctx=ctx,
            file_content=file_content,
            file_format=file_format,
            tool_type='regression',
            y_data=y_data,
            x_data=x_data,
            feature_names=feature_names
        )
        # params 现在包含处理后的所有参数
    """
    return await UnifiedFileInput.handle(
        ctx=ctx,
        file_content=file_content,
        file_format=file_format,
        tool_type=tool_type,
        original_params=kwargs
    )