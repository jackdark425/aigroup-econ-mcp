"""
数据加载辅助模块
提供通用的文件加载功能，支持CSV、JSON和TXT格式
"""

from typing import Dict, List, Union
from pathlib import Path
import pandas as pd
from .file_parser import FileParser


async def load_data_if_path(
    data: Union[Dict[str, List[float]], str],
    ctx = None
) -> Dict[str, List[float]]:
    """
    智能加载数据：如果是字符串则作为文件路径加载，否则直接返回
    
    Args:
        data: 数据字典或文件路径（支持CSV/JSON/TXT）
        ctx: MCP上下文对象（可选，用于日志）
        
    Returns:
        数据字典
        
    Raises:
        ValueError: 文件不存在或读取失败
    """
    # 如果已经是字典，直接返回
    if isinstance(data, dict):
        return data
    
    # 如果是字符串，作为文件路径处理
    if isinstance(data, str):
        if ctx:
            await ctx.info(f"📁 检测到文件路径，正在加载: {data}")
        
        try:
            # 检查文件是否存在
            path = Path(data)
            if not path.exists():
                raise ValueError(f"文件不存在: {data}")
            
            # 使用FileParser解析文件（支持CSV/JSON/TXT自动检测）
            parsed = FileParser.parse_file_path(str(path), "auto")
            
            # 返回数据字典
            result = parsed["data"]
            
            if ctx:
                await ctx.info(
                    f"✅ {parsed['format'].upper()}文件加载成功："
                    f"{parsed['n_variables']}个变量，{parsed['n_observations']}个观测"
                )
            
            return result
            
        except FileNotFoundError:
            raise ValueError(f"文件不存在: {data}")
        except Exception as e:
            raise ValueError(f"文件读取失败: {str(e)}")
    
    # 其他类型报错
    raise TypeError(f"不支持的数据类型: {type(data)}，期望Dict或str")


async def load_single_var_if_path(
    data: Union[List[float], str],
    ctx = None,
    column_name: str = None
) -> List[float]:
    """
    智能加载单变量数据：如果是字符串则作为文件路径加载，否则直接返回
    
    Args:
        data: 数据列表或文件路径（支持CSV/JSON/TXT）
        ctx: MCP上下文对象（可选，用于日志）
        column_name: 文件中要读取的列名（可选，默认读取第一列）
        
    Returns:
        数据列表
        
    Raises:
        ValueError: 文件不存在或读取失败
    """
    # 如果已经是列表，直接返回
    if isinstance(data, list):
        return data
    
    # 如果是字符串，作为文件路径处理
    if isinstance(data, str):
        if ctx:
            await ctx.info(f"📁 检测到文件路径，正在加载: {data}")
        
        try:
            # 检查文件是否存在
            path = Path(data)
            if not path.exists():
                raise ValueError(f"文件不存在: {data}")
            
            # 使用FileParser解析文件
            parsed = FileParser.parse_file_path(str(path), "auto")
            data_dict = parsed["data"]
            
            # 确定要读取的列
            if column_name:
                if column_name not in data_dict:
                    raise ValueError(
                        f"列'{column_name}'不存在于文件中。"
                        f"可用列: {list(data_dict.keys())}"
                    )
                result = data_dict[column_name]
            else:
                # 默认读取第一列
                first_col = parsed["variables"][0]
                result = data_dict[first_col]
                if ctx:
                    await ctx.info(f"未指定列名，使用第一列: {first_col}")
            
            if ctx:
                await ctx.info(
                    f"✅ {parsed['format'].upper()}文件加载成功：{len(result)}个观测"
                )
            
            return result
            
        except FileNotFoundError:
            raise ValueError(f"文件不存在: {data}")
        except Exception as e:
            raise ValueError(f"文件读取失败: {str(e)}")
    
    # 其他类型报错
    raise TypeError(f"不支持的数据类型: {type(data)}，期望List或str")


async def load_x_data_if_path(
    data: Union[List[List[float]], str],
    ctx = None
) -> List[List[float]]:
    """
    智能加载自变量数据：如果是字符串则作为文件路径加载，否则直接返回
    
    Args:
        data: 自变量数据（二维列表）或文件路径（支持CSV/JSON/TXT）
        ctx: MCP上下文对象（可选，用于日志）
        
    Returns:
        自变量数据（二维列表）
        
    Raises:
        ValueError: 文件不存在或读取失败
    """
    # 如果已经是二维列表，直接返回
    if isinstance(data, list) and all(isinstance(item, list) for item in data):
        return data
    
    # 如果是字符串，作为文件路径处理
    if isinstance(data, str):
        if ctx:
            await ctx.info(f"📁 检测到自变量文件路径，正在加载: {data}")
        
        try:
            # 检查文件是否存在
            path = Path(data)
            if not path.exists():
                raise ValueError(f"文件不存在: {data}")
            
            # 使用FileParser解析文件
            parsed = FileParser.parse_file_path(str(path), "auto")
            data_dict = parsed["data"]
            
            # 转换为二维列表格式
            variables = parsed["variables"]
            n_obs = parsed["n_observations"]
            
            result = []
            for i in range(n_obs):
                row = [data_dict[var][i] for var in variables]
                result.append(row)
            
            if ctx:
                await ctx.info(
                    f"✅ 自变量{parsed['format'].upper()}文件加载成功："
                    f"{len(result)}个观测，{len(variables)}个自变量"
                )
            
            return result
            
        except FileNotFoundError:
            raise ValueError(f"文件不存在: {data}")
        except Exception as e:
            raise ValueError(f"自变量文件读取失败: {str(e)}")
    
    # 其他类型报错
    raise TypeError(f"不支持的数据类型: {type(data)}，期望List[List[float]]或str")