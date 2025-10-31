"""
数据加载辅助模块
提供通用的CSV文件加载功能
"""

from typing import Dict, List, Union
from pathlib import Path
import pandas as pd


async def load_data_if_path(
    data: Union[Dict[str, List[float]], str],
    ctx = None
) -> Dict[str, List[float]]:
    """
    智能加载数据：如果是字符串则作为文件路径加载，否则直接返回
    
    Args:
        data: 数据字典或CSV文件路径
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
            
            # 读取CSV文件
            df = pd.read_csv(path)
            
            # 转换为字典格式
            result = {col: df[col].tolist() for col in df.columns}
            
            if ctx:
                await ctx.info(f"✅ CSV文件加载成功：{len(df.columns)}个变量，{len(df)}个观测")
            
            return result
            
        except FileNotFoundError:
            raise ValueError(f"文件不存在: {data}")
        except Exception as e:
            raise ValueError(f"CSV文件读取失败: {str(e)}")
    
    # 其他类型报错

async def load_single_var_if_path(
    data: Union[List[float], str],
    ctx = None,
    column_name: str = None
) -> List[float]:
    """
    智能加载单变量数据：如果是字符串则作为文件路径加载，否则直接返回
    
    Args:
        data: 数据列表或CSV文件路径
        ctx: MCP上下文对象（可选，用于日志）
        column_name: CSV文件中要读取的列名（可选，默认读取第一列）
        
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
            
            # 读取CSV文件
            df = pd.read_csv(path)
            
            # 确定要读取的列
            if column_name:
                if column_name not in df.columns:
                    raise ValueError(f"列'{column_name}'不存在于CSV文件中。可用列: {list(df.columns)}")
                result = df[column_name].tolist()
            else:
                # 默认读取第一列
                result = df.iloc[:, 0].tolist()
                if ctx:
                    await ctx.info(f"未指定列名，使用第一列: {df.columns[0]}")
            
            if ctx:
                await ctx.info(f"✅ CSV文件加载成功：{len(result)}个观测")
            
            return result
            
        except FileNotFoundError:
            raise ValueError(f"文件不存在: {data}")
        except Exception as e:
            raise ValueError(f"CSV文件读取失败: {str(e)}")
    
    # 其他类型报错
    raise TypeError(f"不支持的数据类型: {type(data)}，期望List或str")
async def load_x_data_if_path(
    data: Union[List[List[float]], str],
    ctx = None
) -> List[List[float]]:
    """
    智能加载自变量数据：如果是字符串则作为文件路径加载，否则直接返回
    
    Args:
        data: 自变量数据（二维列表）或CSV文件路径
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
            
            # 读取CSV文件
            df = pd.read_csv(path)
            
            # 转换为二维列表格式
            result = df.values.tolist()
            
            if ctx:
                await ctx.info(f"✅ 自变量CSV文件加载成功：{len(result)}个观测，{len(result[0]) if result else 0}个自变量")
            
            return result
            
        except FileNotFoundError:
            raise ValueError(f"文件不存在: {data}")
        except Exception as e:
            raise ValueError(f"自变量CSV文件读取失败: {str(e)}")
    
    # 其他类型报错
    raise TypeError(f"不支持的数据类型: {type(data)}，期望List[List[float]]或str")
    raise TypeError(f"不支持的数据类型: {type(data)}，期望Dict或str")