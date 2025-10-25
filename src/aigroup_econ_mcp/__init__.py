"""
AIGroup 计量经济学 MCP 服务
专业计量经济学MCP工具 - 让大模型直接进行数据分析

提供：
- 描述性统计分析
- OLS回归分析
- 时间序列分析
- 假设检验
- 模型诊断
"""

__version__ = "0.1.2"
__author__ = "AIGroup"
__description__ = "专业计量经济学MCP工具 - 让大模型直接进行数据分析"

from .server import create_mcp_server

__all__ = ["create_mcp_server", "__version__", "__author__", "__description__"]