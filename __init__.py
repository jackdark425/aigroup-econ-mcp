"""
AIGroup Econometrics MCP Package
专业计量经济学MCP工具包
"""

__version__ = "2.0.0"
__author__ = "AIGroup"
__email__ = "jackdark425@gmail.com"

# 导出主要模块
from .server import main
from .cli import main as cli_main

__all__ = ["main", "cli_main"]