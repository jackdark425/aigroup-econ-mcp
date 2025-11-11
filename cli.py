#!/usr/bin/env python3
"""
AIGroup Econometrics MCP - CLI 入口
"""

import sys
import os
import traceback
import argparse
from pathlib import Path

# 导入版本信息
try:
    from __init__ import __version__, __author__, __email__
except ImportError:
    # 如果相对导入失败，尝试从包导入
    try:
        import __init__
        __version__ = __init__.__version__
        __author__ = __init__.__author__
        __email__ = __init__.__email__
    except ImportError:
        __version__ = "2.0.7"
        __author__ = "AIGroup"
        __email__ = "jackdark425@gmail.com"

def show_version():
    """显示版本信息"""
    print(f"aigroup-econ-mcp v{__version__}")
    print("Professional econometrics MCP tool")
    print(f"Author: {__author__}")
    print(f"Email: {__email__}")

def main():
    """CLI 主函数"""
    parser = argparse.ArgumentParser(
        description="AIGroup Econometrics MCP - Professional econometrics MCP tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aigroup-econ-mcp                    # 启动 MCP 服务器
  aigroup-econ-mcp --version          # 显示版本信息
  aigroup-econ-mcp --help             # 显示帮助信息
        """
    )
    
    parser.add_argument(
        '--version', 
        action='store_true',
        help='显示版本信息'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true', 
        help='启用详细日志输出'
    )
    
    args = parser.parse_args()
    
    if args.version:
        show_version()
        return
    
    try:
        if args.verbose:
            print("正在启动 AIGroup Econometrics MCP 服务器...")
            print(f"版本: {__version__}")
        
        # 导入并运行服务器
        from server import main as server_main
        server_main()
    except ImportError as e:
        print(f"导入错误: {e}")
        traceback.print_exc()
        print("请确保所有依赖已正确安装")
        sys.exit(1)
    except Exception as e:
        print(f"启动服务器时出错: {e}")
        traceback.print_exc()
        sys.exit(1)

# 添加cli函数以匹配pyproject.toml中的入口点定义
cli = main

if __name__ == "__main__":
    main()