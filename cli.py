#!/usr/bin/env python3
"""
AIGroup Econometrics MCP - CLI 入口
"""

import sys
import os
from pathlib import Path

def main():
    """CLI 主函数"""
    try:
        # 导入并运行服务器
        from server import main as server_main
        server_main()
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖已正确安装")
        sys.exit(1)
    except Exception as e:
        print(f"启动服务器时出错: {e}")
        sys.exit(1)

# 添加cli函数以匹配pyproject.toml中的入口点定义
cli = main

if __name__ == "__main__":
    main()