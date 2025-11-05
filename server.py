"""
AIGroup 计量经济学 MCP 服务器 - 简化修复版
直接注册工具，避免复杂的包装器
"""

import sys
import os
from typing import List, Optional, Union
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

# 设置Windows控制台编码
if sys.platform == "win32":
    try:
        # 尝试设置UTF-8编码
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        # 如果失败，使用ASCII字符
        pass

# 导入工具注册中心
from tools.mcp_tools_registry import registry

# 创建 FastMCP 服务器实例
mcp = FastMCP("aigroup-econ-mcp")

# 自动发现并注册所有工具组
print("正在自动发现工具组...")
registry.auto_discover_groups()

# 直接注册所有工具
print("正在注册工具...")
for tool_name, tool_info in registry.get_all_tools().items():
    # 获取原始工具处理器
    original_handler = tool_info["handler"]
    
    # 直接注册工具
    mcp.tool(name=tool_name, description=tool_info["description"])(original_handler)
    
    print(f"  - 已注册: {tool_name}")

@mcp.resource("guide://econometrics")
def get_econometrics_guide() -> str:
    """Get complete econometrics tools guide"""
    try:
        with open("resources/MCP_MASTER_GUIDE.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "完整使用指南文件未找到，请检查 resources/MCP_MASTER_GUIDE.md 文件是否存在。"


def main():
    """Start FastMCP server"""
    print("=" * 60)
    print("AIGroup Econometrics MCP Server - SIMPLE FIXED")
    print("=" * 60)
    print("\n架构: 简化修复版")
    print("\n已注册工具组:")
    
    # 显示工具组信息
    for group in registry.tool_groups:
        tools_in_group = [name for name, info in registry.tools.items() if info["group"] == group.name]
        print(f"  - {group.name} ({len(tools_in_group)} tools)")
    
    print(f"\n总工具数: {len(registry.tools)}")
    print("\n支持格式:")
    print("  输入: txt/json/csv/excel (.xlsx, .xls)")
    print("  输出: json/markdown/txt")
    
    print("\n优势:")
    print("  * 简化工具注册")
    print("  * 避免包装器问题")
    print("  * 直接使用原始处理器")
    
    print("\n启动服务器...")
    print("=" * 60)
    
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()