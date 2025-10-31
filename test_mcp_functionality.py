#!/usr/bin/env python3
"""
测试MCP服务器功能
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_server():
    """测试MCP服务器连接和工具调用"""
    
    # 配置服务器参数
    server_params = StdioServerParameters(
        command="aigroup-econ-mcp",
        args=["--debug"]
    )
    
    try:
        # 连接到MCP服务器
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # 初始化会话
                init_result = await session.initialize()
                print(f"初始化结果: {init_result}")
                
                # 列出可用工具
                tools_result = await session.list_tools()
                print(f"可用工具数量: {len(tools_result.tools)}")
                
                # 显示前几个工具
                for i, tool in enumerate(tools_result.tools[:5]):
                    print(f"工具 {i+1}: {tool.name} - {tool.description}")
                
                # 测试一个简单的工具
                if tools_result.tools:
                    test_tool = tools_result.tools[0]  # 第一个工具
                    print(f"\n测试工具: {test_tool.name}")
                    
                    # 准备测试数据
                    test_data = {
                        "data": {
                            "var1": [1.0, 2.0, 3.0, 4.0, 5.0],
                            "var2": [2.0, 4.0, 6.0, 8.0, 10.0]
                        }
                    }
                    
                    # 调用工具
                    try:
                        result = await session.call_tool(test_tool.name, test_data)
                        print(f"工具调用结果: {result}")
                    except Exception as e:
                        print(f"工具调用失败: {e}")
                
                print("\nMCP服务器功能测试完成！")
                
    except Exception as e:
        print(f"MCP服务器连接失败: {e}")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())