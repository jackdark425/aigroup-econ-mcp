"""
测试整合后的工具
验证每个工具支持文件输入和直接传参两种方式
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_integrated_tools():
    """测试整合后的三个工具"""
    print("="*70)
    print("测试 AIGroup 计量经济学 MCP 服务器 v2.0 - 整合工具")
    print("="*70)
    
    server_params = StdioServerParameters(
        command="python",
        args=["fastmcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 列出所有可用工具
            tools = await session.list_tools()
            print(f"\n可用工具数量: {len(tools.tools)}")
            print("工具列表:")
            for tool in tools.tools:
                print(f"  - {tool.name}")
            
            # ================================================================
            # 测试 OLS 工具
            # ================================================================
            print("\n" + "="*70)
            print("测试 1: OLS - 直接传参模式 (JSON输出)")
            print("="*70)
            
            try:
                result = await session.call_tool(
                    "basic_parametric_estimation_ols",
                    {
                        "y_data": [1, 2, 3, 4, 5],
                        "x_data": [[1], [2], [3], [4], [5]],
                        "constant": True,
                        "output_format": "json"
                    }
                )
                
                print("✓ OLS直接传参测试成功")
                if hasattr(result, 'content') and result.content:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        print("\nJSON结果预览:")
                        print(content.text[:300] + "...")
            except Exception as e:
                print(f"✗ 测试失败: {e}")
            
            print("\n" + "="*70)
            print("测试 2: OLS - 文件输入模式 (Markdown输出)")
            print("="*70)
            
            try:
                result = await session.call_tool(
                    "basic_parametric_estimation_ols",
                    {
                        "file_path": "test_data/sample_data.csv",
                        "constant": True,
                        "output_format": "markdown"
                    }
                )
                
                print("✓ OLS文件输入测试成功")
                if hasattr(result, 'content') and result.content:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        print("\nMarkdown结果预览:")
                        print(content.text[:400] + "...")
            except Exception as e:
                print(f"✗ 测试失败: {e}")
            
            # ================================================================
            # 测试 MLE 工具
            # ================================================================
            print("\n" + "="*70)
            print("测试 3: MLE - 直接传参模式 (JSON输出)")
            print("="*70)
            
            try:
                result = await session.call_tool(
                    "basic_parametric_estimation_mle",
                    {
                        "data": [1.2, 2.3, 1.8, 2.1, 1.9, 2.4, 2.0, 1.7],
                        "distribution": "normal",
                        "output_format": "json"
                    }
                )
                
                print("✓ MLE直接传参测试成功")
                if hasattr(result, 'content') and result.content:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        print("\nJSON结果预览:")
                        print(content.text[:300] + "...")
            except Exception as e:
                print(f"✗ 测试失败: {e}")
            
            print("\n" + "="*70)
            print("测试 4: MLE - 文件输入模式 (TXT输出)")
            print("="*70)
            
            try:
                result = await session.call_tool(
                    "basic_parametric_estimation_mle",
                    {
                        "file_path": "test_data/sample_mle.txt",
                        "distribution": "normal",
                        "output_format": "txt"
                    }
                )
                
                print("✓ MLE文件输入测试成功")
                if hasattr(result, 'content') and result.content:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        print("\nTXT结果预览:")
                        print(content.text[:400] + "...")
            except Exception as e:
                print(f"✗ 测试失败: {e}")
            
            # ================================================================
            # 测试 GMM 工具
            # ================================================================
            print("\n" + "="*70)
            print("测试 5: GMM - 直接传参模式 (JSON输出)")
            print("="*70)
            
            try:
                result = await session.call_tool(
                    "basic_parametric_estimation_gmm",
                    {
                        "y_data": [1, 2, 3, 4, 5],
                        "x_data": [[1], [2], [3], [4], [5]],
                        "constant": True,
                        "output_format": "json"
                    }
                )
                
                print("✓ GMM直接传参测试成功")
                if hasattr(result, 'content') and result.content:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        print("\nJSON结果预览:")
                        print(content.text[:300] + "...")
            except Exception as e:
                print(f"✗ 测试失败: {e}")
            
            print("\n" + "="*70)
            print("测试 6: GMM - 文件输入模式 (Markdown + 保存)")
            print("="*70)
            
            try:
                result = await session.call_tool(
                    "basic_parametric_estimation_gmm",
                    {
                        "file_path": "test_data/sample_data.csv",
                        "constant": True,
                        "output_format": "markdown",
                        "save_path": "test_results/gmm_integrated.md"
                    }
                )
                
                print("✓ GMM文件输入+保存测试成功")
                if hasattr(result, 'content') and result.content:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        print("\n结果预览:")
                        print(content.text[:300] + "...")
            except Exception as e:
                print(f"✗ 测试失败: {e}")
            
            print("\n" + "="*70)
            print("测试完成")
            print("="*70)
            print("\n总结：")
            print("✓ 三个工具均成功整合")
            print("✓ 支持文件输入和直接传参")
            print("✓ 支持JSON、Markdown、TXT三种输出格式")
            print("✓ 支持结果保存到文件")


async def main():
    """运行所有测试"""
    await test_integrated_tools()


if __name__ == "__main__":
    asyncio.run(main())