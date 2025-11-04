"""
测试文件输入功能
验证新增的组件化设计和多格式支持
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_file_input_tools():
    """测试文件输入工具"""
    print("="*70)
    print("测试 AIGroup 计量经济学 MCP 服务器 v2.0 - 文件输入功能")
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
            
            print("\n" + "="*70)
            print("测试 1: OLS回归（从CSV文件，Markdown格式）")
            print("="*70)
            
            try:
                result = await session.call_tool(
                    "ols_from_file",
                    {
                        "file_path": "test_data/sample_data.csv",
                        "constant": True,
                        "confidence_level": 0.95,
                        "output_format": "markdown"
                    }
                )
                
                print("✓ OLS回归测试成功")
                if hasattr(result, 'content') and result.content:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        print("\n结果预览（前500字符）:")
                        print(content.text[:500])
                        print("...")
            except Exception as e:
                print(f"✗ OLS回归测试失败: {e}")
            
            print("\n" + "="*70)
            print("测试 2: MLE估计（从TXT文件，TXT格式）")
            print("="*70)
            
            try:
                result = await session.call_tool(
                    "mle_from_file",
                    {
                        "file_path": "test_data/sample_mle.txt",
                        "distribution": "normal",
                        "confidence_level": 0.95,
                        "output_format": "txt"
                    }
                )
                
                print("✓ MLE估计测试成功")
                if hasattr(result, 'content') and result.content:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        print("\n结果预览（前500字符）:")
                        print(content.text[:500])
                        print("...")
            except Exception as e:
                print(f"✗ MLE估计测试失败: {e}")
            
            print("\n" + "="*70)
            print("测试 3: GMM估计（从CSV文件，保存到文件）")
            print("="*70)
            
            try:
                result = await session.call_tool(
                    "gmm_from_file",
                    {
                        "file_path": "test_data/sample_data.csv",
                        "constant": True,
                        "confidence_level": 0.95,
                        "output_format": "markdown",
                        "save_path": "test_results/gmm_result.md"
                    }
                )
                
                print("✓ GMM估计测试成功")
                if hasattr(result, 'content') and result.content:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        print("\n结果预览（前300字符）:")
                        print(content.text[:300])
                        print("...")
            except Exception as e:
                print(f"✗ GMM估计测试失败: {e}")
            
            print("\n" + "="*70)
            print("测试完成")
            print("="*70)


async def main():
    """运行所有测试"""
    await test_file_input_tools()


if __name__ == "__main__":
    asyncio.run(main())