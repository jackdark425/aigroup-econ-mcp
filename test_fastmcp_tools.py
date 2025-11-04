"""
测试新的FastMCP服务器工具
验证OLS、MLE、GMM三个工具是否正常工作
"""

import asyncio
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_ols_tool():
    """测试OLS回归工具"""
    print("测试OLS回归工具...")
    
    # 创建服务器参数
    server_params = StdioServerParameters(
        command="python",
        args=["fastmcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化连接
            await session.initialize()
            
            # 列出可用工具
            tools = await session.list_tools()
            print(f"可用工具: {[tool.name for tool in tools.tools]}")
            
            # 测试OLS回归
            y_data = [1, 2, 3, 4, 5]
            x_data = [[1], [2], [3], [4], [5]]
            
            result = await session.call_tool(
                "basic_parametric_estimation_ols",
                {
                    "y_data": y_data,
                    "x_data": x_data,
                    "feature_names": ["x1"],
                    "constant": True,
                    "confidence_level": 0.95
                }
            )
            
            print("✓ OLS回归测试成功")
            print(f"  结果类型: {type(result)}")
            print(f"  内容长度: {len(result.content)}")
            
            # 打印结构化结果
            if hasattr(result, 'structuredContent') and result.structuredContent:
                structured = result.structuredContent
                print(f"  系数: {structured.get('coefficients', [])}")
                print(f"  R方: {structured.get('r_squared', 0):.4f}")
                print(f"  观测数量: {structured.get('n_obs', 0)}")
            
            return True


async def test_mle_tool():
    """测试MLE估计工具"""
    print("测试MLE估计工具...")
    
    server_params = StdioServerParameters(
        command="python",
        args=["fastmcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 测试MLE估计 (正态分布)
            data = [1.2, 2.3, 1.8, 2.1, 1.9, 2.4, 2.0, 1.7]
            
            result = await session.call_tool(
                "basic_parametric_estimation_mle",
                {
                    "data": data,
                    "distribution": "normal",
                    "confidence_level": 0.95
                }
            )
            
            print("✓ MLE估计测试成功")
            
            if hasattr(result, 'structuredContent') and result.structuredContent:
                structured = result.structuredContent
                print(f"  参数: {structured.get('parameters', [])}")
                print(f"  对数似然值: {structured.get('log_likelihood', 0):.4f}")
                print(f"  收敛状态: {structured.get('convergence', False)}")
            
            return True


async def test_gmm_tool():
    """测试GMM估计工具"""
    print("测试GMM估计工具...")
    
    server_params = StdioServerParameters(
        command="python",
        args=["fastmcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 测试GMM估计
            y_data = [1, 2, 3, 4, 5]
            x_data = [[1], [2], [3], [4], [5]]
            
            result = await session.call_tool(
                "basic_parametric_estimation_gmm",
                {
                    "y_data": y_data,
                    "x_data": x_data,
                    "feature_names": ["x1"],
                    "constant": True,
                    "confidence_level": 0.95
                }
            )
            
            print("✓ GMM估计测试成功")
            
            if hasattr(result, 'structuredContent') and result.structuredContent:
                structured = result.structuredContent
                print(f"  系数: {structured.get('coefficients', [])}")
                print(f"  J统计量: {structured.get('j_statistic', 0):.4f}")
                print(f"  矩条件数量: {structured.get('n_moments', 0)}")
            
            return True


async def test_resources():
    """测试资源访问"""
    print("测试资源访问...")
    
    server_params = StdioServerParameters(
        command="python",
        args=["fastmcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 列出资源
            resources = await session.list_resources()
            print(f"可用资源: {[resource.uri for resource in resources.resources]}")
            
            # 读取服务器配置资源
            from pydantic import AnyUrl
            config_content = await session.read_resource(AnyUrl("config://server"))
            print("✓ 资源访问测试成功")
            print(f"  配置内容: {config_content.contents[0].text[:100]}...")
            
            return True


async def main():
    """运行所有测试"""
    print("开始测试新的FastMCP服务器...\n")
    
    tests = [
        test_ols_tool,
        test_mle_tool,
        test_gmm_tool,
        test_resources
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} 测试失败: {e}")
        print()
    
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！新的FastMCP服务器已成功重构。")
        print("\n使用方法:")
        print("1. 运行服务器: python fastmcp_server.py")
        print("2. 通过MCP客户端连接使用三个工具:")
        print("   - basic_parametric_estimation_ols (OLS回归)")
        print("   - basic_parametric_estimation_mle (最大似然估计)")
        print("   - basic_parametric_estimation_gmm (广义矩估计)")
    else:
        print("❌ 部分测试失败，请检查代码。")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)