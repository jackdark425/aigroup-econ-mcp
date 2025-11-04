"""
调试GMM工具
"""
import asyncio
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(__file__))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_gmm_detailed():
    """详细测试GMM工具"""
    print("开始详细测试GMM工具...")
    
    server_params = StdioServerParameters(
        command="python",
        args=["fastmcp_server.py"]
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 测试GMM估计
                y_data = [1, 2, 3, 4, 5]
                x_data = [[1], [2], [3], [4], [5]]
                
                print(f"输入数据:")
                print(f"  y_data: {y_data}")
                print(f"  x_data: {x_data}")
                
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
                
                print("✓ GMM估计成功")
                print(f"结果: {result}")
                
                if hasattr(result, 'content') and result.content:
                    print(f"内容: {result.content[0].text}")
                
                return True
                
    except Exception as e:
        print(f"✗ GMM测试失败:")
        print(f"  错误类型: {type(e).__name__}")
        print(f"  错误信息: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_gmm_detailed())
    sys.exit(0 if success else 1)