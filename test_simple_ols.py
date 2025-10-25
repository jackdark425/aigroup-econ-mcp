"""
简单的OLS测试 - 直接调用函数
"""
import asyncio
import sys
sys.path.insert(0, 'src')

from aigroup_econ_mcp.server import mcp
from mcp.server.session import ServerSession
from mcp.server.fastmcp import Context
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class MockAppContext:
    config: Dict[str, Any]
    version: str = "0.1.0"


async def test_ols_direct():
    """直接测试OLS函数"""
    print("🧪 直接测试OLS回归函数...")
    
    # 创建mock上下文
    class MockContext:
        async def info(self, msg):
            print(f"ℹ️  {msg}")
        
        async def error(self, msg):
            print(f"❌ {msg}")
    
    ctx = MockContext()
    
    # 测试数据
    y_data = [12000, 13500, 11800, 14200, 15100, 14800, 16200, 15900]
    x_data = [
        [800, 5.2],
        [900, 5.8],
        [750, 4.9],
        [1000, 6.1],
        [1100, 6.3],
        [1050, 6.0],
        [1200, 6.5],
        [1150, 6.2]
    ]
    feature_names = ["广告支出", "价格指数"]
    
    try:
        # 导入函数
        from aigroup_econ_mcp.server import ols_regression
        
        # 调用函数
        result = await ols_regression(ctx, y_data, x_data, feature_names)
        
        print("\n✅ 函数调用成功!")
        print(f"Result type: {type(result)}")
        print(f"Content: {result.content}")
        if hasattr(result, 'structuredContent'):
            print(f"StructuredContent type: {type(result.structuredContent)}")
            print(f"StructuredContent: {result.structuredContent}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 函数调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_ols_direct())
    print(f"\n{'✅ 测试通过' if success else '❌ 测试失败'}")