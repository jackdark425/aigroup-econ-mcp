#!/usr/bin/env python3
"""
测试CSV文件路径功能
"""

import asyncio
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aigroup_econ_mcp.server import mcp
from mcp.server.session import ServerSession
from mcp.server.fastmcp import Context


async def test_csv_path():
    """测试使用CSV文件路径的功能"""
    
    print("=" * 60)
    print("测试CSV文件路径功能")
    print("=" * 60)
    
    # 创建模拟的上下文
    class MockContext:
        async def info(self, msg):
            print(f"ℹ️  INFO: {msg}")
        
        async def error(self, msg):
            print(f"❌ ERROR: {msg}")
        
        async def warning(self, msg):
            print(f"⚠️  WARNING: {msg}")
    
    ctx = MockContext()
    
    # 测试文件路径
    csv_path = "d:/aigroup-econ-mcp/test_data.csv"
    
    print(f"\n📁 测试文件: {csv_path}")
    print("-" * 60)
    
    # 导入必要的函数
    from aigroup_econ_mcp.server import descriptive_statistics, correlation_analysis
    
    # 测试1: descriptive_statistics with CSV path
    print("\n【测试1】descriptive_statistics 使用CSV文件路径")
    print("-" * 60)
    try:
        result = await descriptive_statistics(ctx, csv_path)
        if result.isError:
            print(f"❌ 测试失败: {result.content[0].text}")
        else:
            print("✅ 测试成功!")
            print(f"\n结果:\n{result.content[0].text}")
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 测试2: correlation_analysis with CSV path
    print("\n【测试2】correlation_analysis 使用CSV文件路径")
    print("-" * 60)
    try:
        result = await correlation_analysis(ctx, csv_path)
        if result.isError:
            print(f"❌ 测试失败: {result.content[0].text}")
        else:
            print("✅ 测试成功!")
            print(f"\n结果:\n{result.content[0].text}")
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 测试3: 使用传统字典方式（确保向后兼容）
    print("\n【测试3】使用传统数据字典方式（向后兼容测试）")
    print("-" * 60)
    data_dict = {
        "GDP增长率": [3.2, 2.8, 3.5, 2.9],
        "通货膨胀率": [2.1, 2.3, 1.9, 2.4],
        "失业率": [4.5, 4.2, 4.0, 4.3]
    }
    try:
        result = await descriptive_statistics(ctx, data_dict)
        if result.isError:
            print(f"❌ 测试失败: {result.content[0].text}")
        else:
            print("✅ 测试成功! (传统方式仍然有效)")
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_csv_path())