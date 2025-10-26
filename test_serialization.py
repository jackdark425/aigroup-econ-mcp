#!/usr/bin/env python3
"""
测试MCP服务器的NumPy序列化修复效果
直接测试服务器函数，避免复杂的MCP协议
"""

import sys
import json
import traceback
import asyncio
from src.aigroup_econ_mcp.server import (
    ols_regression, time_series_analysis, descriptive_statistics
)


def check_numpy_types(obj, path=""):
    """检查对象中是否包含numpy类型"""
    issues = []

    if hasattr(obj, 'dtype'):  # numpy类型检查
        issues.append(f"发现numpy类型在 {path}: {type(obj)} (dtype: {obj.dtype})")
        return issues

    if isinstance(obj, dict):
        for key, value in obj.items():
            issues.extend(check_numpy_types(value, f"{path}.{key}" if path else key))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            issues.extend(check_numpy_types(item, f"{path}[{i}]"))

    return issues


async def test_ols_regression_serialization():
    """测试OLS回归序列化"""
    print("🧪 测试OLS回归序列化...")

    try:
        from mcp.types import CallToolResult
        from mcp.server.session import ServerSession

        # 创建模拟的MCP上下文
        class MockContext:
            async def info(self, msg): print(f"INFO: {msg}")
            async def error(self, msg): print(f"ERROR: {msg}")

        ctx = MockContext()

        # 调用OLS回归
        result = await ols_regression(
            ctx=ctx,
            y_data=[120, 135, 118, 142, 155, 160, 148, 175],
            x_data=[
                [8, 100], [9, 98], [7.5, 102], [10, 97],
                [11, 95], [12, 94], [10.5, 96], [13, 93]
            ],
            feature_names=["advertising", "price"]
        )

        # 检查结果
        if hasattr(result, 'structuredContent') and result.structuredContent:
            structured = result.structuredContent
            print(f"✅ OLS structuredContent: {type(structured)}")
            print(f"📊 OLS结果详情: {json.dumps(structured, indent=2, default=str)}")

            # 检查是否有numpy类型
            issues = check_numpy_types(structured)
            if issues:
                print(f"❌ 发现序列化问题: {issues}")
                return False
            else:
                print("✅ OLS回归序列化正确")
                return True
        else:
            print("❌ OLS没有structuredContent")
            return False

    except Exception as e:
        print(f"❌ OLS测试出错: {e}")
        traceback.print_exc()
        return False


async def test_time_series_serialization():
    """测试时间序列分析序列化"""
    print("🧪 测试时间序列分析序列化...")

    try:
        # 创建模拟的MCP上下文
        class MockContext:
            async def info(self, msg): print(f"INFO: {msg}")
            async def error(self, msg): print(f"ERROR: {msg}")

        ctx = MockContext()

        # 调用时间序列分析
        result = await time_series_analysis(
            ctx=ctx,
            data=[100, 110, 120, 115, 125, 130, 128, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250]
        )

        # 检查结果
        if hasattr(result, 'structuredContent') and result.structuredContent:
            structured = result.structuredContent
            print(f"✅ 时间序列 structuredContent: {type(structured)}")

            # 检查是否有numpy类型
            issues = check_numpy_types(structured)
            if issues:
                print(f"❌ 发现序列化问题: {issues}")
                return False
            else:
                print("✅ 时间序列分析序列化正确")
                return True
        else:
            print("❌ 时间序列分析没有structuredContent")
            return False

    except Exception as e:
        print(f"❌ 时间序列测试出错: {e}")
        traceback.print_exc()
        return False


async def test_descriptive_stats_serialization():
    """测试描述性统计序列化"""
    print("🧪 测试描述性统计序列化...")

    try:
        # 创建模拟的MCP上下文
        class MockContext:
            async def info(self, msg): print(f"INFO: {msg}")
            async def error(self, msg): print(f"ERROR: {msg}")

        ctx = MockContext()

        # 调用描述性统计
        result = await descriptive_statistics(
            ctx=ctx,
            data={
                "销售额": [120, 135, 118, 142, 155, 160, 148, 175],
                "广告支出": [8, 9, 7.5, 10, 11, 12, 10.5, 13],
                "价格": [100, 98, 102, 97, 95, 94, 96, 93]
            }
        )

        # 检查结果
        if hasattr(result, 'structuredContent') and result.structuredContent:
            structured = result.structuredContent
            print(f"✅ 描述性统计 structuredContent: {type(structured)}")

            # 检查是否有numpy类型
            issues = check_numpy_types(structured)
            if issues:
                print(f"❌ 发现序列化问题: {issues}")
                return False
            else:
                print("✅ 描述性统计序列化正确")
                return True
        else:
            print("❌ 描述性统计没有structuredContent")
            return False

    except Exception as e:
        print(f"❌ 描述性统计测试出错: {e}")
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("🔬 开始NumPy序列化修复验证测试")
    print("=" * 50)

    # 导入必要的异步支持
    import asyncio

    # 运行所有测试
    tests = [
        test_ols_regression_serialization,
        test_time_series_serialization,
        test_descriptive_stats_serialization
    ]

    results = []
    for test_func in tests:
        result = await test_func()
        results.append(result)
        print()

    # 总结结果
    print("=" * 50)
    if all(results):
        print("🎉 所有测试通过！NumPy序列化修复成功。")
        print("✅ MCP服务器可以正确地将statsmodels结果转换为Python原生类型")
        return True
    else:
        print("❌ 部分测试失败，序列化修复可能不完整。")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)