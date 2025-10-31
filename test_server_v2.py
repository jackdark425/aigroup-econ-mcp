#!/usr/bin/env python3
"""
测试server_v2.py的所有功能
包括：基础统计、回归、时间序列、面板数据、机器学习
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aigroup_econ_mcp.server_v2 import mcp
from mcp.server.session import ServerSession


class TestContext:
    """模拟MCP上下文"""
    async def info(self, msg): print(f"ℹ️  {msg}")
    async def error(self, msg): print(f"❌ {msg}")
    async def warning(self, msg): print(f"⚠️  {msg}")


async def test_basic_statistics():
    """测试基础统计工具"""
    print("\n" + "="*60)
    print("📊 测试1: 基础统计工具")
    print("="*60)
    
    ctx = TestContext()
    
    # 测试descriptive_statistics
    try:
        from aigroup_econ_mcp.server_v2 import descriptive_statistics
        result = await descriptive_statistics(
            ctx=ctx,
            data={"GDP": [3.2, 2.8, 3.5], "CPI": [2.1, 2.3, 1.9]}
        )
        print("✅ descriptive_statistics - 通过")
    except Exception as e:
        print(f"❌ descriptive_statistics - 失败: {e}")
        return False
    
    # 测试correlation_analysis
    try:
        from aigroup_econ_mcp.server_v2 import correlation_analysis
        result = await correlation_analysis(
            ctx=ctx,
            data={"X": [1, 2, 3], "Y": [2, 4, 6]},
            method="pearson"
        )
        print("✅ correlation_analysis - 通过")
    except Exception as e:
        print(f"❌ correlation_analysis - 失败: {e}")
        return False
    
    return True


async def test_regression():
    """测试回归分析工具"""
    print("\n" + "="*60)
    print("📈 测试2: 回归分析工具")
    print("="*60)
    
    ctx = TestContext()
    
    # 测试OLS回归
    try:
        from aigroup_econ_mcp.server_v2 import ols_regression
        result = await ols_regression(
            ctx=ctx,
            y_data=[10, 20, 30, 40],
            x_data=[[1, 2], [2, 4], [3, 6], [4, 8]],
            feature_names=["X1", "X2"]
        )
        print("✅ ols_regression - 通过")
    except Exception as e:
        print(f"❌ ols_regression - 失败: {e}")
        return False
    
    return True


async def test_hypothesis_testing():
    """测试假设检验"""
    print("\n" + "="*60)
    print("🔬 测试3: 假设检验工具")
    print("="*60)
    
    ctx = TestContext()
    
    try:
        from aigroup_econ_mcp.server_v2 import hypothesis_testing
        result = await hypothesis_testing(
            ctx=ctx,
            data1=[1, 2, 3, 4, 5],
            test_type="t_test"
        )
        print("✅ hypothesis_testing - 通过")
    except Exception as e:
        print(f"❌ hypothesis_testing - 失败: {e}")
        return False
    
    return True


async def test_time_series():
    """测试时间序列工具"""
    print("\n" + "="*60)
    print("📉 测试4: 时间序列工具")
    print("="*60)
    
    ctx = TestContext()
    
    try:
        from aigroup_econ_mcp.server_v2 import time_series_analysis
        result = await time_series_analysis(
            ctx=ctx,
            data=[100, 102, 105, 103, 108, 110, 112, 115]
        )
        print("✅ time_series_analysis - 通过")
    except Exception as e:
        print(f"❌ time_series_analysis - 失败: {e}")
        return False
    
    return True


async def test_machine_learning():
    """测试机器学习工具"""
    print("\n" + "="*60)
    print("🤖 测试5: 机器学习工具")
    print("="*60)
    
    ctx = TestContext()
    
    # 测试随机森林
    try:
        from aigroup_econ_mcp.server_v2 import random_forest_regression_analysis
        result = await random_forest_regression_analysis(
            ctx=ctx,
            y_data=[10, 20, 30, 40, 50],
            x_data=[[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]],
            feature_names=["F1", "F2"],
            n_estimators=10
        )
        print("✅ random_forest_regression - 通过")
    except Exception as e:
        print(f"❌ random_forest_regression - 失败: {e}")
        return False
    
    return True


async def test_imports():
    """测试所有导入"""
    print("\n" + "="*60)
    print("📦 测试6: 模块导入")
    print("="*60)
    
    try:
        from aigroup_econ_mcp.tools.tool_handlers import (
            handle_descriptive_statistics,
            handle_ols_regression,
            handle_hypothesis_testing,
            handle_time_series_analysis,
            handle_correlation_analysis,
        )
        print("✅ tool_handlers导入 - 通过")
    except Exception as e:
        print(f"❌ tool_handlers导入 - 失败: {e}")
        return False
    
    try:
        from aigroup_econ_mcp.tools.decorators import econometric_tool
        print("✅ decorators导入 - 通过")
    except Exception as e:
        print(f"❌ decorators导入 - 失败: {e}")
        return False
    
    try:
        from aigroup_econ_mcp.tools.data_loader import load_data_if_path
        print("✅ data_loader导入 - 通过")
    except Exception as e:
        print(f"❌ data_loader导入 - 失败: {e}")
        return False
    
    return True


async def main():
    """运行所有测试"""
    print("\n" + "🧪 " * 30)
    print("开始测试 server_v2.py")
    print("🧪 " * 30)
    
    tests = [
        ("模块导入", test_imports),
        ("基础统计", test_basic_statistics),
        ("回归分析", test_regression),
        ("假设检验", test_hypothesis_testing),
        ("时间序列", test_time_series),
        ("机器学习", test_machine_learning),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ {name}测试异常: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    print(f"✅ 通过: {passed}/{len(tests)}")
    print(f"❌ 失败: {failed}/{len(tests)}")
    print(f"📈 成功率: {passed/len(tests)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 所有测试通过！server_v2.py可以安全使用。")
        return 0
    else:
        print(f"\n⚠️  有{failed}个测试失败，需要修复。")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)