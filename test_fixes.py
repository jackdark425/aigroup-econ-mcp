"""
测试修复后的MCP服务功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.aigroup_econ_mcp.server import (
    descriptive_statistics,
    ols_regression,
    time_series_analysis,
    correlation_analysis,
    create_mcp_server
)
import asyncio
from mcp.server.session import ServerSession
from mcp.server.fastmcp import Context
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class MockAppContext:
    """模拟应用上下文"""
    config: Dict[str, Any]
    version: str = "0.1.0"


class MockServerSession:
    """模拟服务器会话"""
    pass


class MockContext:
    """模拟上下文对象"""
    
    def __init__(self):
        self.messages = []
    
    async def info(self, message: str):
        print(f"[INFO] {message}")
        self.messages.append(("info", message))
    
    async def warning(self, message: str):
        print(f"[WARNING] {message}")
        self.messages.append(("warning", message))
    
    async def error(self, message: str):
        print(f"[ERROR] {message}")
        self.messages.append(("error", message))


async def test_descriptive_statistics():
    """测试描述性统计功能"""
    print("\n=== 测试描述性统计 ===")
    
    ctx = MockContext()
    server_session = MockServerSession()
    app_context = MockAppContext(config={})
    
    # 测试数据
    data = {
        "GDP增长率": [3.2, 2.8, 3.5, 2.9, 3.1, 2.7, 3.3, 3.0],
        "通货膨胀率": [2.1, 2.3, 1.9, 2.4, 2.2, 2.0, 2.1, 2.3],
        "失业率": [4.5, 4.2, 4.0, 4.3, 4.1, 4.4, 4.2, 4.0]
    }
    
    try:
        result = await descriptive_statistics(
            ctx=Context(server_session, app_context),
            data=data
        )
        print("✅ 描述性统计测试通过")
        print(f"结果类型: {type(result)}")
        print(f"内容长度: {len(result.content)}")
        return True
    except Exception as e:
        print(f"❌ 描述性统计测试失败: {e}")
        return False


async def test_ols_regression():
    """测试OLS回归功能"""
    print("\n=== 测试OLS回归 ===")
    
    ctx = MockContext()
    server_session = MockServerSession()
    app_context = MockAppContext(config={})
    
    # 测试数据
    y_data = [12000, 13500, 11800, 14200, 15100, 14800, 16200, 15900]
    x_data = [
        [800, 5.2],
        [900, 5.8],
        [750, 4.9],
        [1000, 6.1],
        [1100, 6.5],
        [950, 5.9],
        [1200, 7.2],
        [1150, 6.8]
    ]
    
    # 测试1: 不提供feature_names
    try:
        result1 = await ols_regression(
            ctx=Context(server_session, app_context),
            y_data=y_data,
            x_data=x_data
        )
        print("✅ OLS回归测试1（无feature_names）通过")
    except Exception as e:
        print(f"❌ OLS回归测试1失败: {e}")
        return False
    
    # 测试2: 提供feature_names
    try:
        feature_names = ["广告支出", "价格指数"]
        result2 = await ols_regression(
            ctx=Context(server_session, app_context),
            y_data=y_data,
            x_data=x_data,
            feature_names=feature_names
        )
        print("✅ OLS回归测试2（有feature_names）通过")
    except Exception as e:
        print(f"❌ OLS回归测试2失败: {e}")
        return False
    
    return True


async def test_time_series_analysis():
    """测试时间序列分析功能"""
    print("\n=== 测试时间序列分析 ===")
    
    ctx = MockContext()
    server_session = MockServerSession()
    app_context = MockAppContext(config={})
    
    # 测试数据
    time_series_data = [12000, 13500, 11800, 14200, 15100, 14800, 16200, 15900, 16800, 17200]
    
    try:
        result = await time_series_analysis(
            ctx=Context(server_session, app_context),
            data=time_series_data
        )
        print("✅ 时间序列分析测试通过")
        print(f"ADF统计量: {result.structuredContent['adf_statistic']:.4f}")
        print(f"ADF p值: {result.structuredContent['adf_pvalue']:.4f}")
        print(f"是否平稳: {result.structuredContent['stationary']}")
        return True
    except Exception as e:
        print(f"❌ 时间序列分析测试失败: {e}")
        return False


async def test_correlation_analysis():
    """测试相关性分析功能"""
    print("\n=== 测试相关性分析 ===")
    
    ctx = MockContext()
    server_session = MockServerSession()
    app_context = MockAppContext(config={})
    
    # 测试数据
    data = {
        "销售额": [12000, 13500, 11800, 14200, 15100],
        "广告支出": [800, 900, 750, 1000, 1100],
        "价格": [99, 95, 102, 98, 96]
    }
    
    try:
        result = await correlation_analysis(
            ctx=Context(server_session, app_context),
            data=data,
            method="pearson"
        )
        print("✅ 相关性分析测试通过")
        print(f"结果类型: {type(result)}")
        return True
    except Exception as e:
        print(f"❌ 相关性分析测试失败: {e}")
        return False


async def test_error_handling():
    """测试错误处理功能"""
    print("\n=== 测试错误处理 ===")
    
    ctx = MockContext()
    server_session = MockServerSession()
    app_context = MockAppContext(config={})
    
    # 测试空数据
    try:
        result = await descriptive_statistics(
            ctx=Context(server_session, app_context),
            data={}
        )
        print("❌ 空数据测试应该失败但没有失败")
        return False
    except Exception as e:
        print("✅ 空数据错误处理正常")
    
    # 测试不一致的数据长度
    try:
        result = await ols_regression(
            ctx=Context(server_session, app_context),
            y_data=[1, 2, 3],
            x_data=[[1], [2]]  # 长度不一致
        )
        print("❌ 数据长度不一致测试应该失败但没有失败")
        return False
    except Exception as e:
        print("✅ 数据长度不一致错误处理正常")
    
    return True


async def main():
    """主测试函数"""
    print("开始测试修复后的MCP服务功能...")
    
    tests = [
        test_descriptive_statistics,
        test_ols_regression,
        test_time_series_analysis,
        test_correlation_analysis,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if await test():
            passed += 1
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！修复成功！")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步检查")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)