"""
MCP客户端使用示例
展示如何通过MCP协议调用计量经济学工具
"""

import asyncio
import json
from typing import Dict, List, Any

# 注意：这只是一个模拟示例，实际使用需要通过MCP协议
# 可以使用mcp库连接到运行的MCP服务器

class MockMCPClient:
    """模拟MCP客户端"""

    def __init__(self):
        self.connected = False

    async def connect(self):
        """连接到MCP服务器"""
        print("🔌 连接到MCP服务器...")
        self.connected = True
        print("✅ 连接成功")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具"""
        print(f"🔧 调用工具: {tool_name}")
        print(f"📝 参数: {json.dumps(arguments, indent=2, ensure_ascii=False)}")

        # 这里应该通过实际的MCP协议调用
        # 为了演示，我们返回模拟结果
        return await self._simulate_tool_call(tool_name, arguments)

    async def _simulate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """模拟工具调用结果"""
        if tool_name == "descriptive_statistics":
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "描述性统计结果：\n均值: 120.0000\n标准差: 10.0000\n最小值: 110.0000\n最大值: 130.0000"
                    }
                ],
                "structuredContent": {
                    "count": 5,
                    "mean": 120.0,
                    "std": 10.0,
                    "min": 110.0,
                    "max": 130.0,
                    "median": 120.0,
                    "skewness": 0.0,
                    "kurtosis": -1.2
                }
            }

        elif tool_name == "ols_regression":
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "OLS回归分析结果：\nR² = 0.8500\n调整R² = 0.8200\nF统计量 = 25.5000 (p = 0.0000)"
                    }
                ],
                "structuredContent": {
                    "rsquared": 0.85,
                    "rsquared_adj": 0.82,
                    "f_statistic": 25.5,
                    "f_pvalue": 0.0000,
                    "aic": 45.2,
                    "bic": 48.1,
                    "coefficients": {
                        "const": {"coef": 50.0, "p_value": 0.0001},
                        "advertising": {"coef": 3.5, "p_value": 0.002},
                        "price": {"coef": -0.8, "p_value": 0.015}
                    }
                }
            }

        elif tool_name == "hypothesis_testing":
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "t检验结果：\n检验统计量 = 2.4500\np值 = 0.0200\n显著 (5%水平)"
                    }
                ],
                "structuredContent": {
                    "test_type": "t_test",
                    "statistic": 2.45,
                    "p_value": 0.02,
                    "significant": True,
                    "confidence_interval": [95.5, 104.5]
                }
            }

        elif tool_name == "time_series_analysis":
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "时间序列分析结果：\nADF检验统计量 = -3.2400\nADF检验p值 = 0.0150\n平稳序列"
                    }
                ],
                "structuredContent": {
                    "adf_statistic": -3.24,
                    "adf_pvalue": 0.015,
                    "stationary": True,
                    "acf": [1.0, 0.8, 0.6, 0.4, 0.2],
                    "pacf": [1.0, 0.7, 0.3, 0.1, -0.1]
                }
            }

        else:
            return {
                "content": [{"type": "text", "text": f"未知工具: {tool_name}"}],
                "isError": True
            }

    async def disconnect(self):
        """断开连接"""
        print("🔌 断开MCP服务器连接")
        self.connected = False


async def demonstrate_mcp_usage():
    """演示MCP使用"""
    print("🤖 MCP客户端使用示例")
    print("=" * 50)

    client = MockMCPClient()

    try:
        # 连接服务器
        await client.connect()

        # 示例1: 描述性统计
        print("\n📊 示例1: 描述性统计分析")
        result = await client.call_tool("descriptive_statistics", {
            "data": {
                "销售额": [120, 135, 118, 142, 155, 160, 148, 175],
                "广告支出": [8, 9, 7.5, 10, 11, 12, 10.5, 13],
                "价格": [100, 98, 102, 97, 95, 94, 96, 93]
            }
        })

        print("结果:")
        if result.get("structuredContent"):
            stats = result["structuredContent"]
            print(f"  均值: {stats['mean']:.2f}")
            print(f"  标准差: {stats['std']:.2f}")
            print(f"  最小值: {stats['min']:.2f}")
            print(f"  最大值: {stats['max']:.2f}")

        # 示例2: 回归分析
        print("\n📈 示例2: OLS回归分析")
        result = await client.call_tool("ols_regression", {
            "y_data": [120, 135, 118, 142, 155, 160, 148, 175],
            "x_data": [
                [8, 100], [9, 98], [7.5, 102], [10, 97],
                [11, 95], [12, 94], [10.5, 96], [13, 93]
            ],
            "feature_names": ["advertising", "price"]
        })

        print("结果:")
        if result.get("structuredContent"):
            reg = result["structuredContent"]
            print(f"  R² = {reg['rsquared']:.4f}")
            print(f"  调整R² = {reg['rsquared_adj']:.4f}")
            print(f"  F统计量 = {reg['f_statistic']:.4f} (p值 = {reg['f_pvalue']:.4f})")
            print("  回归系数:")
            for var, coef in reg["coefficients"].items():
                print(f"    {var}: {coef['coef']:.4f} (p值 = {coef['p_value']:.4f})")

        # 示例3: 假设检验
        print("\n🧪 示例3: 假设检验")
        result = await client.call_tool("hypothesis_testing", {
            "data1": [100, 110, 120, 115, 125, 130, 128, 135],
            "data2": [95, 105, 115, 120, 130, 135, 140, 145],
            "test_type": "t_test"
        })

        print("结果:")
        if result.get("structuredContent"):
            test = result["structuredContent"]
            print(f"  检验类型: {test['test_type']}")
            print(f"  统计量 = {test['statistic']:.4f}")
            print(f"  p值 = {test['p_value']:.4f}")
            print(f"  显著性 = {'是' if test['significant'] else '否'}")
            if test.get("confidence_interval"):
                print(f"  95%置信区间: [{test['confidence_interval'][0]:.2f}, {test['confidence_interval'][1]:.2f}]")

        # 示例4: 时间序列分析
        print("\n⏰ 示例4: 时间序列分析")
        result = await client.call_tool("time_series_analysis", {
            "data": [100, 110, 120, 115, 125, 130, 128, 135, 140, 145, 150, 155]
        })

        print("结果:")
        if result.get("structuredContent"):
            ts = result["structuredContent"]
            print(f"  ADF检验统计量 = {ts['adf_statistic']:.4f}")
            print(f"  ADF检验p值 = {ts['adf_pvalue']:.4f}")
            print(f"  是否平稳: {'是' if ts['stationary'] else '否'}")
            print(f"  ACF前5阶: {[f'{x:.3f}' for x in ts['acf'][:5]]}")
            print(f"  PACF前5阶: {[f'{x:.3f}' for x in ts['pacf'][:5]]}")

        print("\n✅ 所有示例执行完成！")

    finally:
        await client.disconnect()


async def demonstrate_natural_language_interface():
    """演示自然语言界面"""
    print("\n🗣️  自然语言界面示例")
    print("=" * 50)
    print("用户可以通过自然语言向AI助手提问，AI助手调用MCP工具获取结果")

    # 模拟用户查询和AI助手的工具调用
    queries_and_tools = [
        {
            "query": "帮我分析一下公司的销售数据",
            "tools": [
                {"name": "descriptive_statistics", "params": {"data": {"销售额": [100, 120, 150, 130, 180, 200, 190, 220]}}},
                {"name": "time_series_analysis", "params": {"data": [100, 120, 150, 130, 180, 200, 190, 220]}}
            ]
        },
        {
            "query": "分析广告投入对销售额的影响",
            "tools": [
                {"name": "ols_regression", "params": {
                    "y_data": [120, 135, 150, 160, 175, 190, 200, 220],
                    "x_data": [[8, 9, 10, 11, 12, 13, 14, 15]],
                    "feature_names": ["advertising"]
                }},
                {"name": "hypothesis_testing", "params": {
                    "data1": [120, 135, 150, 160, 175, 190, 200, 220],
                    "data2": [100, 110, 125, 135, 145, 155, 165, 175],
                    "test_type": "t_test"
                }}
            ]
        },
        {
            "query": "检查这个时间序列是否平稳",
            "tools": [
                {"name": "time_series_analysis", "params": {"data": [100, 102, 98, 105, 103, 108, 106, 112, 110, 115, 113, 118]}},
                {"name": "hypothesis_testing", "params": {
                    "data1": [100, 102, 98, 105, 103, 108, 106, 112, 110, 115, 113, 118],
                    "test_type": "adf"
                }}
            ]
        }
    ]

    for i, example in enumerate(queries_and_tools, 1):
        print(f"\n{i}. 用户查询: {example['query']}")
        print("   AI助手调用的工具:")

        client = MockMCPClient()
        await client.connect()

        for tool in example['tools']:
            print(f"   - {tool['name']}")
            result = await client.call_tool(tool['name'], tool['params'])
            if result.get("structuredContent"):
                print(f"     结构化结果: {result['structuredContent']}")

        await client.disconnect()
        print("   AI助手整合结果并用自然语言回复用户")


async def main():
    """主函数"""
    print("🎯 MCP客户端演示")
    print("展示如何通过MCP协议使用计量经济学工具")

    # 基本工具调用演示
    await demonstrate_mcp_usage()

    # 自然语言界面演示
    await demonstrate_natural_language_interface()

    print("\n📚 实际使用说明:")
    print("1. 启动MCP服务器: uvx aigroup-econ-mcp --port 8000")
    print("2. 配置Claude Desktop或其他MCP客户端")
    print("3. 通过自然语言使用各种计量经济学分析功能")
    print("4. 获得结构化的统计结果和专业解释")


if __name__ == "__main__":
    asyncio.run(main())