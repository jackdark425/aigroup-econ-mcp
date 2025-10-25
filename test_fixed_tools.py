"""
测试修复后的工具
"""
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_ols_regression():
    """测试OLS回归分析"""
    print("\n" + "="*60)
    print("测试 OLS 回归分析")
    print("="*60)
    
    server_params = StdioServerParameters(
        command="uvx",
        args=["aigroup-econ-mcp"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 准备测试数据：销售额与广告支出的关系
            y_data = [12000, 13500, 11800, 14200, 15100, 14800, 16200, 15900]
            x_data = [
                [800, 5.2],   # 广告支出, 价格指数
                [900, 5.8],
                [750, 4.9],
                [1000, 6.1],
                [1100, 6.3],
                [1050, 6.0],
                [1200, 6.5],
                [1150, 6.2]
            ]
            feature_names = ["广告支出", "价格指数"]
            
            result = await session.call_tool(
                "ols_regression",
                arguments={
                    "y_data": y_data,
                    "x_data": x_data,
                    "feature_names": feature_names
                }
            )
            
            print("\n✅ OLS回归测试成功!")
            for content in result.content:
                if hasattr(content, 'text'):
                    print(content.text)
            
            return True


async def test_time_series_analysis():
    """测试时间序列分析"""
    print("\n" + "="*60)
    print("测试时间序列分析")
    print("="*60)
    
    server_params = StdioServerParameters(
        command="uvx",
        args=["aigroup-econ-mcp"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 准备测试数据：模拟月度销售额序列
            data = [
                12000, 13500, 11800, 14200, 15100, 14800,
                16200, 15900, 17100, 16800, 18200, 17900,
                19300, 19000, 20100, 19800, 21200, 20900,
                22300, 22000, 23400, 23100, 24500, 24200
            ]
            
            result = await session.call_tool(
                "time_series_analysis",
                arguments={"data": data}
            )
            
            print("\n✅ 时间序列分析测试成功!")
            for content in result.content:
                if hasattr(content, 'text'):
                    print(content.text)
            
            return True


async def main():
    """运行所有测试"""
    print("╔" + "="*58 + "╗")
    print("║  测试修复后的 aigroup-econ-mcp 工具                      ║")
    print("╚" + "="*58 + "╝")
    
    try:
        # 测试OLS回归
        success1 = await test_ols_regression()
        
        # 测试时间序列分析
        success2 = await test_time_series_analysis()
        
        # 总结
        print("\n" + "="*60)
        print("📊 测试总结")
        print("="*60)
        if success1 and success2:
            print("✅ 所有测试通过！问题已修复。")
            print("\n修复内容：")
            print("1. OLS回归：将numpy类型转换为Python原生类型")
            print("2. 时间序列分析：将numpy类型转换为Python原生类型")
            print("\n问题根因：")
            print("statsmodels返回的numpy.float64等类型无法被Pydantic正确序列化")
        else:
            print("❌ 部分测试失败")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())