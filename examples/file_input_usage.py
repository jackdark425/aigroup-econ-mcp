"""
文件输入功能使用示例
演示如何使用CSV和JSON文件作为MCP工具的输入
"""

import asyncio
import json


# 示例1: CSV文件内容（描述性统计）
csv_example_descriptive = """
GDP增长率,通货膨胀率,失业率
3.2,2.1,4.5
2.8,2.3,4.2
3.5,1.9,4.0
2.9,2.4,4.3
3.1,2.2,4.1
2.7,2.5,4.4
3.3,2.0,3.9
3.0,2.3,4.2
"""

# 示例2: JSON文件内容（描述性统计）
json_example_descriptive = {
    "GDP增长率": [3.2, 2.8, 3.5, 2.9, 3.1, 2.7, 3.3, 3.0],
    "通货膨胀率": [2.1, 2.3, 1.9, 2.4, 2.2, 2.5, 2.0, 2.3],
    "失业率": [4.5, 4.2, 4.0, 4.3, 4.1, 4.4, 3.9, 4.2]
}

# 示例3: CSV文件内容（回归分析）
csv_example_regression = """
广告支出,价格,竞争对手数量,销售额
800,99,3,12000
900,95,3,13500
750,102,4,11800
1000,98,3,14200
850,96,4,13800
950,94,3,15100
"""

# 示例4: JSON文件内容（回归分析）
json_example_regression = {
    "广告支出": [800, 900, 750, 1000, 850, 950],
    "价格": [99, 95, 102, 98, 96, 94],
    "竞争对手数量": [3, 3, 4, 3, 4, 3],
    "销售额": [12000, 13500, 11800, 14200, 13800, 15100]
}

# 示例5: 时间序列数据
csv_example_timeseries = """
date,stock_price
2024-01-01,100.5
2024-01-02,102.3
2024-01-03,101.8
2024-01-04,103.5
2024-01-05,104.2
2024-01-06,103.8
2024-01-07,105.1
"""

# 示例6: 面板数据
csv_example_panel = """
company_id,year,revenue,employees,investment
1,2020,1000,50,100
1,2021,1100,52,110
1,2022,1200,55,120
2,2020,800,40,80
2,2021,900,42,90
2,2022,1000,45,100
3,2020,1200,60,150
3,2021,1300,62,160
3,2022,1400,65,170
"""


def demo_usage():
    """演示如何在MCP中使用文件输入"""
    
    print("=" * 60)
    print("文件输入功能使用示例")
    print("=" * 60)
    
    # 示例1: 使用CSV进行描述性统计
    print("\n📊 示例1: CSV格式 - 描述性统计分析")
    print("-" * 60)
    print("工具调用: descriptive_statistics")
    print("\n参数:")
    print(f"  file_content: {csv_example_descriptive[:100]}...")
    print(f"  file_format: 'csv'")
    print("\n✅ 系统会自动:")
    print("  - 检测CSV分隔符")
    print("  - 识别列名（GDP增长率, 通货膨胀率, 失业率）")
    print("  - 解析数值数据")
    print("  - 计算描述性统计")
    
    # 示例2: 使用JSON进行描述性统计
    print("\n📊 示例2: JSON格式 - 描述性统计分析")
    print("-" * 60)
    print("工具调用: descriptive_statistics")
    print("\n参数:")
    print(f"  file_content: {json.dumps(json_example_descriptive, ensure_ascii=False, indent=2)[:150]}...")
    print(f"  file_format: 'json' (或 'auto')")
    print("\n✅ 系统会自动识别JSON格式并解析")
    
    # 示例3: 使用CSV进行回归分析
    print("\n📈 示例3: CSV格式 - OLS回归分析")
    print("-" * 60)
    print("工具调用: ols_regression")
    print("\n参数:")
    print(f"  file_content: {csv_example_regression[:100]}...")
    print(f"  file_format: 'csv'")
    print("\n✅ 系统会自动:")
    print("  - 识别因变量（最后一列：销售额）")
    print("  - 识别自变量（前面的列：广告支出, 价格, 竞争对手数量）")
    print("  - 构建回归模型")
    
    # 示例4: 时间序列分析
    print("\n📉 示例4: CSV格式 - 时间序列分析")
    print("-" * 60)
    print("工具调用: time_series_analysis")
    print("\n参数:")
    print(f"  file_content: {csv_example_timeseries[:80]}...")
    print(f"  file_format: 'csv'")
    print("\n✅ 系统会自动:")
    print("  - 识别时间列（date）")
    print("  - 提取数值序列（stock_price）")
    print("  - 进行平稳性检验和自相关分析")
    
    # 示例5: 面板数据分析
    print("\n🏢 示例5: CSV格式 - 面板数据分析")
    print("-" * 60)
    print("工具调用: panel_fixed_effects")
    print("\n参数:")
    print(f"  file_content: {csv_example_panel[:100]}...")
    print(f"  file_format: 'csv'")
    print("\n✅ 系统会自动:")
    print("  - 识别实体ID（company_id）")
    print("  - 识别时间标识（year）")
    print("  - 识别因变量和自变量")
    print("  - 执行面板数据模型")
    
    print("\n" + "=" * 60)
    print("💡 使用提示:")
    print("=" * 60)
    print("1. file_format参数可以设置为'auto'，系统会自动检测")
    print("2. CSV文件建议包含列名表头，以便更好地识别变量")
    print("3. JSON支持多种格式，包括字典和数组形式")
    print("4. 系统会根据变量名智能识别数据类型（时间序列/面板/回归）")
    print("5. 所有原有的直接数据输入方式仍然完全支持")
    print("=" * 60)


def show_file_formats():
    """展示支持的文件格式"""
    
    print("\n📁 支持的文件格式详解")
    print("=" * 60)
    
    print("\n1️⃣ CSV格式:")
    print("-" * 60)
    print("格式1 - 带表头（推荐）:")
    print("""
变量1,变量2,变量3
1.2,3.4,5.6
2.3,4.5,6.7
3.4,5.6,7.8
""")
    
    print("格式2 - 无表头（自动生成列名）:")
    print("""
1.2,3.4,5.6
2.3,4.5,6.7
3.4,5.6,7.8
""")
    print("→ 自动命名为: var1, var2, var3")
    
    print("\n2️⃣ JSON格式:")
    print("-" * 60)
    print("格式1 - 字典形式（推荐）:")
    print("""
{
    "变量1": [1.2, 2.3, 3.4],
    "变量2": [3.4, 4.5, 5.6],
    "变量3": [5.6, 6.7, 7.8]
}
""")
    
    print("格式2 - 数组形式:")
    print("""
[
    {"变量1": 1.2, "变量2": 3.4, "变量3": 5.6},
    {"变量1": 2.3, "变量2": 4.5, "变量3": 6.7},
    {"变量1": 3.4, "变量2": 5.6, "变量3": 7.8}
]
""")
    
    print("格式3 - 嵌套结构:")
    print("""
{
    "data": {
        "变量1": [1.2, 2.3, 3.4],
        "变量2": [3.4, 4.5, 5.6]
    },
    "metadata": {
        "description": "示例数据"
    }
}
""")


def show_variable_recognition():
    """展示智能变量识别功能"""
    
    print("\n🤖 智能变量识别功能")
    print("=" * 60)
    
    print("\n📌 时间序列自动识别:")
    print("包含以下关键词的列名会被识别为时间标识：")
    print("  • time, date, year, month, day, period, quarter")
    
    print("\n📌 实体ID自动识别:")
    print("包含以下关键词的列名会被识别为实体标识：")
    print("  • id, entity, firm, company, country, region")
    
    print("\n📌 回归分析变量分配:")
    print("  • 最后一列 → 因变量 (y)")
    print("  • 其他数值列 → 自变量 (x)")
    
    print("\n📌 面板数据变量分配:")
    print("  1. 识别实体ID列")
    print("  2. 识别时间标识列")
    print("  3. 最后一个数据列 → 因变量")
    print("  4. 其他数据列 → 自变量")
    
    print("\n💡 示例:")
    print("-" * 60)
    print("CSV文件:")
    print("""
company_id,year,sales,advertising,price
1,2020,1000,100,50
1,2021,1100,110,48
2,2020,900,90,52
2,2021,1000,95,50
""")
    
    print("\n自动识别结果:")
    print("  ✓ 实体ID: company_id")
    print("  ✓ 时间标识: year")
    print("  ✓ 因变量: price")
    print("  ✓ 自变量: sales, advertising")
    print("  ✓ 数据类型: 面板数据")
    print("  ✓ 推荐工具: panel_fixed_effects, panel_random_effects")


if __name__ == "__main__":
    demo_usage()
    show_file_formats()
    show_variable_recognition()
    
    print("\n" + "=" * 60)
    print("✨ 完成！现在你可以使用CSV/JSON文件加速数据分析了！")
    print("=" * 60)