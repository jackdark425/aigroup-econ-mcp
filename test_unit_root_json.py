"""
测试JSON格式的面板单位根检验（更多数据点）
"""

import json
import sys
sys.path.insert(0, 'd:/aigroup-econ-mcp')

from src.aigroup_econ_mcp.tools.file_parser import FileParser
from src.aigroup_econ_mcp.tools.panel_data import panel_unit_root_test

# 准备更多数据点的测试数据（每个实体8个时间点）
json_data = {
    "entity_id": ["A"]*8 + ["B"]*8 + ["C"]*8,
    "time_period": ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017"] * 3,
    "y": [150, 155, 160, 165, 170, 175, 180, 185,  # A
          145, 150, 155, 160, 165, 170, 175, 180,  # B
          148, 153, 158, 163, 168, 173, 178, 183]  # C
}

print("=" * 80)
print("JSON格式面板单位根检验测试（扩展数据）")
print("=" * 80)

# 解析JSON
content = json.dumps(json_data)
parsed = FileParser.parse_file_content(content, 'json')
tool_data = FileParser.convert_to_tool_format(parsed, 'panel')

print(f"\n数据准备完成:")
print(f"  - 实体数: {len(set(tool_data['entity_ids']))}")
print(f"  - 每个实体时间点数: {len(tool_data['entity_ids']) // len(set(tool_data['entity_ids']))}")
print(f"  - 总观测数: {len(tool_data['y_data'])}")

print("\n" + "-" * 80)
print("测试 panel_unit_root_test (JSON格式, 8个时间点)")
print("-" * 80)

try:
    unit_root_result = panel_unit_root_test(
        data=tool_data['y_data'],
        entity_ids=tool_data['entity_ids'],
        time_periods=tool_data['time_periods']
    )
    print(f"✅ 成功!")
    print(f"   统计量: {unit_root_result.statistic:.4f}")
    print(f"   p值: {unit_root_result.p_value:.4f}")
    print(f"   结果: {'平稳' if unit_root_result.stationary else '非平稳'}")
    print(f"   检验类型: {unit_root_result.test_type}")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✅ JSON格式面板单位根检验测试完成！")
print("=" * 80)