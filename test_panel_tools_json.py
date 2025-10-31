"""
测试JSON格式的四个面板数据工具
"""

import json
import sys
sys.path.insert(0, 'd:/aigroup-econ-mcp')

from src.aigroup_econ_mcp.tools.file_parser import FileParser
from src.aigroup_econ_mcp.tools.panel_data import (
    fixed_effects_model,
    random_effects_model,
    hausman_test,
    panel_unit_root_test
)

# 准备测试数据
json_data = {
    "entity_id": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C"],
    "time_period": ["2010", "2011", "2012", "2013", "2014", "2010", "2011", "2012", "2013", "2014", "2010", "2011", "2012", "2013", "2014"],
    "x1": [100, 105, 110, 115, 120, 95, 100, 105, 110, 115, 98, 103, 108, 113, 118],
    "x2": [3.5, 4.2, 4.8, 5.1, 5.5, 3.2, 3.8, 4.1, 4.5, 4.9, 3.3, 3.9, 4.3, 4.7, 5.0],
    "y": [150, 155, 160, 165, 170, 145, 150, 155, 160, 165, 148, 153, 158, 163, 168]
}

print("=" * 80)
print("JSON格式面板数据工具测试")
print("=" * 80)

# 解析JSON
content = json.dumps(json_data)
parsed = FileParser.parse_file_content(content, 'json')
tool_data = FileParser.convert_to_tool_format(parsed, 'panel')

print(f"\n数据准备完成:")
print(f"  - 实体数: {len(set(tool_data['entity_ids']))}")
print(f"  - 时间点数: {len(set(tool_data['time_periods']))}")
print(f"  - 总观测数: {len(tool_data['y_data'])}")
print(f"  - 特征数: {len(tool_data['feature_names'])}")

# 测试1: 固定效应模型
print("\n" + "-" * 80)
print("[1/4] 测试 panel_fixed_effects (JSON格式)")
print("-" * 80)
try:
    fe_result = fixed_effects_model(
        y_data=tool_data['y_data'],
        X_data=tool_data['x_data'],
        entity_ids=tool_data['entity_ids'],
        time_periods=tool_data['time_periods'],
        feature_names=tool_data['feature_names']
    )
    print(f"✅ 成功! R²={fe_result.rsquared:.4f}, AIC={fe_result.aic:.2f}")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试2: 随机效应模型
print("\n" + "-" * 80)
print("[2/4] 测试 panel_random_effects (JSON格式)")
print("-" * 80)
try:
    re_result = random_effects_model(
        y_data=tool_data['y_data'],
        X_data=tool_data['x_data'],
        entity_ids=tool_data['entity_ids'],
        time_periods=tool_data['time_periods'],
        feature_names=tool_data['feature_names']
    )
    print(f"✅ 成功! R²={re_result.rsquared:.4f}, AIC={re_result.aic:.2f}")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试3: Hausman检验
print("\n" + "-" * 80)
print("[3/4] 测试 panel_hausman_test (JSON格式)")
print("-" * 80)
try:
    hausman_result = hausman_test(
        y_data=tool_data['y_data'],
        X_data=tool_data['x_data'],
        entity_ids=tool_data['entity_ids'],
        time_periods=tool_data['time_periods'],
        feature_names=tool_data['feature_names']
    )
    print(f"✅ 成功! p值={hausman_result.p_value:.4f}")
    print(f"   建议: {hausman_result.recommendation}")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试4: 面板单位根检验
print("\n" + "-" * 80)
print("[4/4] 测试 panel_unit_root_test (JSON格式)")
print("-" * 80)
try:
    unit_root_result = panel_unit_root_test(
        data=tool_data['y_data'],
        entity_ids=tool_data['entity_ids'],
        time_periods=tool_data['time_periods']
    )
    print(f"✅ 成功! 结果: {'平稳' if unit_root_result.stationary else '非平稳'}")
    print(f"   p值={unit_root_result.p_value:.4f}")
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("测试总结")
print("=" * 80)
print("✅ JSON格式的面板数据工具测试完成！")