"""
简单测试JSON格式面板数据解析
"""

import json
import sys
sys.path.insert(0, 'd:/aigroup-econ-mcp')

from src.aigroup_econ_mcp.tools.file_parser import FileParser

# 测试JSON数据
json_data = {
    "entity_id": ["A", "A", "A", "B", "B", "B"],
    "time_period": ["2010", "2011", "2012", "2010", "2011", "2012"],
    "x1": [100, 105, 110, 95, 100, 105],
    "y": [150, 155, 160, 145, 150, 155]
}

print("测试JSON面板数据解析...")
print("-" * 60)

# 转为JSON字符串
content = json.dumps(json_data)
print(f"JSON内容: {content[:100]}...")

# 解析
try:
    parsed = FileParser.parse_file_content(content, 'json')
    print(f"\n✅ 解析成功!")
    print(f"数据类型: {parsed['data_type']}")
    print(f"变量: {parsed['variables']}")
    
    # 检查数据
    print(f"\nentity_id数据类型: {type(parsed['data']['entity_id'][0])}")
    print(f"entity_id样本: {parsed['data']['entity_id']}")
    print(f"time_period数据类型: {type(parsed['data']['time_period'][0])}")
    print(f"time_period样本: {parsed['data']['time_period']}")
    
    # 转换为工具格式
    print("\n转换为面板工具格式...")
    tool_data = FileParser.convert_to_tool_format(parsed, 'panel')
    
    print(f"✅ 转换成功!")
    print(f"entity_ids: {tool_data['entity_ids']}")
    print(f"time_periods: {tool_data['time_periods']}")
    print(f"y_data: {tool_data['y_data']}")
    print(f"x_data: {tool_data['x_data']}")
    
except Exception as e:
    print(f"\n❌ 失败: {e}")
    import traceback
    traceback.print_exc()