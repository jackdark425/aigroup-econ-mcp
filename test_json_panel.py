"""
测试JSON格式的面板数据工具
"""

import json
from src.aigroup_econ_mcp.tools.file_parser import FileParser

def test_json_panel_parsing():
    """测试JSON面板数据解析"""
    print("=" * 80)
    print("测试JSON格式面板数据解析")
    print("=" * 80)
    
    # 读取JSON文件
    with open('test_panel_data.json', 'r') as f:
        content = f.read()
    
    # 解析JSON
    parsed = FileParser.parse_file_content(content, 'json')
    
    print(f"\n检测到的数据类型: {parsed['data_type']}")
    print(f"变量列表: {parsed['variables']}")
    print(f"观测数量: {parsed['n_observations']}")
    print(f"变量数量: {parsed['n_variables']}")
    
    # 查看解析后的数据
    print(f"\n解析后的数据样本:")
    for var in parsed['variables']:
        data_sample = parsed['data'][var][:3]
        data_type = type(data_sample[0]).__name__
        print(f"  {var}: {data_sample} (类型: {data_type})")
    
    # 转换为面板工具格式
    print("\n" + "=" * 80)
    print("转换为面板工具格式")
    print("=" * 80)
    
    tool_data = FileParser.convert_to_tool_format(parsed, 'panel')
    
    print(f"\n转换结果:")
    print(f"  y_data样本: {tool_data['y_data'][:3]}")
    print(f"  x_data样本: {tool_data['x_data'][:3]}")
    print(f"  entity_ids样本: {tool_data['entity_ids'][:5]}")
    print(f"  time_periods样本: {tool_data['time_periods'][:5]}")
    print(f"  feature_names: {tool_data['feature_names']}")
    
    # 验证数据类型
    print(f"\n数据类型验证:")
    print(f"  entity_ids[0]类型: {type(tool_data['entity_ids'][0])}")
    print(f"  time_periods[0]类型: {type(tool_data['time_periods'][0])}")
    print(f"  y_data[0]类型: {type(tool_data['y_data'][0])}")
    
    return tool_data

def test_panel_tools_with_json():
    """测试面板数据工具与JSON文件"""
    from src.aigroup_econ_mcp.tools.panel_data import (
        fixed_effects_model,
        random_effects_model,
        hausman_test,
        panel_unit_root_test
    )
    
    print("\n" + "=" * 80)
    print("测试面板数据工具（JSON格式）")
    print("=" * 80)
    
    # 读取并解析JSON
    with open('test_panel_data.json', 'r') as f:
        content = f.read()
    
    parsed = FileParser.parse_file_content(content, 'json')
    tool_data = FileParser.convert_to_tool_format(parsed, 'panel')
    
    # 测试1: 固定效应模型
    print("\n[1/4] 测试 panel_fixed_effects...")
    try:
        fe_result = fixed_effects_model(
            y_data=tool_data['y_data'],
            X_data=tool_data['x_data'],
            entity_ids=tool_data['entity_ids'],
            time_periods=tool_data['time_periods'],
            feature_names=tool_data['feature_names']
        )
        print(f"✅ 固定效应模型成功: R²={fe_result.rsquared:.4f}")
    except Exception as e:
        print(f"❌ 固定效应模型失败: {e}")
    
    # 测试2: 随机效应模型
    print("\n[2/4] 测试 panel_random_effects...")
    try:
        re_result = random_effects_model(
            y_data=tool_data['y_data'],
            X_data=tool_data['x_data'],
            entity_ids=tool_data['entity_ids'],
            time_periods=tool_data['time_periods'],
            feature_names=tool_data['feature_names']
        )
        print(f"✅ 随机效应模型成功: R²={re_result.rsquared:.4f}")
    except Exception as e:
        print(f"❌ 随机效应模型失败: {e}")
    
    # 测试3: Hausman检验
    print("\n[3/4] 测试 panel_hausman_test...")
    try:
        hausman_result = hausman_test(
            y_data=tool_data['y_data'],
            X_data=tool_data['x_data'],
            entity_ids=tool_data['entity_ids'],
            time_periods=tool_data['time_periods'],
            feature_names=tool_data['feature_names']
        )
        print(f"✅ Hausman检验成功: p={hausman_result.p_value:.4f}")
        print(f"   建议: {hausman_result.recommendation}")
    except Exception as e:
        print(f"❌ Hausman检验失败: {e}")
    
    # 测试4: 面板单位根检验
    print("\n[4/4] 测试 panel_unit_root_test...")
    try:
        unit_root_result = panel_unit_root_test(
            data=tool_data['y_data'],
            entity_ids=tool_data['entity_ids'],
            time_periods=tool_data['time_periods']
        )
        print(f"✅ 面板单位根检验成功: {'平稳' if unit_root_result.stationary else '非平稳'}")
    except Exception as e:
        print(f"❌ 面板单位根检验失败: {e}")

if __name__ == "__main__":
    # 测试JSON解析
    tool_data = test_json_panel_parsing()
    
    # 测试面板数据工具
    test_panel_tools_with_json()
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)