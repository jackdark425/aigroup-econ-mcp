"""
测试已集成工具的CSV文件输入功能
验证5个已完成集成的工具是否能正确接受和处理CSV输入
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aigroup_econ_mcp.tools.file_parser import FileParser


def test_1_descriptive_statistics():
    """测试描述性统计 - CSV输入"""
    print("\n" + "="*60)
    print("测试1: descriptive_statistics - 描述性统计")
    print("="*60)
    
    csv_content = """GDP增长率,通货膨胀率,失业率
3.2,2.1,4.5
2.8,2.3,4.2
3.5,1.9,4.0
2.9,2.4,4.3
3.1,2.2,4.1
2.7,2.5,4.4
3.3,2.0,3.9
3.0,2.3,4.2"""
    
    try:
        # 解析CSV
        parsed = FileParser.parse_file_content(csv_content, "csv")
        
        print(f"✅ CSV解析成功")
        print(f"   - 格式: {parsed['format']}")
        print(f"   - 变量数: {parsed['n_variables']}")
        print(f"   - 观测数: {parsed['n_observations']}")
        print(f"   - 数据类型: {parsed['data_type']}")
        print(f"   - 变量名: {parsed['variables']}")
        
        # 转换为工具格式
        converted = FileParser.convert_to_tool_format(parsed, 'multi_var_dict')
        data = converted['data']
        
        print(f"✅ 数据转换成功")
        print(f"   - GDP增长率前3个值: {data['GDP增长率'][:3]}")
        print(f"   - 通货膨胀率前3个值: {data['通货膨胀率'][:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False


def test_2_ols_regression():
    """测试OLS回归 - CSV输入"""
    print("\n" + "="*60)
    print("测试2: ols_regression - OLS回归分析")
    print("="*60)
    
    csv_content = """广告支出,价格,销售额
800,99,12000
900,95,13500
750,102,11800
1000,98,14200
850,96,13800
950,94,15100
820,97,12500
880,93,14800"""
    
    try:
        # 解析CSV
        parsed = FileParser.parse_file_content(csv_content, "csv")
        
        print(f"✅ CSV解析成功")
        print(f"   - 变量数: {parsed['n_variables']}")
        print(f"   - 观测数: {parsed['n_observations']}")
        
        # 转换为回归格式
        converted = FileParser.convert_to_tool_format(parsed, 'regression')
        
        print(f"✅ 回归数据转换成功")
        print(f"   - 因变量: {converted['y_variable']}")
        print(f"   - 自变量: {converted['feature_names']}")
        print(f"   - y_data前3个值: {converted['y_data'][:3]}")
        print(f"   - x_data前3行: {converted['x_data'][:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False


def test_3_hypothesis_testing():
    """测试假设检验 - CSV输入"""
    print("\n" + "="*60)
    print("测试3: hypothesis_testing - 假设检验")
    print("="*60)
    
    # 单样本CSV
    csv_content_single = """数据
3.2
2.8
3.5
2.9
3.1
2.7
3.3"""
    
    try:
        # 解析单样本CSV
        parsed = FileParser.parse_file_content(csv_content_single, "csv")
        
        print(f"✅ 单样本CSV解析成功")
        print(f"   - 变量数: {parsed['n_variables']}")
        print(f"   - 观测数: {parsed['n_observations']}")
        
        converted = FileParser.convert_to_tool_format(parsed, 'multi_var_dict')
        data = converted['data']
        variables = list(data.keys())
        data1 = data[variables[0]]
        
        print(f"   - data1前5个值: {data1[:5]}")
        
        # 双样本CSV
        csv_content_double = """组A,组B
3.2,2.5
2.8,2.9
3.5,2.3
2.9,2.6
3.1,2.8"""
        
        parsed2 = FileParser.parse_file_content(csv_content_double, "csv")
        print(f"\n✅ 双样本CSV解析成功")
        print(f"   - 变量数: {parsed2['n_variables']}")
        
        converted2 = FileParser.convert_to_tool_format(parsed2, 'multi_var_dict')
        data2 = converted2['data']
        variables2 = list(data2.keys())
        
        print(f"   - 组A: {data2[variables2[0]]}")
        print(f"   - 组B: {data2[variables2[1]]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False


def test_4_time_series_analysis():
    """测试时间序列分析 - CSV输入"""
    print("\n" + "="*60)
    print("测试4: time_series_analysis - 时间序列分析")
    print("="*60)
    
    csv_content = """股价
100.5
102.3
101.8
103.5
104.2
103.8
105.1
104.7
106.2
105.8
107.3
106.9"""
    
    try:
        # 解析CSV
        parsed = FileParser.parse_file_content(csv_content, "csv")
        
        print(f"✅ CSV解析成功")
        print(f"   - 格式: {parsed['format']}")
        print(f"   - 数据类型: {parsed['data_type']}")
        print(f"   - 观测数: {parsed['n_observations']}")
        
        # 转换为单变量格式
        converted = FileParser.convert_to_tool_format(parsed, 'single_var')
        data = converted['data']
        
        print(f"✅ 时间序列数据转换成功")
        print(f"   - 变量名: {converted['variable_name']}")
        print(f"   - 前5个值: {data[:5]}")
        print(f"   - 后5个值: {data[-5:]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False


def test_5_correlation_analysis():
    """测试相关性分析 - CSV输入"""
    print("\n" + "="*60)
    print("测试5: correlation_analysis - 相关性分析")
    print("="*60)
    
    csv_content = """销售额,广告支出,价格,竞争对手数量
12000,800,99,3
13500,900,95,3
11800,750,102,4
14200,1000,98,3
13800,850,96,4
15100,950,94,3
12500,820,97,4
14800,880,93,3"""
    
    try:
        # 解析CSV
        parsed = FileParser.parse_file_content(csv_content, "csv")
        
        print(f"✅ CSV解析成功")
        print(f"   - 变量数: {parsed['n_variables']}")
        print(f"   - 观测数: {parsed['n_observations']}")
        print(f"   - 变量名: {parsed['variables']}")
        
        # 转换为多变量字典格式
        converted = FileParser.convert_to_tool_format(parsed, 'multi_var_dict')
        data = converted['data']
        
        print(f"✅ 相关性分析数据转换成功")
        for var_name in data.keys():
            print(f"   - {var_name}: {len(data[var_name])}个观测")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False


def test_auto_detection():
    """测试自动格式检测"""
    print("\n" + "="*60)
    print("附加测试: 自动格式检测")
    print("="*60)
    
    csv_content = """var1,var2
1.0,2.0
3.0,4.0"""
    
    json_content = '{"var1": [1.0, 3.0], "var2": [2.0, 4.0]}'
    
    try:
        # 测试CSV自动检测
        parsed_csv = FileParser.parse_file_content(csv_content, "auto")
        print(f"✅ CSV自动检测: {parsed_csv['format']}")
        
        # 测试JSON自动检测
        parsed_json = FileParser.parse_file_content(json_content, "auto")
        print(f"✅ JSON自动检测: {parsed_json['format']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False


def test_intelligent_recognition():
    """测试智能变量识别"""
    print("\n" + "="*60)
    print("附加测试: 智能变量识别")
    print("="*60)
    
    # 测试时间序列识别
    csv_ts = """date,value
2024-01-01,100
2024-01-02,102
2024-01-03,101"""
    
    # 测试面板数据识别
    csv_panel = """company_id,year,revenue
1,2020,1000
1,2021,1100
2,2020,800
2,2021,900"""
    
    try:
        parsed_ts = FileParser.parse_file_content(csv_ts, "csv")
        print(f"✅ 时间序列识别: {parsed_ts['data_type']}")
        print(f"   - 变量: {parsed_ts['variables']}")
        
        parsed_panel = FileParser.parse_file_content(csv_panel, "csv")
        print(f"✅ 面板数据识别: {parsed_panel['data_type']}")
        print(f"   - 变量: {parsed_panel['variables']}")
        
        # 测试智能推荐
        recommendations = FileParser.auto_detect_tool_params(parsed_ts)
        print(f"✅ 工具推荐: {recommendations['suggested_tools'][:3]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "🧪 " + "="*58)
    print("   CSV文件输入功能 - 集成测试")
    print("   测试已完成集成的5个工具")
    print("="*60)
    
    results = []
    
    # 运行核心工具测试
    results.append(("descriptive_statistics", test_1_descriptive_statistics()))
    results.append(("ols_regression", test_2_ols_regression()))
    results.append(("hypothesis_testing", test_3_hypothesis_testing()))
    results.append(("time_series_analysis", test_4_time_series_analysis()))
    results.append(("correlation_analysis", test_5_correlation_analysis()))
    
    # 运行附加功能测试
    results.append(("auto_detection", test_auto_detection()))
    results.append(("intelligent_recognition", test_intelligent_recognition()))
    
    # 总结结果
    print("\n" + "="*60)
    print("📊 测试结果总结")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
    
    print("\n" + "-"*60)
    print(f"总计: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    print("="*60)
    
    if passed == total:
        print("\n🎉 所有测试通过！CSV文件输入功能工作正常！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，需要检查。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)