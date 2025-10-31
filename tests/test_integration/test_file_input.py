"""
集成测试 - 文件输入功能
测试CSV/JSON文件输入功能的集成和解析
"""

import pytest
import json
import sys
import os
from typing import Dict, Any

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from aigroup_econ_mcp.tools.file_parser import FileParser
from aigroup_econ_mcp.tools.file_input_handler import process_file_for_tool


class TestCSVFileInput:
    """测试CSV文件输入"""
    
    def test_csv_descriptive_statistics(self):
        """测试CSV文件用于描述性统计"""
        print("\n--- CSV描述性统计测试 ---")
        
        csv_content = """GDP增长率,通货膨胀率,失业率
3.2,2.1,4.5
2.8,2.3,4.2
3.5,1.9,4.0
2.9,2.4,4.3
3.1,2.2,4.1"""
        
        try:
            # 解析CSV
            parsed = FileParser.parse_file_content(csv_content, "csv")
            
            print(f"✅ CSV解析成功")
            print(f"   - 格式: {parsed['format']}")
            print(f"   - 变量数: {parsed['n_variables']}")
            print(f"   - 观测数: {parsed['n_observations']}")
            print(f"   - 变量名: {parsed['variables']}")
            
            # 转换为多变量字典格式
            converted = FileParser.convert_to_tool_format(parsed, 'multi_var_dict')
            data = converted['data']
            
            print(f"✅ 数据转换成功")
            print(f"   - GDP增长率: {len(data['GDP增长率'])}个观测")
            print(f"   - 通货膨胀率: {len(data['通货膨胀率'])}个观测")
            print(f"   - 失业率: {len(data['失业率'])}个观测")
            
            return True
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            return False
    
    def test_csv_regression_analysis(self):
        """测试CSV文件用于回归分析"""
        print("\n--- CSV回归分析测试 ---")
        
        csv_content = """广告支出,价格,销售额
800,99,12000
900,95,13500
750,102,11800
1000,98,14200
850,96,13800
950,94,15100"""
        
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
            print(f"   - y_data长度: {len(converted['y_data'])}")
            print(f"   - x_data行数: {len(converted['x_data'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            return False
    
    def test_csv_time_series_analysis(self):
        """测试CSV文件用于时间序列分析"""
        print("\n--- CSV时间序列分析测试 ---")
        
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
105.8"""
        
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
            print(f"   - 数据长度: {len(data)}")
            print(f"   - 前5个值: {data[:5]}")
            
            return True
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            return False
    
    def test_csv_panel_data_analysis(self):
        """测试CSV文件用于面板数据分析"""
        print("\n--- CSV面板数据分析测试 ---")
        
        csv_content = """公司ID,年份,收入,员工数,利润
1,2020,1000,50,100
1,2021,1100,52,110
2,2020,800,40,80
2,2021,900,42,90
3,2020,1200,60,120
3,2021,1300,62,130"""
        
        try:
            # 解析CSV
            parsed = FileParser.parse_file_content(csv_content, "csv")
            
            print(f"✅ CSV解析成功")
            print(f"   - 数据类型: {parsed['data_type']}")
            print(f"   - 变量数: {parsed['n_variables']}")
            print(f"   - 观测数: {parsed['n_observations']}")
            
            # 转换为面板数据格式
            converted = FileParser.convert_to_tool_format(parsed, 'panel')
            
            print(f"✅ 面板数据转换成功")
            print(f"   - 实体ID数量: {len(converted['entity_ids'])}")
            print(f"   - 时间周期数量: {len(converted['time_periods'])}")
            print(f"   - 因变量: {converted['y_variable']}")
            print(f"   - 自变量: {converted['feature_names']}")
            
            return True
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            return False


class TestJSONFileInput:
    """测试JSON文件输入"""
    
    def test_json_descriptive_statistics(self):
        """测试JSON文件用于描述性统计"""
        print("\n--- JSON描述性统计测试 ---")
        
        json_content = {
            "GDP增长率": [3.2, 2.8, 3.5, 2.9, 3.1],
            "通货膨胀率": [2.1, 2.3, 1.9, 2.4, 2.2],
            "失业率": [4.5, 4.2, 4.0, 4.3, 4.1]
        }
        
        try:
            # 解析JSON
            parsed = FileParser.parse_file_content(
                json.dumps(json_content), "json"
            )
            
            print(f"✅ JSON解析成功")
            print(f"   - 格式: {parsed['format']}")
            print(f"   - 变量数: {parsed['n_variables']}")
            print(f"   - 观测数: {parsed['n_observations']}")
            print(f"   - 变量名: {parsed['variables']}")
            
            # 转换为多变量字典格式
            converted = FileParser.convert_to_tool_format(parsed, 'multi_var_dict')
            data = converted['data']
            
            print(f"✅ 数据转换成功")
            for var_name in data.keys():
                print(f"   - {var_name}: {len(data[var_name])}个观测")
            
            return True
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            return False
    
    def test_json_regression_analysis(self):
        """测试JSON文件用于回归分析"""
        print("\n--- JSON回归分析测试 ---")
        
        json_content = {
            "广告支出": [800, 900, 750, 1000, 850],
            "价格": [99, 95, 102, 98, 96],
            "销售额": [12000, 13500, 11800, 14200, 13800]
        }
        
        try:
            # 解析JSON
            parsed = FileParser.parse_file_content(
                json.dumps(json_content), "json"
            )
            
            print(f"✅ JSON解析成功")
            print(f"   - 变量数: {parsed['n_variables']}")
            print(f"   - 观测数: {parsed['n_observations']}")
            
            # 转换为回归格式
            converted = FileParser.convert_to_tool_format(parsed, 'regression')
            
            print(f"✅ 回归数据转换成功")
            print(f"   - 因变量: {converted['y_variable']}")
            print(f"   - 自变量: {converted['feature_names']}")
            print(f"   - y_data长度: {len(converted['y_data'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            return False


class TestAutoDetection:
    """测试自动格式检测"""
    
    def test_auto_detection_csv(self):
        """测试自动检测CSV格式"""
        print("\n--- 自动检测CSV格式 ---")
        
        csv_content = """变量1,变量2
1.0,2.0
3.0,4.0
5.0,6.0"""
        
        try:
            parsed = FileParser.parse_file_content(csv_content, "auto")
            
            print(f"✅ 自动检测成功")
            print(f"   - 检测格式: {parsed['format']}")
            print(f"   - 变量数: {parsed['n_variables']}")
            print(f"   - 观测数: {parsed['n_observations']}")
            
            assert parsed['format'] == 'csv'
            return True
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            return False
    
    def test_auto_detection_json(self):
        """测试自动检测JSON格式"""
        print("\n--- 自动检测JSON格式 ---")
        
        json_content = '{"变量1": [1.0, 3.0, 5.0], "变量2": [2.0, 4.0, 6.0]}'
        
        try:
            parsed = FileParser.parse_file_content(json_content, "auto")
            
            print(f"✅ 自动检测成功")
            print(f"   - 检测格式: {parsed['format']}")
            print(f"   - 变量数: {parsed['n_variables']}")
            print(f"   - 观测数: {parsed['n_observations']}")
            
            assert parsed['format'] == 'json'
            return True
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            return False


class TestIntelligentRecognition:
    """测试智能识别功能"""
    
    def test_time_series_recognition(self):
        """测试时间序列数据识别"""
        print("\n--- 时间序列数据识别 ---")
        
        csv_content = """日期,数值
2024-01-01,100
2024-01-02,102
2024-01-03,101
2024-01-04,103
2024-01-05,105"""
        
        try:
            parsed = FileParser.parse_file_content(csv_content, "csv")
            
            print(f"✅ 时间序列识别成功")
            print(f"   - 数据类型: {parsed['data_type']}")
            print(f"   - 变量: {parsed['variables']}")
            
            # 测试智能推荐
            recommendations = FileParser.auto_detect_tool_params(parsed)
            
            print(f"✅ 工具推荐成功")
            print(f"   - 推荐工具: {recommendations['suggested_tools'][:3]}")
            
            return True
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            return False
    
    def test_panel_data_recognition(self):
        """测试面板数据识别"""
        print("\n--- 面板数据识别 ---")
        
        csv_content = """公司ID,年份,收入,利润
1,2020,1000,100
1,2021,1100,110
2,2020,800,80
2,2021,900,90
3,2020,1200,120
3,2021,1300,130"""
        
        try:
            parsed = FileParser.parse_file_content(csv_content, "csv")
            
            print(f"✅ 面板数据识别成功")
            print(f"   - 数据类型: {parsed['data_type']}")
            print(f"   - 变量: {parsed['variables']}")
            
            # 测试智能推荐
            recommendations = FileParser.auto_detect_tool_params(parsed)
            
            print(f"✅ 工具推荐成功")
            print(f"   - 推荐工具: {recommendations['suggested_tools'][:3]}")
            
            return True
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            return False


class TestErrorHandling:
    """测试错误处理"""
    
    def test_empty_file(self):
        """测试空文件"""
        print("\n--- 空文件处理 ---")
        
        try:
            FileParser.parse_file_content("", "csv")
            print(f"❌ 应该抛出错误")
            return False
        except ValueError as e:
            print(f"✅ 正确处理空文件: {str(e)}")
            return True
    
    def test_invalid_json(self):
        """测试无效JSON"""
        print("\n--- 无效JSON处理 ---")
        
        try:
            FileParser.parse_file_content("{invalid json", "json")
            print(f"❌ 应该抛出错误")
            return False
        except ValueError as e:
            print(f"✅ 正确处理无效JSON: {str(e)}")
            return True
    
    def test_no_numeric_data(self):
        """测试无数值数据"""
        print("\n--- 无数值数据处理 ---")
        
        csv_content = """姓名,职业
张三,工程师
李四,医生
王五,教师"""
        
        try:
            FileParser.parse_file_content(csv_content, "csv")
            print(f"❌ 应该抛出错误")
            return False
        except ValueError as e:
            print(f"✅ 正确处理无数值数据: {str(e)}")
            return True


def main():
    """运行所有文件输入测试"""
    print("\n" + "="*80)
    print("开始运行文件输入功能集成测试")
    print("="*80)
    
    results = []
    
    # CSV文件测试
    csv_tester = TestCSVFileInput()
    results.append(("CSV描述性统计", csv_tester.test_csv_descriptive_statistics()))
    results.append(("CSV回归分析", csv_tester.test_csv_regression_analysis()))
    results.append(("CSV时间序列", csv_tester.test_csv_time_series_analysis()))
    results.append(("CSV面板数据", csv_tester.test_csv_panel_data_analysis()))
    
    # JSON文件测试
    json_tester = TestJSONFileInput()
    results.append(("JSON描述性统计", json_tester.test_json_descriptive_statistics()))
    results.append(("JSON回归分析", json_tester.test_json_regression_analysis()))
    
    # 自动检测测试
    auto_tester = TestAutoDetection()
    results.append(("自动检测CSV", auto_tester.test_auto_detection_csv()))
    results.append(("自动检测JSON", auto_tester.test_auto_detection_json()))
    
    # 智能识别测试
    recognition_tester = TestIntelligentRecognition()
    results.append(("时间序列识别", recognition_tester.test_time_series_recognition()))
    results.append(("面板数据识别", recognition_tester.test_panel_data_recognition()))
    
    # 错误处理测试
    error_tester = TestErrorHandling()
    results.append(("空文件处理", error_tester.test_empty_file()))
    results.append(("无效JSON处理", error_tester.test_invalid_json()))
    results.append(("无数值数据处理", error_tester.test_no_numeric_data()))
    
    # 总结结果
    print("\n" + "="*80)
    print("📊 文件输入功能测试结果总结")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
    
    print("\n" + "-"*80)
    print(f"总计: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    print("="*80)
    
    if passed == total:
        print("\n🎉 所有文件输入功能测试通过！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，需要检查。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)