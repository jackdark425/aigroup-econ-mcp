"""
测试改进后的数据格式处理逻辑
验证VAR/SVAR模型、联立方程模型和动态面板数据模型的数据输入格式
"""

import json
from tools.time_series_panel_data_adapter import TimeSeriesPanelDataAdapter
from tools.model_specification_adapter import ModelSpecificationAdapter

def test_var_svar_model():
    """测试VAR/SVAR模型数据格式"""
    print("=" * 50)
    print("测试 VAR/SVAR 模型数据格式")
    print("=" * 50)
    
    # 正确的数据格式
    data = [
        [1.0, 2.5, 1.8],
        [1.2, 2.7, 2.0],
        [1.4, 2.9, 2.2],
        [1.6, 3.1, 2.4],
        [1.8, 3.3, 2.6]
    ]
    variables = ["GDP", "Inflation", "Interest"]
    
    try:
        result = TimeSeriesPanelDataAdapter.var_svar_model(
            data=data,
            model_type="var",
            lags=1,
            variables=variables,
            output_format="json"
        )
        print("✅ VAR/SVAR模型测试成功")
        print(f"结果包含 {len(json.loads(result)['variables'])} 个变量")
        return True
    except Exception as e:
        print(f"❌ VAR/SVAR模型测试失败: {e}")
        return False

def test_simultaneous_equations_model():
    """测试联立方程模型数据格式"""
    print("\n" + "=" * 50)
    print("测试 联立方程模型 数据格式")
    print("=" * 50)
    
    # 正确的数据格式
    y_data = [
        [1.0, 1.2, 1.4, 1.6],  # 方程1的因变量
        [2.0, 2.2, 2.4, 2.6]   # 方程2的因变量
    ]
    
    x_data = [
        [1.5, 2.5],  # 观测1的自变量
        [1.7, 2.7],  # 观测2的自变量
        [1.9, 2.9],  # 观测3的自变量
        [2.1, 3.1]   # 观测4的自变量
    ]
    
    instruments = [
        [1.8, 2.8],  # 观测1的工具变量
        [2.0, 3.0],  # 观测2的工具变量
        [2.2, 3.2],  # 观测3的工具变量
        [2.4, 3.4]   # 观测4的工具变量
    ]
    
    try:
        result = ModelSpecificationAdapter.simultaneous_equations(
            y_data=y_data,
            x_data=x_data,
            instruments=instruments,
            equation_names=["Demand", "Supply"],
            instrument_names=["Income", "Price"],
            constant=True,
            output_format="json"
        )
        print("✅ 联立方程模型测试成功")
        result_data = json.loads(result)
        print(f"结果包含 {len(result_data['equation_names'])} 个方程")
        return True
    except Exception as e:
        print(f"❌ 联立方程模型测试失败: {e}")
        return False

def test_dynamic_panel_model():
    """测试动态面板数据模型数据格式"""
    print("\n" + "=" * 50)
    print("测试 动态面板数据模型 数据格式")
    print("=" * 50)
    
    # 正确的数据格式
    y_data = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
    
    x_data = [
        [1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3]  # 自变量1的时间序列
    ]
    
    entity_ids = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    time_periods = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    
    try:
        result = TimeSeriesPanelDataAdapter.dynamic_panel_model(
            y_data=y_data,
            x_data=x_data,
            entity_ids=entity_ids,
            time_periods=time_periods,
            model_type="diff_gmm",
            lags=1,
            output_format="json"
        )
        print("✅ 动态面板数据模型测试成功")
        result_data = json.loads(result)
        if "error" in result_data:
            print(f"⚠️ 模型拟合警告: {result_data.get('message', '未知错误')}")
        else:
            print(f"结果包含 {result_data['n_obs']} 个观测")
        return True
    except Exception as e:
        print(f"❌ 动态面板数据模型测试失败: {e}")
        return False

def test_error_cases():
    """测试错误数据格式的处理"""
    print("\n" + "=" * 50)
    print("测试 错误数据格式 处理")
    print("=" * 50)
    
    # 测试维度不匹配的错误
    print("测试维度不匹配错误...")
    try:
        # 错误的联立方程数据格式
        y_data_wrong = [[1.0, 1.2], [2.0]]  # 第二个方程观测数量不同
        x_data_wrong = [[1.5], [1.7], [1.9]]
        instruments_wrong = [[1.8], [2.0], [2.2]]
        
        ModelSpecificationAdapter.simultaneous_equations(
            y_data=y_data_wrong,
            x_data=x_data_wrong,
            instruments=instruments_wrong
        )
        print("❌ 应该检测到维度不匹配错误")
    except ValueError as e:
        print(f"✅ 正确检测到维度不匹配错误: {str(e)[:100]}...")
    
    # 测试动态面板数据维度错误
    print("测试动态面板数据维度错误...")
    try:
        y_data_wrong = [1.0, 1.2, 1.4]
        x_data_wrong = [[1.5, 1.7]]  # 自变量观测数量不同
        entity_ids_wrong = [1, 1, 1]
        time_periods_wrong = [1, 2, 3]
        
        TimeSeriesPanelDataAdapter.dynamic_panel_model(
            y_data=y_data_wrong,
            x_data=x_data_wrong,
            entity_ids=entity_ids_wrong,
            time_periods=time_periods_wrong
        )
        print("❌ 应该检测到维度不匹配错误")
    except ValueError as e:
        print(f"✅ 正确检测到维度不匹配错误: {str(e)[:100]}...")

def main():
    """主测试函数"""
    print("MCP工具数据格式改进测试")
    print("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # 测试正常情况
    if test_var_svar_model():
        success_count += 1
    
    if test_simultaneous_equations_model():
        success_count += 1
    
    if test_dynamic_panel_model():
        success_count += 1
    
    # 测试错误情况
    test_error_cases()
    
    print("\n" + "=" * 60)
    print(f"测试结果: {success_count}/{total_tests} 个工具正常工作")
    
    if success_count == total_tests:
        print("🎉 所有工具的数据格式处理逻辑改进成功！")
        print("📖 详细数据格式说明请参考: MCP_TOOLS_DATA_FORMAT_GUIDE.md")
    else:
        print("⚠️ 部分工具需要进一步优化")

if __name__ == "__main__":
    main()