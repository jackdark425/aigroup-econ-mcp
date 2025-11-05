"""
测试修复后的三个计量经济学工具
1. VAR/SVAR模型
2. 联立方程模型
3. 动态面板数据模型
"""

import numpy as np
import json
from tools.time_series_panel_data_adapter import var_svar_adapter, dynamic_panel_adapter
from tools.model_specification_adapter import simultaneous_equations_adapter

def test_var_svar_model():
    """测试VAR/SVAR模型"""
    print("=" * 50)
    print("测试VAR/SVAR模型")
    print("=" * 50)
    
    # 生成测试数据
    np.random.seed(42)
    n_obs = 100
    n_vars = 3
    
    # 生成多元时间序列数据
    data = []
    for i in range(n_obs):
        obs = [np.random.normal(0, 1) for _ in range(n_vars)]
        data.append(obs)
    
    variables = ["GDP", "Inflation", "Interest_Rate"]
    
    try:
        # 测试VAR模型
        print("测试VAR模型...")
        result = var_svar_adapter(
            data=data,
            model_type="var",
            lags=2,
            variables=variables,
            output_format="json"
        )
        
        result_dict = json.loads(result)
        print(f"VAR模型测试成功!")
        print(f"模型类型: {result_dict.get('model_type', 'N/A')}")
        print(f"变量数量: {len(result_dict.get('variables', []))}")
        print(f"系数矩阵维度: {len(result_dict.get('coefficients', []))}x{len(result_dict.get('coefficients', [[]])[0]) if result_dict.get('coefficients') else 0}")
        print(f"观测数量: {result_dict.get('n_obs', 0)}")
        
        # 测试SVAR模型
        print("\n测试SVAR模型...")
        a_matrix = [[1, 0, 0], [0.5, 1, 0], [0.3, 0.2, 1]]
        b_matrix = [[0.5, 0, 0], [0, 0.8, 0], [0, 0, 0.6]]
        
        result = var_svar_adapter(
            data=data,
            model_type="svar",
            lags=1,
            variables=variables,
            a_matrix=a_matrix,
            b_matrix=b_matrix,
            output_format="json"
        )
        
        result_dict = json.loads(result)
        print(f"SVAR模型测试成功!")
        print(f"模型类型: {result_dict.get('model_type', 'N/A')}")
        print(f"变量数量: {len(result_dict.get('variables', []))}")
        
        return True
        
    except Exception as e:
        print(f"VAR/SVAR模型测试失败: {str(e)}")
        return False

def test_simultaneous_equations():
    """测试联立方程模型"""
    print("\n" + "=" * 50)
    print("测试联立方程模型")
    print("=" * 50)
    
    # 生成测试数据
    np.random.seed(42)
    n_obs = 50
    n_equations = 2
    n_x_vars = 3
    n_instruments = 4
    
    # 生成因变量数据
    y_data = []
    for eq in range(n_equations):
        y_eq = np.random.normal(0, 1, n_obs)
        y_data.append(y_eq.tolist())
    
    # 生成自变量数据
    x_data = []
    for obs in range(n_obs):
        x_obs = np.random.normal(0, 1, n_x_vars)
        x_data.append(x_obs.tolist())
    
    # 生成工具变量数据
    instruments = []
    for obs in range(n_obs):
        inst_obs = np.random.normal(0, 1, n_instruments)
        instruments.append(inst_obs.tolist())
    
    equation_names = ["Demand_Equation", "Supply_Equation"]
    instrument_names = ["IV1", "IV2", "IV3", "IV4"]
    
    try:
        print("测试联立方程模型(2SLS)...")
        result = simultaneous_equations_adapter(
            y_data=y_data,
            x_data=x_data,
            instruments=instruments,
            equation_names=equation_names,
            instrument_names=instrument_names,
            constant=True,
            output_format="json"
        )
        
        result_dict = json.loads(result)
        print(f"联立方程模型测试成功!")
        print(f"方程数量: {len(result_dict.get('equation_names', []))}")
        print(f"观测数量: {result_dict.get('n_obs', 0)}")
        print(f"内生变量: {result_dict.get('endogenous_vars', [])}")
        print(f"外生变量: {result_dict.get('exogenous_vars', [])}")
        
        return True
        
    except Exception as e:
        print(f"联立方程模型测试失败: {str(e)}")
        return False

def test_dynamic_panel_models():
    """测试动态面板数据模型"""
    print("\n" + "=" * 50)
    print("测试动态面板数据模型")
    print("=" * 50)
    
    # 生成测试数据
    np.random.seed(42)
    n_individuals = 10
    n_time_periods = 5
    n_obs = n_individuals * n_time_periods
    
    # 生成面板数据
    y_data = np.random.normal(0, 1, n_obs).tolist()
    
    # 生成自变量数据
    x_data = []
    for i in range(2):  # 2个自变量
        x_var = np.random.normal(0, 1, n_obs)
        x_data.append(x_var.tolist())
    
    # 生成个体和时间标识符
    entity_ids = []
    time_periods = []
    for i in range(n_individuals):
        for t in range(n_time_periods):
            entity_ids.append(i)
            time_periods.append(t)
    
    try:
        # 测试差分GMM
        print("测试差分GMM模型...")
        result = dynamic_panel_adapter(
            y_data=y_data,
            x_data=x_data,
            entity_ids=entity_ids,
            time_periods=time_periods,
            model_type="diff_gmm",
            lags=1,
            output_format="json"
        )
        
        result_dict = json.loads(result)
        print(f"差分GMM模型测试成功!")
        print(f"模型类型: {result_dict.get('model_type', 'N/A')}")
        print(f"系数数量: {len(result_dict.get('coefficients', []))}")
        print(f"观测数量: {result_dict.get('n_obs', 0)}")
        print(f"个体数量: {result_dict.get('n_individuals', 0)}")
        print(f"时间期数: {result_dict.get('n_time_periods', 0)}")
        
        # 测试系统GMM
        print("\n测试系统GMM模型...")
        result = dynamic_panel_adapter(
            y_data=y_data,
            x_data=x_data,
            entity_ids=entity_ids,
            time_periods=time_periods,
            model_type="sys_gmm",
            lags=1,
            output_format="json"
        )
        
        result_dict = json.loads(result)
        print(f"系统GMM模型测试成功!")
        print(f"模型类型: {result_dict.get('model_type', 'N/A')}")
        print(f"系数数量: {len(result_dict.get('coefficients', []))}")
        
        return True
        
    except Exception as e:
        print(f"动态面板模型测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("开始测试修复后的计量经济学工具...")
    
    success_count = 0
    total_tests = 3
    
    # 测试VAR/SVAR模型
    if test_var_svar_model():
        success_count += 1
    
    # 测试联立方程模型
    if test_simultaneous_equations():
        success_count += 1
    
    # 测试动态面板数据模型
    if test_dynamic_panel_models():
        success_count += 1
    
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    print(f"总测试数: {total_tests}")
    print(f"成功数: {success_count}")
    print(f"成功率: {success_count/total_tests*100:.1f}%")
    
    if success_count == total_tests:
        print("🎉 所有工具测试成功！修复工作完成。")
    else:
        print("⚠️ 部分工具测试失败，需要进一步调试。")

if __name__ == "__main__":
    main()