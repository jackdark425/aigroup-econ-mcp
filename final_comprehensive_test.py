"""
最终综合测试 - 验证所有三个修复的工具
"""

import numpy as np
import json
from tools.time_series_panel_data_adapter import var_svar_adapter
from tools.model_specification_adapter import simultaneous_equations_adapter
from tools.time_series_panel_data_adapter import dynamic_panel_adapter

def test_var_model():
    """测试VAR模型"""
    print("=" * 50)
    print("测试VAR模型...")
    
    np.random.seed(42)
    n_obs = 50
    n_vars = 2
    
    # 生成多元时间序列数据
    data = []
    for i in range(n_obs):
        obs = [np.random.normal(0, 1) for _ in range(n_vars)]
        data.append(obs)
    
    variables = ["GDP", "Inflation"]
    
    try:
        result = var_svar_adapter(
            data=data,
            model_type="var",
            lags=1,
            variables=variables,
            output_format="json"
        )
        
        result_dict = json.loads(result)
        print("✅ VAR模型测试成功!")
        print(f"  模型类型: {result_dict.get('model_type', 'N/A')}")
        print(f"  变量: {result_dict.get('variables', [])}")
        print(f"  系数矩阵维度: {len(result_dict.get('coefficients', []))}x{len(result_dict.get('coefficients', [[]])[0]) if result_dict.get('coefficients') else 0}")
        return True
        
    except Exception as e:
        print(f"❌ VAR模型测试失败: {str(e)}")
        return False

def test_simultaneous_equations():
    """测试联立方程模型"""
    print("=" * 50)
    print("测试联立方程模型...")
    
    # 生成测试数据
    np.random.seed(123)
    n_obs = 100
    
    # 内生变量数据 - 每个方程一个列表
    y_data = []
    # 方程1的因变量
    y1_data = np.random.normal(0, 1, n_obs).tolist()
    # 方程2的因变量
    y2_data = np.random.normal(0, 1, n_obs).tolist()
    y_data = [y1_data, y2_data]
    
    # 外生变量数据 - 每个观测一个列表
    x_data = []
    for i in range(n_obs):
        x1 = np.random.normal(0, 1)
        x2 = np.random.normal(0, 1)
        x_data.append([x1, x2])
    
    # 工具变量数据 - 每个观测一个列表
    instruments = []
    for i in range(n_obs):
        z1 = np.random.normal(0, 1)
        z2 = np.random.normal(0, 1)
        instruments.append([z1, z2])
    
    try:
        result = simultaneous_equations_adapter(
            y_data=y_data,
            x_data=x_data,
            instruments=instruments,
            equation_names=["Eq1", "Eq2"],
            instrument_names=["Z1", "Z2"],
            constant=True,
            output_format="json"
        )
        
        result_dict = json.loads(result)
        print("✅ 联立方程模型测试成功!")
        print(f"  方程数量: {len(result_dict.get('equations', []))}")
        return True
        
    except Exception as e:
        print(f"❌ 联立方程模型测试失败: {str(e)}")
        return False

def test_dynamic_panel():
    """测试动态面板数据模型"""
    print("=" * 50)
    print("测试动态面板数据模型...")
    
    # 生成面板数据
    np.random.seed(456)
    n_entities = 5
    n_periods = 10
    n_obs = n_entities * n_periods
    
    # 生成因变量数据
    y_data = np.random.normal(0, 1, n_obs).tolist()
    
    # 生成自变量数据
    x_data = []
    for i in range(n_obs):
        x1 = np.random.normal(0, 1)
        x2 = np.random.normal(0, 1)
        x_data.append([x1, x2])
    
    # 生成实体ID和时间周期
    entity_ids = []
    time_periods = []
    for i in range(n_entities):
        for t in range(n_periods):
            entity_ids.append(i)
            time_periods.append(t)
    
    try:
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
        print("✅ 动态面板数据模型测试成功!")
        print(f"  模型类型: {result_dict.get('model_type', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"❌ 动态面板数据模型测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("开始最终综合测试...")
    print("验证三个修复的计量经济学工具")
    print()
    
    results = []
    
    # 测试VAR模型
    results.append(test_var_model())
    
    # 测试联立方程模型
    results.append(test_simultaneous_equations())
    
    # 测试动态面板数据模型
    results.append(test_dynamic_panel())
    
    print("=" * 50)
    print("测试结果总结:")
    print(f"VAR模型: {'✅ 通过' if results[0] else '❌ 失败'}")
    print(f"联立方程模型: {'✅ 通过' if results[1] else '❌ 失败'}")
    print(f"动态面板数据模型: {'✅ 通过' if results[2] else '❌ 失败'}")
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n总体结果: {success_count}/{total_count} 个工具测试通过")
    
    if success_count == total_count:
        print("🎉 所有工具修复成功!")
    else:
        print("⚠️ 部分工具仍需调试")

if __name__ == "__main__":
    main()