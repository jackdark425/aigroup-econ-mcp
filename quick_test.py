"""
快速测试修复后的工具
"""

import numpy as np
from tools.time_series_panel_data_adapter import var_svar_adapter
import json

def quick_test():
    """快速测试VAR模型"""
    print("快速测试VAR模型...")
    
    # 生成简单的测试数据
    np.random.seed(42)
    n_obs = 50
    n_vars = 2
    
    # 生成多元时间序列数据
    data = []
    for i in range(n_obs):
        obs = [np.random.normal(0, 1) for _ in range(n_vars)]
        data.append(obs)
    
    variables = ["Var1", "Var2"]
    
    try:
        # 测试VAR模型
        result = var_svar_adapter(
            data=data,
            model_type="var",
            lags=1,
            variables=variables,
            output_format="json"
        )
        
        result_dict = json.loads(result)
        print("✅ VAR模型测试成功!")
        print(f"模型类型: {result_dict.get('model_type', 'N/A')}")
        print(f"变量: {result_dict.get('variables', [])}")
        print(f"系数矩阵维度: {len(result_dict.get('coefficients', []))}x{len(result_dict.get('coefficients', [[]])[0]) if result_dict.get('coefficients') else 0}")
        return True
        
    except Exception as e:
        print(f"❌ VAR模型测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    quick_test()