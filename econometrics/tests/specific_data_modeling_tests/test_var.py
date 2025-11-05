"""
VAR模型测试脚本
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from econometrics.specific_data_modeling.time_series_panel_data.var_svar_model import var_model, svar_model, VARResult


def test_var_basic():
    """测试基本VAR功能"""
    print("测试基本VAR功能...")
    
    # 生成VAR(1)过程数据
    np.random.seed(42)
    n = 100
    
    # VAR(1)系数矩阵
    A = np.array([[0.5, 0.2], 
                  [0.1, 0.3]])
    
    # 生成VAR过程
    data = [np.array([0.0, 0.0])]  # 初始值
    
    for i in range(1, n):
        # VAR(1): y[t] = A @ y[t-1] + noise
        noise = np.random.multivariate_normal([0, 0], [[0.5, 0.1], [0.1, 0.5]])
        new_value = A @ data[i-1] + noise
        data.append(new_value)
    
    # 转换为列表格式
    data = np.array(data).T  # 转置以匹配要求的格式
    data_list = [data[0].tolist(), data[1].tolist()]
    
    # 执行VAR模型
    result = var_model(data_list, lags=1, variables=['y1', 'y2'])
    
    # 验证结果类型
    assert isinstance(result, VARResult), "结果应为VARResult类型"
    
    # 验证模型类型
    assert "VAR(1)" in result.model_type, "模型类型应为VAR(1)"
    
    # 验证统计量合理性
    assert result.n_obs == n, f"观测数量应为{n}"
    assert len(result.variables) == 2, "变量数量应为2"
    
    print("  模型类型:", result.model_type)
    print("  变量:", result.variables)
    print("  AIC:", result.aic)
    print("  基本VAR功能测试通过")


def test_var_lag2():
    """测试2阶滞后的VAR模型"""
    print("测试2阶滞后的VAR模型...")
    
    # 生成VAR(2)过程数据
    np.random.seed(42)
    n = 120
    
    # 生成VAR过程
    data = [np.array([0.0, 0.0])]  # 初始值
    data.append(np.array([0.1, 0.05]))  # 第二个初始值
    
    # VAR(2)系数矩阵
    A1 = np.array([[0.5, 0.2], 
                   [0.1, 0.3]])
    A2 = np.array([[0.1, -0.1], 
                   [0.05, 0.2]])
    
    for i in range(2, n):
        # VAR(2): y[t] = A1 @ y[t-1] + A2 @ y[t-2] + noise
        noise = np.random.multivariate_normal([0, 0], [[0.5, 0.1], [0.1, 0.5]])
        new_value = A1 @ data[i-1] + A2 @ data[i-2] + noise
        data.append(new_value)
    
    # 转换为列表格式
    data = np.array(data).T  # 转置以匹配要求的格式
    data_list = [data[0].tolist(), data[1].tolist()]
    
    # 执行VAR模型 (滞后2阶)
    result = var_model(data_list, lags=2, variables=['y1', 'y2'])
    
    # 验证模型类型
    assert "VAR(2)" in result.model_type, "模型类型应为VAR(2)"
    
    print("  模型类型:", result.model_type)
    print("  2阶滞后的VAR模型测试通过")


def test_var_errors():
    """测试VAR错误处理"""
    print("测试VAR错误处理...")
    
    # 测试空数据
    try:
        var_model([])
        assert False, "应该抛出异常"
    except Exception:
        print("  空数据错误处理正确")
    
    # 测试不一致的序列长度
    try:
        data_list = [[1, 2, 3], [1, 2]]  # 不一致的长度
        var_model(data_list)
        assert False, "应该抛出异常"
    except Exception:
        print("  不一致序列长度错误处理正确")
    
    print("  VAR错误处理测试通过")


if __name__ == "__main__":
    print("开始测试VAR模型...")
    test_var_basic()
    test_var_lag2()
    test_var_errors()
    print("所有VAR测试通过!")