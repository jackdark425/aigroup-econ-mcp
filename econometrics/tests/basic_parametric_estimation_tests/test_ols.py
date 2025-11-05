"""
OLS模型测试脚本
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from econometrics.basic_parametric_estimation.ols.ols_model import ols_regression, OLSResult


def test_ols_basic():
    """测试基本OLS功能"""
    print("测试基本OLS功能...")
    
    # 生成测试数据
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    # 真实模型: y = 2 + 3*x1 + 2*x2 + noise
    y = 2 + 3*x1 + 2*x2 + np.random.randn(n) * 0.5
    
    # 准备数据
    x_data = [[x1[i], x2[i]] for i in range(n)]
    y_data = y.tolist()
    
    # 执行OLS回归
    result = ols_regression(y_data, x_data, feature_names=['x1', 'x2'])
    
    # 验证结果类型
    assert isinstance(result, OLSResult), "结果应为OLSResult类型"
    
    # 验证系数数量
    assert len(result.coefficients) == 3, "应该有3个系数（包括常数项）"
    
    # 验证统计量合理性
    assert 0 <= result.r_squared <= 1, "R方应在0到1之间"
    assert result.n_obs == n, f"观测数量应为{n}"
    
    print("  系数:", result.coefficients)
    print("  R方:", result.r_squared)
    print("  调整R方:", result.adj_r_squared)
    print("  F统计量:", result.f_statistic)
    print("  基本OLS功能测试通过")


def test_ols_no_constant():
    """测试不包含常数项的OLS"""
    print("测试不包含常数项的OLS...")
    
    # 生成测试数据
    np.random.seed(42)
    n = 50
    x = np.random.randn(n)
    # 真实模型: y = 2*x + noise
    y = 2*x + np.random.randn(n) * 0.3
    
    # 执行OLS回归（不包含常数项）
    result = ols_regression(y.tolist(), x.tolist(), constant=False, feature_names=['x1'])
    
    # 验证系数数量
    assert len(result.coefficients) == 1, "应该有1个系数（不包括常数项）"
    
    print("  系数:", result.coefficients)
    print("  不包含常数项的OLS测试通过")


def test_ols_errors():
    """测试OLS错误处理"""
    print("测试OLS错误处理...")
    
    # 测试空数据
    try:
        ols_regression([], [])
        assert False, "应该抛出ValueError异常"
    except ValueError:
        print("  空数据错误处理正确")
    
    # 测试不一致的数据维度
    try:
        y_data = [1, 2, 3]
        x_data = [[1, 2], [3, 4]]  # 维度不匹配
        ols_regression(y_data, x_data)
        assert False, "应该抛出ValueError异常"
    except ValueError:
        print("  数据维度不一致错误处理正确")
    
    print("  OLS错误处理测试通过")


if __name__ == "__main__":
    print("开始测试OLS模型...")
    test_ols_basic()
    test_ols_no_constant()
    test_ols_errors()
    print("所有OLS测试通过!")