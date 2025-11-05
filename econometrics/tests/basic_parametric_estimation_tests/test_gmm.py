"""
GMM模型测试脚本
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from econometrics.basic_parametric_estimation.gmm.gmm_model import gmm_estimation, GMMResult


def test_gmm_basic():
    """测试基本GMM功能"""
    print("测试基本GMM功能...")
    
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
    
    # 执行GMM回归（无工具变量，退化为OLS）
    result = gmm_estimation(y_data, x_data, feature_names=['x1', 'x2'])
    
    # 验证结果类型
    assert isinstance(result, GMMResult), "结果应为GMMResult类型"
    
    # 验证系数数量
    assert len(result.coefficients) == 3, "应该有3个系数（包括常数项）"
    
    # 验证统计量合理性
    assert result.n_obs == n, f"观测数量应为{n}"
    assert result.n_moments == 3, "矩条件数量应为3"
    
    print("  系数:", result.coefficients)
    print("  J统计量:", result.j_statistic)
    print("  基本GMM功能测试通过")


def test_gmm_with_instruments():
    """测试带工具变量的GMM"""
    print("测试带工具变量的GMM...")
    
    # 生成测试数据（工具变量）
    np.random.seed(42)
    n = 80
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    z1 = np.random.randn(n)  # 工具变量
    z2 = np.random.randn(n)  # 工具变量
    # 真实模型: y = 2 + 3*x1 + 2*x2 + noise
    y = 2 + 3*x1 + 2*x2 + np.random.randn(n) * 0.5
    
    # 准备数据
    x_data = [[x1[i], x2[i]] for i in range(n)]
    instruments = [[z1[i], z2[i]] for i in range(n)]
    y_data = y.tolist()
    
    # 执行GMM回归（带工具变量）
    result = gmm_estimation(y_data, x_data, instruments=instruments, feature_names=['x1', 'x2'])
    
    # 验证系数数量
    assert len(result.coefficients) == 3, "应该有3个系数（包括常数项）"
    
    print("  系数:", result.coefficients)
    print("  带工具变量的GMM测试通过")


def test_gmm_no_constant():
    """测试不包含常数项的GMM"""
    print("测试不包含常数项的GMM...")
    
    # 生成测试数据
    np.random.seed(42)
    n = 50
    x = np.random.randn(n)
    # 真实模型: y = 2*x + noise
    y = 2*x + np.random.randn(n) * 0.3
    
    # 执行GMM回归（不包含常数项）
    result = gmm_estimation(y.tolist(), x.tolist(), constant=False, feature_names=['x1'])
    
    # 验证系数数量
    assert len(result.coefficients) == 1, "应该有1个系数（不包括常数项）"
    
    print("  系数:", result.coefficients)
    print("  不包含常数项的GMM测试通过")


def test_gmm_errors():
    """测试GMM错误处理"""
    print("测试GMM错误处理...")
    
    # 测试空数据
    try:
        gmm_estimation([], [])
        assert False, "应该抛出ValueError异常"
    except ValueError:
        print("  空数据错误处理正确")
    
    # 测试不一致的数据维度
    try:
        y_data = [1, 2, 3]
        x_data = [[1, 2], [3, 4]]  # 维度不匹配
        gmm_estimation(y_data, x_data)
        assert False, "应该抛出ValueError异常"
    except ValueError:
        print("  数据维度不一致错误处理正确")
    
    print("  GMM错误处理测试通过")


if __name__ == "__main__":
    print("开始测试GMM模型...")
    test_gmm_basic()
    test_gmm_with_instruments()
    test_gmm_no_constant()
    test_gmm_errors()
    print("所有GMM测试通过!")