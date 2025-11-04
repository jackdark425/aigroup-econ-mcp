"""
测试脚本：模型设定、诊断与稳健推断模块全面功能测试
"""

import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 修复导入问题
try:
    from econometrics.model_specification_diagnostics_robust_inference import (
        robust_errors_regression,
        gls_regression,
        wls_regression,
        regularized_regression,
        two_stage_least_squares,
        diagnostic_tests,
        model_selection_criteria
    )
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保在项目根目录下运行此脚本")
    sys.exit(1)


def test_robust_errors():
    """测试稳健标准误模块"""
    print("测试稳健标准误模块...")
    
    # 生成测试数据
    np.random.seed(42)
    n = 100
    x = np.random.randn(n, 2)
    # 异方差误差
    errors = np.random.randn(n) * (1 + 0.5 * np.abs(x[:, 0]))
    y = 1 + 2 * x[:, 0] + 3 * x[:, 1] + errors
    
    # 测试不同协方差类型
    for cov_type in ['HC0', 'HC1', 'HC2', 'HC3']:
        try:
            result = robust_errors_regression(
                y_data=y.tolist(),
                x_data=x.tolist(),
                cov_type=cov_type,
                feature_names=['x1', 'x2']
            )
            print(f"  {cov_type}: 成功")
        except Exception as e:
            print(f"  {cov_type}: 失败 - {e}")


def test_gls():
    """测试广义最小二乘法模块"""
    print("测试广义最小二乘法模块...")
    
    # 生成测试数据
    np.random.seed(42)
    n = 50
    x = np.random.randn(n, 2)
    # 生成协方差矩阵（自相关结构）
    sigma = np.eye(n)
    for i in range(n):
        for j in range(n):
            sigma[i, j] = 0.5 ** abs(i - j)
    
    errors = np.random.multivariate_normal(np.zeros(n), sigma)
    y = 1 + 2 * x[:, 0] + 3 * x[:, 1] + errors
    
    try:
        result = gls_regression(
            y_data=y.tolist(),
            x_data=x.tolist(),
            sigma=sigma.tolist(),
            feature_names=['x1', 'x2']
        )
        print("  GLS: 成功")
    except Exception as e:
        print(f"  GLS: 失败 - {e}")


def test_wls():
    """测试加权最小二乘法模块"""
    print("测试加权最小二乘法模块...")
    
    # 生成测试数据
    np.random.seed(42)
    n = 100
    x = np.random.randn(n, 2)
    # 异方差结构
    weights = 1 + np.abs(x[:, 0])
    errors = np.random.randn(n) * np.sqrt(1/weights)
    y = 1 + 2 * x[:, 0] + 3 * x[:, 1] + errors
    
    try:
        result = wls_regression(
            y_data=y.tolist(),
            x_data=x.tolist(),
            weights=weights.tolist(),
            feature_names=['x1', 'x2']
        )
        print("  WLS: 成功")
    except Exception as e:
        print(f"  WLS: 失败 - {e}")


def test_regularization():
    """测试正则化方法模块"""
    print("测试正则化方法模块...")
    
    # 生成测试数据
    np.random.seed(42)
    n = 100
    x = np.random.randn(n, 5)
    y = 1 + 2 * x[:, 0] + 3 * x[:, 1] + 0.1 * x[:, 2] + np.random.randn(n) * 0.5
    
    # 测试不同方法
    methods = ['ridge', 'lasso', 'elastic_net']
    for method in methods:
        try:
            result = regularized_regression(
                y_data=y.tolist(),
                x_data=x.tolist(),
                method=method,
                alpha=0.1,
                feature_names=[f'x{i+1}' for i in range(5)]
            )
            print(f"  {method}: 成功")
        except Exception as e:
            print(f"  {method}: 失败 - {e}")


def test_simultaneous_equations():
    """测试联立方程模型模块"""
    print("测试联立方程模型模块...")
    
    # 生成测试数据
    np.random.seed(42)
    n = 100
    
    # 简单供需模型示例
    z = np.random.randn(n, 2)  # 工具变量
    x1 = np.random.randn(n)    # 外生变量
    
    # 生成内生变量 (供需均衡)
    # 需求: q = α1*p + β1*x1 + ε1
    # 供给: q = α2*p + β2*z1 + ε2
    # 均衡: qd = qs, pd = ps
    
    # 简化版本：生成满足工具变量假设的数据
    p = 1 + 0.5 * z[:, 0] + 0.3 * x1 + np.random.randn(n) * 0.2  # 价格（内生）
    q = 2 + 1.2 * p + 0.8 * x1 + np.random.randn(n) * 0.3  # 需求量（因变量）
    
    try:
        result = two_stage_least_squares(
            y_data=[q.tolist(), p.tolist()],  # 两个方程的因变量
            x_data=[[np.column_stack([p, x1]).tolist()],  # 需求方程
                   [np.column_stack([q, z[:, 1]]).tolist()]], # 供给方程
            instruments=np.column_stack([z, x1]).tolist(),  # 工具变量
            equation_names=['demand', 'supply'],
            endogenous_vars=['quantity', 'price'],
            exogenous_vars=['instrument1', 'instrument2', 'exog_var']
        )
        print("  2SLS: 成功")
    except Exception as e:
        print(f"  2SLS: 失败 - {e}")


def test_diagnostic_tests():
    """测试模型诊断测试模块"""
    print("测试模型诊断测试模块...")
    
    # 生成测试数据
    np.random.seed(42)
    n = 100
    x = np.random.randn(n, 3)
    
    # 生成具有异方差性和自相关性的数据
    errors = np.random.randn(n)
    # 添加自相关
    for i in range(1, n):
        errors[i] += 0.3 * errors[i-1]
    # 添加异方差性
    errors *= (1 + 0.5 * np.abs(x[:, 0]))
    
    y = 1 + 2 * x[:, 0] + 3 * x[:, 1] + errors
    
    try:
        result = diagnostic_tests(
            y_data=y.tolist(),
            x_data=x.tolist(),
            feature_names=['x1', 'x2', 'x3']
        )
        print("  诊断测试: 成功")
    except Exception as e:
        print(f"  诊断测试: 失败 - {e}")


def test_model_selection():
    """测试模型选择模块"""
    print("测试模型选择模块...")
    
    # 生成测试数据
    np.random.seed(42)
    n = 100
    x = np.random.randn(n, 3)
    y = 1 + 2 * x[:, 0] + 3 * x[:, 1] + np.random.randn(n) * 0.5
    
    try:
        result = model_selection_criteria(
            y_data=y.tolist(),
            x_data=x.tolist(),
            feature_names=['x1', 'x2', 'x3'],
            cv_folds=5  # 5折交叉验证
        )
        print("  模型选择: 成功")
    except Exception as e:
        print(f"  模型选择: 失败 - {e}")


def main():
    """主测试函数"""
    print("开始测试模型设定、诊断与稳健推断模块...")
    print("=" * 50)
    
    test_robust_errors()
    test_gls()
    test_wls()
    test_regularization()
    test_simultaneous_equations()
    test_diagnostic_tests()
    test_model_selection()
    
    print("=" * 50)
    print("测试完成")


if __name__ == "__main__":
    main()