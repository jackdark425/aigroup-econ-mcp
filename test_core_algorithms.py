"""
测试脚本：直接测试模型设定、诊断与稳健推断模块的核心算法
"""

import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_robust_errors():
    """测试稳健标准误模块"""
    print("1. 测试稳健标准误模块...")
    try:
        from econometrics.model_specification_diagnostics_robust_inference.robust_errors import robust_errors_model
        print("   模块导入成功")
        
        # 生成测试数据
        np.random.seed(42)
        n = 100
        x = np.random.randn(n, 2)
        # 异方差误差
        errors = np.random.randn(n) * (1 + 0.5 * np.abs(x[:, 0]))
        y = 1 + 2 * x[:, 0] + 3 * x[:, 1] + errors
        
        # 测试函数
        result = robust_errors_model.robust_errors_regression(
            y_data=y.tolist(),
            x_data=x.tolist(),
            feature_names=['x1', 'x2'],
            cov_type='HC1'
        )
        print("   算法执行成功")
        print(f"   系数: {result.coefficients}")
        print(f"   稳健标准误: {result.robust_std_errors}")
    except Exception as e:
        print(f"   测试失败: {e}")


def test_gls():
    """测试广义最小二乘法模块"""
    print("\n2. 测试广义最小二乘法模块...")
    try:
        from econometrics.model_specification_diagnostics_robust_inference.generalized_least_squares import gls_model
        print("   模块导入成功")
        
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
        
        # 测试函数
        result = gls_model.gls_regression(
            y_data=y.tolist(),
            x_data=x.tolist(),
            sigma=sigma.tolist(),
            feature_names=['x1', 'x2']
        )
        print("   算法执行成功")
        print(f"   系数: {result.coefficients}")
        print(f"   标准误: {result.std_errors}")
    except Exception as e:
        print(f"   测试失败: {e}")


def test_wls():
    """测试加权最小二乘法模块"""
    print("\n3. 测试加权最小二乘法模块...")
    try:
        from econometrics.model_specification_diagnostics_robust_inference.weighted_least_squares import wls_model
        print("   模块导入成功")
        
        # 生成测试数据
        np.random.seed(42)
        n = 100
        x = np.random.randn(n, 2)
        # 异方差结构
        weights = 1 + np.abs(x[:, 0])
        errors = np.random.randn(n) * np.sqrt(1/weights)
        y = 1 + 2 * x[:, 0] + 3 * x[:, 1] + errors
        
        # 测试函数
        result = wls_model.wls_regression(
            y_data=y.tolist(),
            x_data=x.tolist(),
            weights=weights.tolist(),
            feature_names=['x1', 'x2']
        )
        print("   算法执行成功")
        print(f"   系数: {result.coefficients}")
        print(f"   标准误: {result.std_errors}")
    except Exception as e:
        print(f"   测试失败: {e}")


def test_regularization():
    """测试正则化方法模块"""
    print("\n4. 测试正则化方法模块...")
    try:
        from econometrics.model_specification_diagnostics_robust_inference.regularization import regularization_model
        print("   模块导入成功")
        
        # 生成测试数据
        np.random.seed(42)
        n = 100
        x = np.random.randn(n, 5)
        y = 1 + 2 * x[:, 0] + 3 * x[:, 1] + 0.1 * x[:, 2] + np.random.randn(n) * 0.5
        
        # 测试不同方法
        methods = ['ridge', 'lasso', 'elastic_net']
        for method in methods:
            try:
                result = regularization_model.regularized_regression(
                    y_data=y.tolist(),
                    x_data=x.tolist(),
                    method=method,
                    alpha=0.1,
                    feature_names=[f'x{i+1}' for i in range(5)]
                )
                print(f"   {method}: 算法执行成功")
                print(f"   系数: {result.coefficients[:2]}...")  # 只显示前两个系数
            except Exception as e:
                print(f"   {method}: 测试失败 - {e}")
    except Exception as e:
        print(f"   模块导入失败: {e}")


def test_simultaneous_equations():
    """测试联立方程模型模块"""
    print("\n5. 测试联立方程模型模块...")
    try:
        from econometrics.model_specification_diagnostics_robust_inference.simultaneous_equations import simultaneous_equations_model
        print("   模块导入成功")
        
        # 生成测试数据
        np.random.seed(42)
        n = 50
        
        # 简化版本：生成满足工具变量假设的数据
        z = np.random.randn(n, 2)  # 工具变量
        x1 = np.random.randn(n)    # 外生变量
        
        # 生成内生变量
        p = 1 + 0.5 * z[:, 0] + 0.3 * x1 + np.random.randn(n) * 0.2  # 价格（内生）
        q = 2 + 1.2 * p + 0.8 * x1 + np.random.randn(n) * 0.3  # 需求量（因变量）
        
        # 修复数据结构问题
        demand_x = np.column_stack([p, x1])  # 需求方程的自变量
        supply_x = np.column_stack([q, z[:, 1]])  # 供给方程的自变量
        
        # 测试函数
        result = simultaneous_equations_model.two_stage_least_squares(
            y_data=[q.tolist(), p.tolist()],  # 两个方程的因变量
            x_data=[demand_x.tolist(), supply_x.tolist()],  # 两个方程的自变量
            instruments=np.column_stack([z, x1]).tolist(),  # 工具变量
            equation_names=['demand', 'supply'],
            endogenous_vars=['quantity', 'price'],
            exogenous_vars=['instrument1', 'instrument2', 'exog_var']
        )
        print("   算法执行成功")
        print(f"   需求方程系数: {result.coefficients[0]}")
        print(f"   供给方程系数: {result.coefficients[1]}")
    except Exception as e:
        print(f"   测试失败: {e}")


def test_diagnostic_tests():
    """测试模型诊断测试模块"""
    print("\n6. 测试模型诊断测试模块...")
    try:
        from econometrics.model_specification_diagnostics_robust_inference.diagnostic_tests import diagnostic_tests_model
        print("   模块导入成功")
        
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
        
        # 测试函数
        result = diagnostic_tests_model.diagnostic_tests(
            y_data=y.tolist(),
            x_data=x.tolist(),
            feature_names=['x1', 'x2', 'x3']
        )
        print("   算法执行成功")
        print(f"   BP检验统计量: {result.het_breuschpagan_stat}")
        print(f"   BP检验p值: {result.het_breuschpagan_pvalue}")
    except Exception as e:
        print(f"   测试失败: {e}")


def test_model_selection():
    """测试模型选择模块"""
    print("\n7. 测试模型选择模块...")
    try:
        from econometrics.model_specification_diagnostics_robust_inference.model_selection import model_selection_model
        print("   模块导入成功")
        
        # 生成测试数据
        np.random.seed(42)
        n = 100
        x = np.random.randn(n, 3)
        y = 1 + 2 * x[:, 0] + 3 * x[:, 1] + np.random.randn(n) * 0.5
        
        # 测试函数
        result = model_selection_model.model_selection_criteria(
            y_data=y.tolist(),
            x_data=x.tolist(),
            feature_names=['x1', 'x2', 'x3'],
            cv_folds=5  # 5折交叉验证
        )
        print("   算法执行成功")
        print(f"   AIC: {result.aic}")
        print(f"   BIC: {result.bic}")
        print(f"   交叉验证得分: {result.cv_score}")
    except Exception as e:
        print(f"   测试失败: {e}")


def main():
    """主测试函数"""
    print("开始测试模型设定、诊断与稳健推断模块的核心算法...")
    print("=" * 60)
    
    test_robust_errors()
    test_gls()
    test_wls()
    test_regularization()
    test_simultaneous_equations()
    test_diagnostic_tests()
    test_model_selection()
    
    print("\n" + "=" * 60)
    print("核心算法测试完成")


if __name__ == "__main__":
    main()