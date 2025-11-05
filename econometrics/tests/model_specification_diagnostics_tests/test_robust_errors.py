"""
稳健标准误模型测试脚本
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# 尝试导入稳健标准误模型
try:
    from econometrics.model_specification_diagnostics_robust_inference.robust_errors.robust_errors_model import robust_errors_regression, RobustErrorsResult
    ROBUST_ERRORS_AVAILABLE = True
except ImportError:
    ROBUST_ERRORS_AVAILABLE = False
    print("警告: 未找到稳健标准误模型，相关测试将被跳过")


def test_robust_errors_basic():
    """测试基本稳健标准误功能"""
    if not ROBUST_ERRORS_AVAILABLE:
        print("跳过稳健标准误测试（模块不可用）")
        return
        
    print("测试基本稳健标准误功能...")
    
    # 生成测试数据
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    # 真实模型: y = 2 + 3*x1 + 2*x2 + noise (异方差)
    noise = np.random.randn(n) * (0.5 + 0.3 * np.abs(x1))  # 异方差噪声
    y = 2 + 3*x1 + 2*x2 + noise
    
    # 准备数据
    x_data = [[x1[i], x2[i]] for i in range(n)]
    y_data = y.tolist()
    
    # 执行稳健标准误回归
    result = robust_errors_regression(y_data, x_data, feature_names=['x1', 'x2'])
    
    # 验证结果类型
    assert isinstance(result, RobustErrorsResult), "结果应为RobustErrorsResult类型"
    
    # 验证系数数量
    assert len(result.coefficients) == 3, "应该有3个系数（包括常数项）"
    
    # 验证统计量合理性
    assert result.n_obs == n, f"观测数量应为{n}"
    
    print("  系数:", result.coefficients)
    print("  稳健标准误:", result.robust_std_errors)
    print("  基本稳健标准误功能测试通过")


def test_robust_errors_no_constant():
    """测试不包含常数项的稳健标准误"""
    if not ROBUST_ERRORS_AVAILABLE:
        print("跳过稳健标准误测试（模块不可用）")
        return
        
    print("测试不包含常数项的稳健标准误...")
    
    # 生成测试数据
    np.random.seed(42)
    n = 50
    x = np.random.randn(n)
    # 真实模型: y = 2*x + noise (异方差)
    noise = np.random.randn(n) * (0.3 + 0.2 * np.abs(x))  # 异方差噪声
    y = 2*x + noise
    
    # 执行稳健标准误回归（不包含常数项）
    result = robust_errors_regression(y.tolist(), x.tolist(), constant=False, feature_names=['x1'])
    
    # 验证系数数量
    assert len(result.coefficients) == 1, "应该有1个系数（不包括常数项）"
    
    print("  系数:", result.coefficients)
    print("  不包含常数项的稳健标准误测试通过")


if __name__ == "__main__":
    print("开始测试稳健标准误模型...")
    test_robust_errors_basic()
    test_robust_errors_no_constant()
    print("稳健标准误测试完成!")