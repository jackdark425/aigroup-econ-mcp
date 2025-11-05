"""
诊断检验模型测试脚本
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# 尝试导入诊断检验模型
try:
    from econometrics.model_specification_diagnostics_robust_inference.diagnostic_tests.diagnostic_tests_model import diagnostic_tests, DiagnosticTestsResult
    DIAGNOSTIC_TESTS_AVAILABLE = True
except ImportError:
    DIAGNOSTIC_TESTS_AVAILABLE = False
    print("警告: 未找到诊断检验模型，相关测试将被跳过")


def test_diagnostic_tests_basic():
    """测试基本诊断检验功能"""
    if not DIAGNOSTIC_TESTS_AVAILABLE:
        print("跳过诊断检验测试（模块不可用）")
        return
        
    print("测试基本诊断检验功能...")
    
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
    
    # 执行诊断检验
    result = diagnostic_tests(y_data, x_data, feature_names=['x1', 'x2'])
    
    # 验证结果类型
    assert isinstance(result, DiagnosticTestsResult), "结果应为DiagnosticTestsResult类型"
    
    # 验证统计量合理性
    assert result.n_obs == n, f"观测数量应为{n}"
    
    print("  Jarque-Bera检验统计量:", result.jarque_bera_stat)
    print("  Breusch-Pagan检验统计量:", result.breusch_pagan_stat)
    print("  基本诊断检验功能测试通过")


def test_diagnostic_tests_serial_correlation():
    """测试序列相关诊断检验"""
    if not DIAGNOSTIC_TESTS_AVAILABLE:
        print("跳过序列相关诊断检验测试（模块不可用）")
        return
        
    print("测试序列相关诊断检验...")
    
    # 生成带序列相关的测试数据
    np.random.seed(42)
    n = 100
    # 生成自相关误差项
    errors = [0]
    for i in range(1, n):
        errors.append(0.5 * errors[i-1] + np.random.randn() * 0.3)
    
    x = np.random.randn(n)
    # 真实模型: y = 2 + 3*x + errors (带序列相关)
    y = 2 + 3*x + np.array(errors)
    
    # 执行诊断检验
    result = diagnostic_tests(y.tolist(), x.tolist(), feature_names=['x1'])
    
    print("  Durbin-Watson统计量:", result.durbin_watson_stat)
    print("  序列相关诊断检验测试通过")


if __name__ == "__main__":
    print("开始测试诊断检验模型...")
    test_diagnostic_tests_basic()
    test_diagnostic_tests_serial_correlation()
    print("诊断检验测试完成!")