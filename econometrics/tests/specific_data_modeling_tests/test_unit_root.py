"""
单位根检验测试脚本
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from econometrics.specific_data_modeling.time_series_panel_data.unit_root_tests import adf_test, pp_test, kpss_test, UnitRootTestResult


def test_adf_test():
    """测试ADF单位根检验"""
    print("测试ADF单位根检验...")
    
    # 生成平稳时间序列数据 (AR(1)过程，系数<1)
    np.random.seed(42)
    n = 100
    ar_param = 0.6  # |ar_param| < 1，序列平稳
    data = [0]  # 初始值
    
    # 生成AR(1)过程: y[t] = 0.6 * y[t-1] + noise
    for i in range(1, n):
        data.append(ar_param * data[i-1] + np.random.randn() * 0.5)
    
    # 执行ADF检验
    result = adf_test(data)
    
    # 验证结果类型
    assert isinstance(result, UnitRootTestResult), "结果应为UnitRootTestResult类型"
    
    # 验证检验类型
    assert "Augmented Dickey-Fuller" in result.test_type, "检验类型应为ADF"
    
    # 验证统计量合理性
    assert result.n_obs == n, f"观测数量应为{n}"
    assert isinstance(result.stationary, bool), "平稳性判断应为布尔值"
    
    print("  检验统计量:", result.test_statistic)
    print("  p值:", result.p_value)
    print("  是否平稳:", result.stationary)
    print("  ADF单位根检验测试通过")


def test_adf_test_nonstationary():
    """测试ADF单位根检验（非平稳序列）"""
    print("测试ADF单位根检验（非平稳序列）...")
    
    # 生成非平稳时间序列数据 (随机游走)
    np.random.seed(42)
    n = 100
    data = [0]  # 初始值
    
    # 生成随机游走过程: y[t] = y[t-1] + noise
    for i in range(1, n):
        data.append(data[i-1] + np.random.randn() * 0.3)
    
    # 执行ADF检验
    result = adf_test(data)
    
    print("  检验统计量:", result.test_statistic)
    print("  p值:", result.p_value)
    print("  是否平稳:", result.stationary)
    print("  ADF单位根检验（非平稳序列）测试通过")


def test_pp_test():
    """测试PP单位根检验"""
    print("测试PP单位根检验...")
    
    # 生成平稳时间序列数据
    np.random.seed(42)
    n = 100
    ar_param = 0.7
    data = [0]  # 初始值
    
    # 生成AR(1)过程: y[t] = 0.7 * y[t-1] + noise
    for i in range(1, n):
        data.append(ar_param * data[i-1] + np.random.randn() * 0.5)
    
    # 执行PP检验
    result = pp_test(data)
    
    # 验证结果类型
    assert isinstance(result, UnitRootTestResult), "结果应为UnitRootTestResult类型"
    
    # 验证检验类型
    assert "Phillips-Perron" in result.test_type, "检验类型应为PP"
    
    print("  检验统计量:", result.test_statistic)
    print("  p值:", result.p_value)
    print("  是否平稳:", result.stationary)
    print("  PP单位根检验测试通过")


def test_kpss_test():
    """测试KPSS单位根检验"""
    print("测试KPSS单位根检验...")
    
    # 生成平稳时间序列数据
    np.random.seed(42)
    n = 100
    data = np.random.randn(n).tolist()  # 白噪声序列，平稳
    
    # 执行KPSS检验
    result = kpss_test(data)
    
    # 验证结果类型
    assert isinstance(result, UnitRootTestResult), "结果应为UnitRootTestResult类型"
    
    # 验证检验类型
    assert "KPSS" in result.test_type, "检验类型应为KPSS"
    
    print("  检验统计量:", result.test_statistic)
    print("  p值:", result.p_value)
    print("  是否平稳:", result.stationary)
    print("  KPSS单位根检验测试通过")


def test_unit_root_errors():
    """测试单位根检验错误处理"""
    print("测试单位根检验错误处理...")
    
    # 测试空数据
    try:
        adf_test([])
        assert False, "应该抛出异常"
    except Exception:
        print("  ADF空数据错误处理正确")
    
    try:
        pp_test([])
        assert False, "应该抛出异常"
    except Exception:
        print("  PP空数据错误处理正确")
    
    try:
        kpss_test([])
        assert False, "应该抛出异常"
    except Exception:
        print("  KPSS空数据错误处理正确")
    
    print("  单位根检验错误处理测试通过")


if __name__ == "__main__":
    print("开始测试单位根检验...")
    test_adf_test()
    test_adf_test_nonstationary()
    test_pp_test()
    test_kpss_test()
    test_unit_root_errors()
    print("所有单位根检验测试通过!")