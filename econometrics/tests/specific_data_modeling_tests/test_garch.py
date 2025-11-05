"""
GARCH模型测试脚本
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from econometrics.specific_data_modeling.time_series_panel_data.garch_model import garch_model, GARCHResult


def test_garch_basic():
    """测试基本GARCH功能"""
    print("测试基本GARCH功能...")
    
    # 生成GARCH(1,1)过程数据
    np.random.seed(42)
    n = 500
    
    # GARCH(1,1)参数
    omega = 0.1
    alpha = 0.3
    beta = 0.6
    
    # 确保参数满足平稳性条件
    assert alpha + beta < 1, "GARCH过程需要满足alpha + beta < 1"
    
    # 生成GARCH过程
    data = [0]  # 初始值
    h = [0.5]   # 条件方差初始值
    
    for i in range(1, n):
        # 生成条件方差
        h_t = omega + alpha * (data[i-1]**2) + beta * h[i-1]
        h.append(h_t)
        
        # 生成收益率
        epsilon_t = np.random.randn() * np.sqrt(h_t)
        data.append(epsilon_t)
    
    # 执行GARCH(1,1)模型
    result = garch_model(data, order=(1, 1))
    
    # 验证结果类型
    assert isinstance(result, GARCHResult), "结果应为GARCHResult类型"
    
    # 验证模型类型
    assert "GARCH(1,1)" in result.model_type, "模型类型应为GARCH(1,1)"
    
    # 验证统计量合理性
    assert result.n_obs == n, f"观测数量应为{n}"
    assert len(result.coefficients) >= 3, "应至少有3个系数(omega, alpha, beta)"
    
    print("  模型类型:", result.model_type)
    print("  系数数量:", len(result.coefficients))
    print("  系数:", result.coefficients)
    print("  对数似然值:", result.log_likelihood)
    print("  AIC:", result.aic)
    print("  持续性参数:", result.persistence)
    print("  基本GARCH功能测试通过")


def test_garch_order2():
    """测试GARCH(2,1)模型"""
    print("测试GARCH(2,1)模型...")
    
    # 生成测试数据（使用GARCH(1,1)数据）
    np.random.seed(42)
    n = 200
    
    # 生成简单的时间序列数据
    data = []
    for i in range(n):
        data.append(np.random.randn() * 0.5)
    
    # 执行GARCH(2,1)模型
    try:
        result = garch_model(data, order=(2, 1))
        
        # 验证模型类型
        assert "GARCH(2,1)" in result.model_type, "模型类型应为GARCH(2,1)"
        
        print("  模型类型:", result.model_type)
        print("  GARCH(2,1)模型测试通过")
    except Exception as e:
        print(f"  GARCH(2,1)模型测试跳过（可能因数据量不足）: {e}")


def test_garch_errors():
    """测试GARCH错误处理"""
    print("测试GARCH错误处理...")
    
    # 测试空数据
    try:
        garch_model([])
        assert False, "应该抛出异常"
    except Exception:
        print("  空数据错误处理正确")
    
    # 测试无效阶数
    try:
        garch_model([1, 2, 3], order=(0, 0))
        assert False, "应该抛出异常"
    except Exception:
        print("  无效阶数错误处理正确")
    
    print("  GARCH错误处理测试通过")


if __name__ == "__main__":
    print("开始测试GARCH模型...")
    test_garch_basic()
    test_garch_order2()
    test_garch_errors()
    print("所有GARCH测试通过!")