"""
ARIMA模型测试脚本
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from econometrics.specific_data_modeling.time_series_panel_data.arima_model import arima_model, ARIMAResult


def test_arima_basic():
    """测试基本ARIMA功能"""
    print("测试基本ARIMA功能...")
    
    # 生成AR(1)过程数据
    np.random.seed(42)
    n = 100
    ar_param = 0.6
    data = [0]  # 初始值
    
    # 生成AR(1)过程: y[t] = 0.6 * y[t-1] + noise
    for i in range(1, n):
        data.append(ar_param * data[i-1] + np.random.randn() * 0.5)
    
    # 执行ARIMA模型 (1,0,0) 即 AR(1)
    result = arima_model(data, order=(1, 0, 0), forecast_steps=5)
    
    # 验证结果类型
    assert isinstance(result, ARIMAResult), "结果应为ARIMAResult类型"
    
    # 验证模型类型
    assert "ARIMA(1,0,0)" in result.model_type, "模型类型应为ARIMA(1,0,0)"
    
    # 验证系数数量
    assert len(result.coefficients) >= 1, "应该至少有1个系数"
    
    # 验证统计量合理性
    assert result.n_obs == n, f"观测数量应为{n}"
    assert len(result.forecast) == 5, "预测步数应为5"
    
    print("  模型类型:", result.model_type)
    print("  系数:", result.coefficients)
    print("  AIC:", result.aic)
    print("  预测值:", result.forecast[:3], "...")
    print("  基本ARIMA功能测试通过")


def test_arima_integrated():
    """测试带积分的ARIMA模型"""
    print("测试带积分的ARIMA模型...")
    
    # 生成带趋势的随机游走数据
    np.random.seed(42)
    n = 80
    data = [0]  # 初始值
    
    # 生成随机游走过程: y[t] = y[t-1] + noise
    for i in range(1, n):
        data.append(data[i-1] + np.random.randn() * 0.3)
    
    # 添加线性趋势
    data = [data[i] + 0.1 * i for i in range(len(data))]
    
    # 执行ARIMA模型 (0,1,0) 即随机游走
    result = arima_model(data, order=(0, 1, 0), forecast_steps=3)
    
    # 验证模型类型
    assert "ARIMA(0,1,0)" in result.model_type, "模型类型应为ARIMA(0,1,0)"
    
    print("  模型类型:", result.model_type)
    print("  预测值:", result.forecast)
    print("  带积分的ARIMA模型测试通过")


def test_arima_errors():
    """测试ARIMA错误处理"""
    print("测试ARIMA错误处理...")
    
    # 测试空数据
    try:
        arima_model([])
        assert False, "应该抛出异常"
    except Exception:
        print("  空数据错误处理正确")
    
    print("  ARIMA错误处理测试通过")


if __name__ == "__main__":
    print("开始测试ARIMA模型...")
    test_arima_basic()
    test_arima_integrated()
    test_arima_errors()
    print("所有ARIMA测试通过!")