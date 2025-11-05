"""
指数平滑模型测试脚本
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from econometrics.specific_data_modeling.time_series_panel_data.exponential_smoothing import exponential_smoothing_model, ExponentialSmoothingResult


def test_exponential_smoothing_basic():
    """测试基本指数平滑功能"""
    print("测试基本指数平滑功能...")
    
    # 生成带趋势的时间序列数据
    np.random.seed(42)
    n = 100
    trend = 0.3
    data = []
    
    # 生成带线性趋势和噪声的数据
    for i in range(n):
        value = 10 + trend * i + np.random.randn() * 2
        data.append(value)
    
    # 执行指数平滑模型（带趋势）
    result = exponential_smoothing_model(data, trend=True, seasonal=False, forecast_steps=5)
    
    # 验证结果类型
    assert isinstance(result, ExponentialSmoothingResult), "结果应为ExponentialSmoothingResult类型"
    
    # 验证模型类型
    assert "Exponential Smoothing with Trend" in result.model_type, "模型类型应包含趋势成分"
    
    # 验证统计量合理性
    assert result.n_obs == n, f"观测数量应为{n}"
    assert len(result.coefficients) >= 1, "应至少有1个系数"
    assert len(result.forecast) == 5, "预测步数应为5"
    
    print("  模型类型:", result.model_type)
    print("  系数数量:", len(result.coefficients))
    print("  系数:", result.coefficients[:3], "...")
    print("  平滑水平参数:", result.smoothing_level)
    print("  平滑趋势参数:", result.smoothing_trend)
    print("  MSE:", result.mse)
    print("  预测值:", result.forecast[:3], "...")
    print("  基本指数平滑功能测试通过")


def test_exponential_smoothing_simple():
    """测试简单指数平滑（无趋势）"""
    print("测试简单指数平滑（无趋势）...")
    
    # 生成平稳时间序列数据
    np.random.seed(42)
    n = 80
    mean_value = 5
    data = []
    
    # 生成平稳数据
    for i in range(n):
        value = mean_value + np.random.randn() * 1.5
        data.append(value)
    
    # 执行简单指数平滑模型（无趋势）
    result = exponential_smoothing_model(data, trend=False, seasonal=False, forecast_steps=3)
    
    # 验证模型类型
    assert "Exponential Smoothing" == result.model_type, "模型类型应为简单指数平滑"
    
    print("  模型类型:", result.model_type)
    print("  简单指数平滑测试通过")


def test_exponential_smoothing_errors():
    """测试指数平滑错误处理"""
    print("测试指数平滑错误处理...")
    
    # 测试空数据
    try:
        exponential_smoothing_model([])
        assert False, "应该抛出异常"
    except Exception:
        print("  空数据错误处理正确")
    
    # 测试数据过少
    try:
        exponential_smoothing_model([1])
        assert False, "应该抛出异常"
    except Exception:
        print("  数据过少错误处理正确")
    
    print("  指数平滑错误处理测试通过")


if __name__ == "__main__":
    print("开始测试指数平滑模型...")
    test_exponential_smoothing_basic()
    test_exponential_smoothing_simple()
    test_exponential_smoothing_errors()
    print("所有指数平滑测试通过!")