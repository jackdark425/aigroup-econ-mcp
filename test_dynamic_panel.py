"""
测试动态面板模型
"""
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from econometrics.specific_data_modeling.time_series_panel_data.dynamic_panel_models import (
    diff_gmm_model, sys_gmm_model
)

# 测试数据
y_data = [1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9]
x_data = [[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]]
entity_ids = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
time_periods = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

print("测试数据:")
print(f"y_data: {y_data}")
print(f"x_data: {x_data}")
print(f"entity_ids: {entity_ids}")
print(f"time_periods: {time_periods}")

try:
    print("\n测试差分GMM模型...")
    result = diff_gmm_model(y_data, x_data, entity_ids, time_periods, lags=1)
    print("差分GMM模型成功!")
    print(f"系数: {result.coefficients}")
    print(f"标准误: {result.std_errors}")
except Exception as e:
    print(f"差分GMM模型失败: {e}")

try:
    print("\n测试系统GMM模型...")
    result = sys_gmm_model(y_data, x_data, entity_ids, time_periods, lags=1)
    print("系统GMM模型成功!")
    print(f"系数: {result.coefficients}")
    print(f"标准误: {result.std_errors}")
except Exception as e:
    print(f"系统GMM模型失败: {e}")