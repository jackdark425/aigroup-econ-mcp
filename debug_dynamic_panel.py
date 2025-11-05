"""
调试动态面板模型
"""
import sys
import os
import numpy as np

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 测试数据
y_data = [1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9]
x_data = [[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]]
entity_ids = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
time_periods = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

print("=== 调试动态面板模型 ===")
print(f"y_data: {y_data}")
print(f"x_data: {x_data}")
print(f"entity_ids: {entity_ids}")
print(f"time_periods: {time_periods}")

# 手动执行差分GMM的核心步骤
print("\n=== 手动执行差分GMM步骤 ===")

# 1. 数据转换
y_array = np.array(y_data)
print(f"y_array shape: {y_array.shape}")

# 检查x_data格式
print(f"x_data类型: {type(x_data)}")
print(f"x_data[0]类型: {type(x_data[0])}")

# 转换x_data
if isinstance(x_data[0], (list, tuple)):
    x_array = np.array(x_data)
    print(f"x_array shape: {x_array.shape}")
else:
    x_array = np.array(x_data).reshape(-1, 1)
    print(f"x_array shape: {x_array.shape}")

# 确保x_array是二维的
if x_array.ndim == 1:
    x_array = x_array.reshape(-1, 1)
    print(f"调整后x_array shape: {x_array.shape}")

n_obs = len(y_data)
n_vars = x_array.shape[1]
print(f"n_obs: {n_obs}, n_vars: {n_vars}")

# 2. 构建差分数据
dy = np.diff(y_array)
dx = np.diff(x_array, axis=0)
print(f"dy shape: {dy.shape}")
print(f"dx shape: {dx.shape}")

# 3. 构建工具变量矩阵
print("\n=== 构建工具变量矩阵 ===")
Z_list = []
for t in range(2, n_obs):  # 从第2期开始
    print(f"t={t}:")
    
    # 使用滞后水平作为工具变量
    lag_y = y_array[:t-1]  # 滞后因变量
    lag_x = x_array[:t-1, :]  # 滞后自变量
    
    print(f"  lag_y shape: {lag_y.shape}")
    print(f"  lag_x shape: {lag_x.shape}")
    
    # 构建该时期的工具变量
    lag_y_flat = lag_y.flatten() if lag_y.ndim > 1 else lag_y
    lag_x_flat = lag_x.flatten() if lag_x.ndim > 1 else lag_x
    
    print(f"  lag_y_flat shape: {lag_y_flat.shape}")
    print(f"  lag_x_flat shape: {lag_x_flat.shape}")
    
    try:
        z_t = np.concatenate([lag_y_flat, lag_x_flat])
        print(f"  z_t shape: {z_t.shape}")
        Z_list.append(z_t)
    except Exception as e:
        print(f"  错误: {e}")
        break

if Z_list:
    print(f"Z_list长度: {len(Z_list)}")
    try:
        Z = np.array(Z_list)
        print(f"Z shape: {Z.shape}")
    except Exception as e:
        print(f"创建Z数组错误: {e}")
else:
    print("无法构建工具变量")