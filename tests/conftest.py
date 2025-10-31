"""
pytest配置文件
提供测试用的fixtures和配置
"""

import pytest
import numpy as np
import sys
import os
from typing import Dict, List, Any

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_economic_data():
    """提供样本经济数据"""
    np.random.seed(42)
    
    n_samples = 100
    gdp_growth = np.random.normal(3.0, 0.8, n_samples)
    inflation = np.random.normal(2.0, 0.5, n_samples)
    unemployment = np.random.normal(5.0, 1.2, n_samples)
    interest_rate = np.random.normal(2.5, 0.7, n_samples)
    stock_returns = np.random.normal(0.1, 2.0, n_samples)
    
    return {
        "gdp_growth": gdp_growth.tolist(),
        "inflation": inflation.tolist(),
        "unemployment": unemployment.tolist(),
        "interest_rate": interest_rate.tolist(),
        "stock_returns": stock_returns.tolist()
    }


@pytest.fixture
def sample_regression_data():
    """提供样本回归数据"""
    np.random.seed(42)
    
    n_samples = 50
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    x3 = np.random.normal(0, 1, n_samples)
    
    # 线性关系：y = 2*x1 + 3*x2 - 1*x3 + 噪声
    y = 2 * x1 + 3 * x2 - 1 * x3 + np.random.normal(0, 0.5, n_samples)
    
    X_data = [[x1_val, x2_val, x3_val] for x1_val, x2_val, x3_val in zip(x1, x2, x3)]
    feature_names = ["特征1", "特征2", "特征3"]
    
    return {
        "y_data": y.tolist(),
        "X_data": X_data,
        "feature_names": feature_names
    }


@pytest.fixture
def sample_time_series_data():
    """提供样本时间序列数据"""
    np.random.seed(42)
    
    n_samples = 100
    trend = np.linspace(0, 10, n_samples)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 12)
    noise = np.random.normal(0, 1, n_samples)
    ts_data = trend + seasonal + noise
    
    return {
        "data": ts_data.tolist(),
        "trend": trend.tolist(),
        "seasonal": seasonal.tolist(),
        "noise": noise.tolist()
    }


@pytest.fixture
def sample_panel_data():
    """提供样本面板数据"""
    np.random.seed(42)
    
    n_entities = 5
    n_periods = 4
    n_obs = n_entities * n_periods
    
    y_data = []
    X_data = []
    entity_ids = []
    time_periods = []
    
    for i in range(n_entities):
        entity_id = f"Company_{i+1}"
        for j in range(n_periods):
            time_period = f"Year_{2020 + j}"
            
            # 生成数据
            x1 = np.random.normal(10, 2)
            x2 = np.random.normal(5, 1)
            y = 2 * x1 + 3 * x2 + np.random.normal(0, 1)
            
            y_data.append(y)
            X_data.append([x1, x2])
            entity_ids.append(entity_id)
            time_periods.append(time_period)
    
    feature_names = ["广告支出", "研发投入"]
    
    return {
        "y_data": y_data,
        "X_data": X_data,
        "entity_ids": entity_ids,
        "time_periods": time_periods,
        "feature_names": feature_names
    }


@pytest.fixture
def sample_csv_content():
    """提供样本CSV内容"""
    return """GDP增长率,通货膨胀率,失业率
3.2,2.1,4.5
2.8,2.3,4.2
3.5,1.9,4.0
2.9,2.4,4.3
3.1,2.2,4.1"""


@pytest.fixture
def sample_json_content():
    """提供样本JSON内容"""
    return {
        "GDP增长率": [3.2, 2.8, 3.5, 2.9, 3.1],
        "通货膨胀率": [2.1, 2.3, 1.9, 2.4, 2.2],
        "失业率": [4.5, 4.2, 4.0, 4.3, 4.1]
    }


@pytest.fixture
def sample_ml_data():
    """提供样本机器学习数据"""
    np.random.seed(42)
    
    n_samples = 100
    X_data = []
    y_data = []
    
    for i in range(n_samples):
        x1 = np.random.normal(0, 1)
        x2 = np.random.normal(0, 1)
        x3 = np.random.normal(0, 1)
        
        # 非线性关系
        y = 2 * x1 + 3 * x2**2 - 1 * np.sin(x3) + np.random.normal(0, 0.5)
        
        X_data.append([x1, x2, x3])
        y_data.append(y)
    
    feature_names = ["特征1", "特征2", "特征3"]
    
    return {
        "y_data": y_data,
        "X_data": X_data,
        "feature_names": feature_names
    }


def pytest_configure(config):
    """pytest配置"""
    # 添加自定义标记
    config.addinivalue_line(
        "markers", "slow: 标记为慢速测试（需要较长时间）"
    )
    config.addinivalue_line(
        "markers", "integration: 标记为集成测试"
    )
    config.addinivalue_line(
        "markers", "unit: 标记为单元测试"
    )
    config.addinivalue_line(
        "markers", "performance: 标记为性能测试"
    )
    config.addinivalue_line(
        "markers", "file_input: 标记为文件输入测试"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试项集合"""
    # 根据标记重新排序测试
    unit_tests = []
    integration_tests = []
    file_input_tests = []
    performance_tests = []
    slow_tests = []
    other_tests = []
    
    for item in items:
        if item.get_closest_marker("slow"):
            slow_tests.append(item)
        elif item.get_closest_marker("integration"):
            integration_tests.append(item)
        elif item.get_closest_marker("file_input"):
            file_input_tests.append(item)
        elif item.get_closest_marker("performance"):
            performance_tests.append(item)
        elif item.get_closest_marker("unit"):
            unit_tests.append(item)
        else:
            other_tests.append(item)
    
    # 重新排序：单元测试 -> 集成测试 -> 文件输入测试 -> 性能测试 -> 慢速测试 -> 其他测试
    items[:] = unit_tests + integration_tests + file_input_tests + performance_tests + slow_tests + other_tests