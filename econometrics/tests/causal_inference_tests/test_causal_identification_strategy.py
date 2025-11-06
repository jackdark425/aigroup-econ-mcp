"""
因果识别策略模块测试
"""

import numpy as np
import unittest
from econometrics.causal_inference.causal_identification_strategy import (
    instrumental_variables_2sls,
    difference_in_differences,
    regression_discontinuity,
    fixed_effects_model,
    random_effects_model
)


class TestCausalIdentificationStrategy(unittest.TestCase):
    
    def test_instrumental_variables_2sls(self):
        """测试工具变量法"""
        # 生成模拟数据
        np.random.seed(42)
        n = 100
        
        # 工具变量
        z = np.random.normal(0, 1, n)
        
        # 内生变量（与误差项相关）
        e1 = np.random.normal(0, 1, n)
        x = 1 + 0.5 * z + e1
        
        # 结果变量
        e2 = np.random.normal(0, 1, n)
        y = 2 + 1.5 * x + e2 + 0.3 * e1  # 包含内生性
        
        # 执行工具变量回归
        result = instrumental_variables_2sls(
            y=y.tolist(),
            x=x.reshape(-1, 1).tolist(),
            instruments=z.reshape(-1, 1).tolist()
        )
        
        # 检查结果
        self.assertIsNotNone(result.estimate)
        self.assertIsNotNone(result.std_error)
        self.assertIsNotNone(result.p_value)
        self.assertGreater(result.n_observations, 0)
    
    def test_difference_in_differences(self):
        """测试双重差分法"""
        # 生成模拟数据
        np.random.seed(42)
        n = 200
        
        # 处理组标识（0=控制组，1=处理组）
        treatment = np.concatenate([np.zeros(100), np.ones(100)]).tolist()
        
        # 时间标识（0=处理前，1=处理后）
        time_period = np.concatenate([np.zeros(50), np.ones(50), np.zeros(50), np.ones(50)]).tolist()
        
        # 结果变量
        # 控制组处理前均值为10，处理后为10
        # 处理组处理前均值为10，处理后为12（处理效应为2）
        outcome = []
        for i in range(n):
            if treatment[i] == 0 and time_period[i] == 0:
                outcome.append(np.random.normal(10, 1))
            elif treatment[i] == 0 and time_period[i] == 1:
                outcome.append(np.random.normal(10, 1))
            elif treatment[i] == 1 and time_period[i] == 0:
                outcome.append(np.random.normal(10, 1))
            else:  # treatment[i] == 1 and time_period[i] == 1
                outcome.append(np.random.normal(12, 1))
        
        # 执行DID分析
        result = difference_in_differences(
            treatment=treatment,
            time_period=time_period,
            outcome=outcome
        )
        
        # 检查结果
        self.assertIsNotNone(result.estimate)
        self.assertIsNotNone(result.std_error)
        self.assertIsNotNone(result.p_value)
        self.assertGreater(result.n_observations, 0)
    
    def test_regression_discontinuity(self):
        """测试断点回归设计"""
        # 生成模拟数据
        np.random.seed(42)
        n = 200
        cutoff = 0.0
        
        # 运行变量
        running_variable = np.random.uniform(-1, 1, n).tolist()
        
        # 结果变量 - 在断点处有跳跃
        outcome = []
        for r in running_variable:
            if r >= cutoff:
                outcome.append(2 + 1.5 * r + np.random.normal(0, 0.5) + 1.0)  # +1.0是处理效应
            else:
                outcome.append(2 + 1.5 * r + np.random.normal(0, 0.5))
        
        # 执行RDD分析
        result = regression_discontinuity(
            running_variable=running_variable,
            outcome=outcome,
            cutoff=cutoff,
            bandwidth=0.5
        )
        
        # 检查结果
        self.assertIsNotNone(result.estimate)
        self.assertIsNotNone(result.std_error)
        self.assertIsNotNone(result.p_value)
        self.assertGreater(result.n_observations, 0)
        self.assertEqual(result.discontinuity_location, cutoff)
    
    def test_fixed_effects_model(self):
        """测试固定效应模型"""
        # 生成面板数据
        np.random.seed(42)
        n_entities = 20
        n_periods = 10
        n = n_entities * n_periods
        
        # 个体标识
        entity_ids = [f"entity_{i}" for i in range(n_entities) for _ in range(n_periods)]
        
        # 时间标识
        time_periods = [f"period_{t}" for _ in range(n_entities) for t in range(n_periods)]
        
        # 自变量
        x = np.random.normal(0, 1, (n, 2)).tolist()
        
        # 因变量（包含个体固定效应）
        entity_effects = np.random.normal(0, 1, n_entities)
        y = []
        for i in range(n):
            entity_idx = i // n_periods
            y_value = 1 + 2 * x[i][0] + 1.5 * x[i][1] + entity_effects[entity_idx] + np.random.normal(0, 0.5)
            y.append(y_value)
        
        # 执行固定效应模型
        result = fixed_effects_model(
            y=y,
            x=x,
            entity_ids=entity_ids,
            time_periods=time_periods
        )
        
        # 检查结果
        self.assertIsNotNone(result.estimate)
        self.assertIsNotNone(result.std_error)
        self.assertIsNotNone(result.p_value)
        self.assertEqual(result.n_observations, n)
        self.assertEqual(result.model_type, "FE")
    
    def test_random_effects_model(self):
        """测试随机效应模型"""
        # 生成面板数据
        np.random.seed(42)
        n_entities = 20
        n_periods = 10
        n = n_entities * n_periods
        
        # 个体标识
        entity_ids = [f"entity_{i}" for i in range(n_entities) for _ in range(n_periods)]
        
        # 时间标识
        time_periods = [f"period_{t}" for _ in range(n_entities) for t in range(n_periods)]
        
        # 自变量
        x = np.random.normal(0, 1, (n, 2)).tolist()
        
        # 因变量（包含个体随机效应）
        entity_effects = np.random.normal(0, 1, n_entities)
        y = []
        for i in range(n):
            entity_idx = i // n_periods
            y_value = 1 + 2 * x[i][0] + 1.5 * x[i][1] + entity_effects[entity_idx] + np.random.normal(0, 0.5)
            y.append(y_value)
        
        # 执行随机效应模型
        result = random_effects_model(
            y=y,
            x=x,
            entity_ids=entity_ids,
            time_periods=time_periods
        )
        
        # 检查结果
        self.assertIsNotNone(result.estimate)
        self.assertIsNotNone(result.std_error)
        self.assertIsNotNone(result.p_value)
        self.assertEqual(result.n_observations, n)
        self.assertEqual(result.model_type, "RE")


if __name__ == "__main__":
    unittest.main()