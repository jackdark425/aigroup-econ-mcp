"""
双重差分法测试
"""

import numpy as np
import unittest
from econometrics.causal_inference.causal_identification_strategy import difference_in_differences


class TestDifferenceInDifferences(unittest.TestCase):
    
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


if __name__ == "__main__":
    unittest.main()