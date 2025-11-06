"""
工具变量法测试
"""

import numpy as np
import unittest
from econometrics.causal_inference.causal_identification_strategy import instrumental_variables_2sls


class TestInstrumentalVariables(unittest.TestCase):
    
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


if __name__ == "__main__":
    unittest.main()