"""
单元测试 - 统计分析模块
测试描述性统计、假设检验、相关性分析等基础统计功能
"""

import pytest
import numpy as np
from typing import Dict, List

from src.aigroup_econ_mcp.tools.statistics import (
    calculate_descriptive_stats,
    calculate_correlation_matrix,
    perform_hypothesis_test,
    normality_test
)


class TestDescriptiveStatistics:
    """测试描述性统计"""
    
    def test_descriptive_stats_basic(self):
        """测试基础描述性统计"""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = calculate_descriptive_stats(data)

        assert result.count == 10
        assert result.mean == 5.5
        assert result.median == 5.5
        assert result.min == 1
        assert result.max == 10
        assert result.std > 0
        assert result.variance > 0
        assert result.skewness is not None
        assert result.kurtosis is not None

    def test_descriptive_stats_empty_data(self):
        """测试空数据"""
        with pytest.raises(ValueError, match="数据不能为空"):
            calculate_descriptive_stats([])

    def test_descriptive_stats_single_value(self):
        """测试单值数据"""
        data = [5.0]
        result = calculate_descriptive_stats(data)
        
        assert result.count == 1
        assert result.mean == 5.0
        assert result.median == 5.0
        assert result.std == 0


class TestCorrelationAnalysis:
    """测试相关性分析"""
    
    def test_correlation_matrix_pearson(self):
        """测试Pearson相关系数矩阵"""
        data = {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
            "z": [1, 3, 5, 7, 9]
        }

        result = calculate_correlation_matrix(data, "pearson")

        assert "x" in result.correlation_matrix
        assert "y" in result.correlation_matrix
        assert "z" in result.correlation_matrix
        assert result.method == "pearson"
        assert result.n_obs == 5

    def test_correlation_matrix_spearman(self):
        """测试Spearman相关系数矩阵"""
        data = {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10]
        }

        result = calculate_correlation_matrix(data, "spearman")

        assert result.method == "spearman"
        assert len(result.correlation_matrix) == 2

    def test_correlation_matrix_single_variable(self):
        """测试单变量相关性分析"""
        data = {"x": [1, 2, 3]}
        
        with pytest.raises(ValueError, match="至少需要2个变量"):
            calculate_correlation_matrix(data, "pearson")


class TestHypothesisTesting:
    """测试假设检验"""
    
    def test_t_test_two_sample(self):
        """测试双样本t检验"""
        data1 = [1, 2, 3, 4, 5]
        data2 = [2, 3, 4, 5, 6]

        result = perform_hypothesis_test(data1, data2, "t_test")

        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert result["test_type"] == "双样本t检验"
        assert "confidence_interval" in result

    def test_t_test_one_sample(self):
        """测试单样本t检验"""
        data1 = [1, 2, 3, 4, 5]
        
        result = perform_hypothesis_test(data1, test_type="t_test")

        assert result["test_type"] == "单样本t检验"
        assert "statistic" in result
        assert "p_value" in result

    def test_adf_test(self):
        """测试ADF平稳性检验"""
        # 随机游走（非平稳）
        np.random.seed(42)
        data = np.cumsum(np.random.normal(0, 1, 50)).tolist()

        result = perform_hypothesis_test(data, test_type="adf_test")

        assert result["test_type"] == "ADF检验"
        assert "statistic" in result
        assert "p_value" in result
        assert "stationary" in result

    def test_f_test(self):
        """测试F检验"""
        data1 = [1, 2, 3, 4, 5]
        data2 = [2, 3, 4, 5, 6]

        result = perform_hypothesis_test(data1, data2, "f_test")

        assert result["test_type"] == "F检验"
        assert "statistic" in result
        assert "p_value" in result


class TestNormalityTesting:
    """测试正态性检验"""
    
    def test_normality_test_normal_data(self):
        """测试正态分布数据"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100).tolist()

        result = normality_test(data)

        assert "shapiro_wilk" in result
        assert "kolmogorov_smirnov" in result
        assert "statistic" in result["shapiro_wilk"]
        assert "p_value" in result["shapiro_wilk"]
        assert "statistic" in result["kolmogorov_smirnov"]
        assert "p_value" in result["kolmogorov_smirnov"]

    def test_normality_test_non_normal_data(self):
        """测试非正态分布数据"""
        # 均匀分布数据
        np.random.seed(42)
        data = np.random.uniform(0, 1, 100).tolist()

        result = normality_test(data)

        assert result["shapiro_wilk"]["p_value"] < 0.05  # 应该拒绝正态性假设


class TestDataValidation:
    """测试数据验证"""
    
    def test_invalid_data_types(self):
        """测试无效数据类型"""
        with pytest.raises(ValueError, match="数据必须为数值列表"):
            calculate_descriptive_stats(["a", "b", "c"])

    def test_missing_values(self):
        """测试缺失值处理"""
        data = [1, 2, None, 4, 5]
        
        with pytest.raises(ValueError, match="数据包含无效值"):
            calculate_descriptive_stats(data)

    def test_insufficient_data(self):
        """测试数据不足"""
        data = [1]
        
        with pytest.raises(ValueError, match="数据不足"):
            perform_hypothesis_test(data, test_type="t_test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])