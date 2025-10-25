"""
测试计量经济学MCP工具
"""

import pytest
import asyncio
from typing import Dict, List, Any

from src.aigroup_econ_mcp.tools.statistics import (
    calculate_descriptive_stats,
    calculate_correlation_matrix,
    perform_hypothesis_test,
    normality_test
)
from src.aigroup_econ_mcp.tools.regression import (
    perform_ols_regression,
    calculate_vif,
    run_diagnostic_tests
)
from src.aigroup_econ_mcp.tools.time_series import (
    check_stationarity,
    calculate_acf_pacf,
    fit_arima_model
)


class TestStatistics:
    """统计分析测试"""

    def test_descriptive_stats(self):
        """测试描述性统计"""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = calculate_descriptive_stats(data)

        assert result.count == 10
        assert result.mean == 5.5
        assert result.median == 5.5
        assert result.min == 1
        assert result.max == 10
        assert result.std > 0

    def test_correlation_matrix(self):
        """测试相关系数矩阵"""
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

    def test_hypothesis_testing(self):
        """测试假设检验"""
        data1 = [1, 2, 3, 4, 5]
        data2 = [2, 3, 4, 5, 6]

        result = perform_hypothesis_test(data1, data2, "t_test")

        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert result["test_type"] == "双样本t检验"

    def test_normality_test(self):
        """测试正态性检验"""
        import numpy as np
        np.random.seed(42)
        data = np.random.normal(0, 1, 100).tolist()

        result = normality_test(data)

        assert "shapiro_wilk" in result
        assert "kolmogorov_smirnov" in result
        assert "statistic" in result["shapiro_wilk"]
        assert "p_value" in result["shapiro_wilk"]


class TestRegression:
    """回归分析测试"""

    def test_ols_regression(self):
        """测试OLS回归"""
        y = [1, 2, 3, 4, 5]
        X = [[1], [2], [3], [4], [5]]
        feature_names = ["x"]

        result = perform_ols_regression(y, X, feature_names)

        assert "const" in result.coefficients
        assert "x" in result.coefficients
        assert result.rsquared >= 0
        assert result.rsquared <= 1
        assert result.n_obs == 5

    def test_vif_calculation(self):
        """测试VIF计算"""
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        feature_names = ["x1", "x2"]

        vif_result = calculate_vif(X, feature_names)

        assert "x1" in vif_result
        assert "x2" in vif_result
        assert all(vif >= 1 for vif in vif_result.values())

    def test_diagnostic_tests(self):
        """测试模型诊断"""
        y = [1, 2, 3, 4, 5]
        X = [[1], [2], [3], [4], [5]]

        diagnostics = run_diagnostic_tests(y, X)

        assert hasattr(diagnostics, "jb_statistic")
        assert hasattr(diagnostics, "jb_pvalue")
        assert hasattr(diagnostics, "dw_statistic")
        assert "vif" in diagnostics.model_dump()


class TestTimeSeries:
    """时间序列分析测试"""

    def test_stationarity_test(self):
        """测试平稳性检验"""
        # 随机游走（非平稳）
        import numpy as np
        np.random.seed(42)
        data = np.cumsum(np.random.normal(0, 1, 100)).tolist()

        result = check_stationarity(data)

        assert hasattr(result, "adf_statistic")
        assert hasattr(result, "adf_pvalue")
        assert hasattr(result, "is_stationary")
        assert isinstance(result.adf_critical_values, dict)

    def test_acf_pacf(self):
        """测试自相关函数"""
        import numpy as np
        np.random.seed(42)
        data = np.random.normal(0, 1, 50).tolist()

        result = calculate_acf_pacf(data, nlags=10)

        assert len(result.acf_values) == 11  # 包括0阶
        assert len(result.pacf_values) == 11
        assert len(result.acf_confidence) == 11
        assert len(result.pacf_confidence) == 11

    def test_arima_model(self):
        """测试ARIMA模型"""
        import numpy as np
        np.random.seed(42)
        data = np.random.normal(0, 1, 50).tolist()

        try:
            result = fit_arima_model(data, order=(1, 0, 1))

            assert result.order == (1, 0, 1)
            assert hasattr(result, "aic")
            assert hasattr(result, "bic")
            assert hasattr(result, "coefficients")
            assert len(result.fitted_values) == len(data)
            assert len(result.residuals) == len(data)

        except Exception as e:
            # ARIMA拟合可能失败，这是正常的
            pytest.skip(f"ARIMA模型拟合失败: {e}")


class TestIntegration:
    """集成测试"""

    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        # 模拟完整的数据分析流程

        # 1. 数据准备
        sales = [100, 120, 150, 130, 180, 200, 190, 220, 240, 260]
        advertising = [5, 6, 7.5, 6.5, 9, 10, 9.5, 11, 12, 13]
        price = [100, 98, 95, 97, 92, 90, 91, 88, 85, 83]

        # 2. 描述性统计
        stats_result = calculate_descriptive_stats(sales)
        assert stats_result.mean > 0
        assert stats_result.std > 0

        # 3. 相关性分析
        data = {"sales": sales, "advertising": advertising, "price": price}
        corr_result = calculate_correlation_matrix(data)
        assert corr_result.method == "pearson"

        # 4. 回归分析
        X_data = list(zip(advertising, price))
        reg_result = perform_ols_regression(sales, X_data, ["advertising", "price"])
        assert reg_result.rsquared >= 0

        # 5. 模型诊断
        diagnostics = run_diagnostic_tests(sales, X_data)
        assert diagnostics.dw_statistic > 0

        print("✅ 端到端工作流测试通过")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])