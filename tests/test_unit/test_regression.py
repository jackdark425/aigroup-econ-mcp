"""
单元测试 - 回归分析模块
测试OLS回归、模型诊断、VIF计算等回归分析功能
"""

import pytest
import numpy as np
from typing import List

from src.aigroup_econ_mcp.tools.regression import (
    perform_ols_regression,
    calculate_vif,
    run_diagnostic_tests,
    stepwise_regression
)


class TestOLSRegression:
    """测试OLS回归分析"""
    
    def test_ols_regression_basic(self):
        """测试基础OLS回归"""
        y = [1, 2, 3, 4, 5]
        X = [[1], [2], [3], [4], [5]]
        feature_names = ["x"]

        result = perform_ols_regression(y, X, feature_names)

        assert "const" in result.coefficients
        assert "x" in result.coefficients
        assert result.rsquared >= 0
        assert result.rsquared <= 1
        assert result.n_obs == 5
        assert result.df_model == 1
        assert result.df_resid == 3
        assert result.f_statistic > 0
        assert result.f_pvalue >= 0
        assert result.f_pvalue <= 1

    def test_ols_regression_multiple_features(self):
        """测试多特征OLS回归"""
        y = [10, 12, 15, 18, 20]
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        feature_names = ["x1", "x2"]

        result = perform_ols_regression(y, X, feature_names)

        assert len(result.coefficients) == 3  # 常数项 + 2个特征
        assert result.rsquared >= 0
        assert result.rsquared <= 1
        assert result.n_obs == 5

    def test_ols_regression_no_constant(self):
        """测试无常数项的OLS回归"""
        y = [1, 2, 3, 4, 5]
        X = [[1], [2], [3], [4], [5]]
        feature_names = ["x"]

        result = perform_ols_regression(y, X, feature_names, add_constant=False)

        assert "const" not in result.coefficients
        assert "x" in result.coefficients
        assert result.n_obs == 5

    def test_ols_regression_insufficient_data(self):
        """测试数据不足"""
        y = [1, 2]
        X = [[1], [2]]
        
        with pytest.raises(ValueError, match="数据不足"):
            perform_ols_regression(y, X, ["x"])

    def test_ols_regression_dimension_mismatch(self):
        """测试维度不匹配"""
        y = [1, 2, 3]
        X = [[1], [2]]  # 长度不匹配
        
        with pytest.raises(ValueError, match="数据长度不匹配"):
            perform_ols_regression(y, X, ["x"])


class TestVIFCalculation:
    """测试VIF计算"""
    
    def test_vif_calculation_basic(self):
        """测试基础VIF计算"""
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        feature_names = ["x1", "x2"]

        vif_result = calculate_vif(X, feature_names)

        assert "x1" in vif_result
        assert "x2" in vif_result
        assert all(vif >= 1 for vif in vif_result.values())

    def test_vif_calculation_perfect_collinearity(self):
        """测试完全共线性"""
        X = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
        feature_names = ["x1", "x2"]

        vif_result = calculate_vif(X, feature_names)

        # 完全共线性时VIF应该很大
        assert all(vif > 10 for vif in vif_result.values())

    def test_vif_calculation_single_feature(self):
        """测试单特征VIF计算"""
        X = [[1], [2], [3]]
        feature_names = ["x"]
        
        with pytest.raises(ValueError, match="至少需要2个特征"):
            calculate_vif(X, feature_names)


class TestDiagnosticTests:
    """测试模型诊断"""
    
    def test_diagnostic_tests_basic(self):
        """测试基础模型诊断"""
        y = [1, 2, 3, 4, 5]
        X = [[1], [2], [3], [4], [5]]

        diagnostics = run_diagnostic_tests(y, X)

        assert hasattr(diagnostics, "jb_statistic")
        assert hasattr(diagnostics, "jb_pvalue")
        assert hasattr(diagnostics, "dw_statistic")
        assert hasattr(diagnostics, "breusch_pagan_statistic")
        assert hasattr(diagnostics, "breusch_pagan_pvalue")
        assert "vif" in diagnostics.model_dump()

    def test_diagnostic_tests_with_vif(self):
        """测试包含VIF的诊断"""
        y = [10, 12, 15, 18, 20]
        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]

        diagnostics = run_diagnostic_tests(y, X, calculate_vif=True)

        assert "vif" in diagnostics.model_dump()
        vif_values = diagnostics.model_dump()["vif"]
        assert len(vif_values) == 2

    def test_diagnostic_tests_insufficient_data(self):
        """测试数据不足的诊断"""
        y = [1, 2, 3]
        X = [[1], [2], [3]]
        
        with pytest.raises(ValueError, match="数据不足"):
            run_diagnostic_tests(y, X)


class TestStepwiseRegression:
    """测试逐步回归"""
    
    def test_stepwise_regression_forward(self):
        """测试前向逐步回归"""
        y = [10, 12, 15, 18, 20, 22, 25, 28, 30, 32]
        X = [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9],
            [8, 9, 10],
            [9, 10, 11],
            [10, 11, 12]
        ]
        feature_names = ["x1", "x2", "x3"]

        result = stepwise_regression(y, X, feature_names, direction="forward")

        assert "selected_features" in result
        assert "final_model" in result
        assert "selection_history" in result
        assert len(result["selected_features"]) > 0
        assert result["final_model"]["rsquared"] >= 0

    def test_stepwise_regression_backward(self):
        """测试后向逐步回归"""
        y = [10, 12, 15, 18, 20, 22, 25, 28, 30, 32]
        X = [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9],
            [8, 9, 10],
            [9, 10, 11],
            [10, 11, 12]
        ]
        feature_names = ["x1", "x2", "x3"]

        result = stepwise_regression(y, X, feature_names, direction="backward")

        assert "selected_features" in result
        assert "final_model" in result
        assert len(result["selected_features"]) >= 0

    def test_stepwise_regression_bidirectional(self):
        """测试双向逐步回归"""
        y = [10, 12, 15, 18, 20, 22, 25, 28, 30, 32]
        X = [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9],
            [8, 9, 10],
            [9, 10, 11],
            [10, 11, 12]
        ]
        feature_names = ["x1", "x2", "x3"]

        result = stepwise_regression(y, X, feature_names, direction="both")

        assert "selected_features" in result
        assert "final_model" in result
        assert len(result["selected_features"]) > 0

    def test_stepwise_regression_insufficient_data(self):
        """测试数据不足的逐步回归"""
        y = [1, 2, 3]
        X = [[1, 2], [2, 3], [3, 4]]
        
        with pytest.raises(ValueError, match="数据不足"):
            stepwise_regression(y, X, ["x1", "x2"])


class TestRegressionValidation:
    """测试回归分析验证"""
    
    def test_invalid_feature_names(self):
        """测试无效特征名"""
        y = [1, 2, 3, 4, 5]
        X = [[1], [2], [3], [4], [5]]
        
        with pytest.raises(ValueError, match="特征名数量不匹配"):
            perform_ols_regression(y, X, ["x1", "x2"])  # 特征名过多

    def test_non_numeric_data(self):
        """测试非数值数据"""
        y = [1, 2, 3, 4, 5]
        X = [["a"], ["b"], ["c"], ["d"], ["e"]]  # 非数值数据
        
        with pytest.raises(ValueError, match="数据必须为数值"):
            perform_ols_regression(y, X, ["x"])

    def test_missing_values(self):
        """测试缺失值"""
        y = [1, 2, None, 4, 5]
        X = [[1], [2], [3], [4], [5]]
        
        with pytest.raises(ValueError, match="数据包含无效值"):
            perform_ols_regression(y, X, ["x"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])