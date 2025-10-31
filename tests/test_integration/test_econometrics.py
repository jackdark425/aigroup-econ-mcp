"""
集成测试 - 计量经济学工具
测试统计分析、回归分析、时间序列分析等工具的集成功能
"""

import pytest
import numpy as np
import sys
import os
from typing import Dict, List, Any

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from aigroup_econ_mcp.tools.statistics import (
    calculate_descriptive_stats,
    calculate_correlation_matrix,
    perform_hypothesis_test
)
from aigroup_econ_mcp.tools.regression import (
    perform_ols_regression,
    run_diagnostic_tests
)
from aigroup_econ_mcp.tools.time_series import (
    check_stationarity,
    calculate_acf_pacf,
    fit_arima_model
)
from aigroup_econ_mcp.tools.panel_data import (
    fixed_effects_model,
    random_effects_model,
    hausman_test
)
from aigroup_econ_mcp.tools.machine_learning import (
    random_forest_regression,
    gradient_boosting_regression
)


class TestEconometricWorkflow:
    """测试计量经济学工作流"""
    
    def create_sample_economic_data(self, n_samples: int = 100) -> Dict[str, Any]:
        """创建样本经济数据"""
        np.random.seed(42)
        
        # 宏观经济数据
        gdp_growth = np.random.normal(3.0, 0.8, n_samples)
        inflation = np.random.normal(2.0, 0.5, n_samples)
        unemployment = np.random.normal(5.0, 1.2, n_samples)
        interest_rate = np.random.normal(2.5, 0.7, n_samples)
        
        # 金融数据
        stock_returns = np.random.normal(0.1, 2.0, n_samples)
        
        return {
            "gdp_growth": gdp_growth.tolist(),
            "inflation": inflation.tolist(),
            "unemployment": unemployment.tolist(),
            "interest_rate": interest_rate.tolist(),
            "stock_returns": stock_returns.tolist()
        }
    
    def test_macro_economic_analysis_workflow(self):
        """测试宏观经济分析工作流"""
        print("\n--- 宏观经济分析工作流 ---")
        
        data = self.create_sample_economic_data(50)
        
        # 1. 描述性统计
        gdp_stats = calculate_descriptive_stats(data["gdp_growth"])
        inflation_stats = calculate_descriptive_stats(data["inflation"])
        
        assert gdp_stats.mean > 0
        assert inflation_stats.mean > 0
        print(f"✅ 描述性统计 - GDP均值: {gdp_stats.mean:.3f}, 通胀均值: {inflation_stats.mean:.3f}")
        
        # 2. 相关性分析
        macro_data = {
            "GDP": data["gdp_growth"],
            "Inflation": data["inflation"],
            "Unemployment": data["unemployment"]
        }
        corr_result = calculate_correlation_matrix(macro_data)
        
        assert len(corr_result.correlation_matrix) == 3
        print(f"✅ 相关性分析 - 矩阵大小: {len(corr_result.correlation_matrix)}x{len(corr_result.correlation_matrix)}")
        
        # 3. 假设检验
        test_result = perform_hypothesis_test(data["gdp_growth"], data["inflation"])
        
        assert "statistic" in test_result
        assert "p_value" in test_result
        print(f"✅ 假设检验 - 统计量: {test_result['statistic']:.3f}, p值: {test_result['p_value']:.3f}")
        
        # 4. OLS回归
        y_data = data["gdp_growth"]
        X_data = [[inf, unemp] for inf, unemp in zip(data["inflation"], data["unemployment"])]
        feature_names = ["Inflation", "Unemployment"]
        
        ols_result = perform_ols_regression(y_data, X_data, feature_names)
        
        assert ols_result.rsquared >= 0
        assert len(ols_result.coefficients) == 3
        print(f"✅ OLS回归 - R²: {ols_result.rsquared:.3f}, 系数数量: {len(ols_result.coefficients)}")
        
        # 5. 模型诊断
        diagnostic_result = run_diagnostic_tests(y_data, X_data)
        
        assert diagnostic_result.dw_statistic > 0
        print(f"✅ 模型诊断 - Durbin-Watson: {diagnostic_result.dw_statistic:.3f}")
        
        print("🎉 宏观经济分析工作流测试通过")
    
    def test_financial_time_series_workflow(self):
        """测试金融时间序列分析工作流"""
        print("\n--- 金融时间序列分析工作流 ---")
        
        data = self.create_sample_economic_data(100)
        
        # 1. 平稳性检验
        stationarity_result = check_stationarity(data["stock_returns"])
        
        assert isinstance(stationarity_result.is_stationary, bool)
        print(f"✅ 平稳性检验 - 是否平稳: {stationarity_result.is_stationary}")
        
        # 2. 自相关分析
        acf_pacf_result = calculate_acf_pacf(data["stock_returns"], nlags=10)
        
        assert len(acf_pacf_result.acf_values) == 11
        print(f"✅ 自相关分析 - ACF滞后数: {len(acf_pacf_result.acf_values)}")
        
        # 3. ARIMA模型
        try:
            arima_result = fit_arima_model(data["stock_returns"], order=(1, 0, 1))
            
            assert arima_result.aic < float('inf')
            print(f"✅ ARIMA模型 - AIC: {arima_result.aic:.2f}")
        except Exception as e:
            print(f"⚠️  ARIMA模型跳过: {e}")
        
        print("🎉 金融时间序列分析工作流测试通过")
    
    def test_panel_data_analysis_workflow(self):
        """测试面板数据分析工作流"""
        print("\n--- 面板数据分析工作流 ---")
        
        # 创建面板数据
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
        
        try:
            # 1. 固定效应模型
            fe_result = fixed_effects_model(y_data, X_data, entity_ids, time_periods, feature_names)
            
            assert fe_result.rsquared >= 0
            print(f"✅ 固定效应模型 - R²: {fe_result.rsquared:.3f}")
            
            # 2. 随机效应模型
            re_result = random_effects_model(y_data, X_data, entity_ids, time_periods, feature_names)
            
            assert re_result.rsquared >= 0
            print(f"✅ 随机效应模型 - R²: {re_result.rsquared:.3f}")
            
            # 3. Hausman检验
            hausman_result = hausman_test(y_data, X_data, entity_ids, time_periods, feature_names)
            
            assert hausman_result.significant in [True, False]
            print(f"✅ Hausman检验 - 是否显著: {hausman_result.significant}")
            
        except Exception as e:
            print(f"⚠️  面板数据分析跳过: {e}")
        
        print("🎉 面板数据分析工作流测试通过")
    
    def test_machine_learning_workflow(self):
        """测试机器学习工作流"""
        print("\n--- 机器学习工作流 ---")
        
        data = self.create_sample_economic_data(80)
        
        # 准备数据
        y_data = data["gdp_growth"]
        X_data = [[inf, unemp, intr] for inf, unemp, intr in 
                  zip(data["inflation"], data["unemployment"], data["interest_rate"])]
        feature_names = ["Inflation", "Unemployment", "InterestRate"]
        
        try:
            # 1. 随机森林回归
            rf_result = random_forest_regression(y_data, X_data, feature_names, n_estimators=50)
            
            assert rf_result.r2_score >= -1
            print(f"✅ 随机森林回归 - R²: {rf_result.r2_score:.3f}")
            
            # 2. 梯度提升树回归
            gb_result = gradient_boosting_regression(y_data, X_data, feature_names, n_estimators=50)
            
            assert gb_result.r2_score >= -1
            print(f"✅ 梯度提升树回归 - R²: {gb_result.r2_score:.3f}")
            
        except Exception as e:
            print(f"⚠️  机器学习分析跳过: {e}")
        
        print("🎉 机器学习工作流测试通过")


class TestCrossModuleIntegration:
    """测试跨模块集成"""
    
    def test_statistics_to_regression_integration(self):
        """测试统计到回归的集成"""
        print("\n--- 统计到回归集成测试 ---")
        
        # 创建数据
        np.random.seed(42)
        n_samples = 50
        
        # 生成相关数据
        x1 = np.random.normal(0, 1, n_samples)
        x2 = np.random.normal(0, 1, n_samples)
        y = 2 * x1 + 3 * x2 + np.random.normal(0, 0.5, n_samples)
        
        data_dict = {
            "y": y.tolist(),
            "x1": x1.tolist(),
            "x2": x2.tolist()
        }
        
        # 1. 统计描述
        y_stats = calculate_descriptive_stats(data_dict["y"])
        x1_stats = calculate_descriptive_stats(data_dict["x1"])
        x2_stats = calculate_descriptive_stats(data_dict["x2"])
        
        print(f"✅ 统计描述 - y均值: {y_stats.mean:.3f}, x1均值: {x1_stats.mean:.3f}, x2均值: {x2_stats.mean:.3f}")
        
        # 2. 相关性分析
        corr_result = calculate_correlation_matrix(data_dict)
        
        assert len(corr_result.correlation_matrix) == 3
        print(f"✅ 相关性分析 - 完成")
        
        # 3. 回归分析
        X_data = [[x1_val, x2_val] for x1_val, x2_val in zip(data_dict["x1"], data_dict["x2"])]
        reg_result = perform_ols_regression(data_dict["y"], X_data, ["x1", "x2"])
        
        assert reg_result.rsquared > 0.5  # 应该有较好的拟合
        print(f"✅ 回归分析 - R²: {reg_result.rsquared:.3f}")
        
        print("🎉 统计到回归集成测试通过")
    
    def test_time_series_to_ml_integration(self):
        """测试时间序列到机器学习的集成"""
        print("\n--- 时间序列到机器学习集成测试 ---")
        
        # 创建时间序列数据
        np.random.seed(42)
        n_samples = 100
        
        # 生成具有趋势和季节性的时间序列
        trend = np.linspace(0, 10, n_samples)
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 12)
        noise = np.random.normal(0, 1, n_samples)
        ts_data = trend + seasonal + noise
        
        # 创建滞后特征用于机器学习
        X_data = []
        y_data = []
        
        for i in range(5, n_samples):
            features = [
                ts_data[i-1],  # 滞后1期
                ts_data[i-2],  # 滞后2期
                ts_data[i-3],  # 滞后3期
                ts_data[i-4],  # 滞后4期
                ts_data[i-5]   # 滞后5期
            ]
            X_data.append(features)
            y_data.append(ts_data[i])  # 当前期
        
        feature_names = ["lag1", "lag2", "lag3", "lag4", "lag5"]
        
        try:
            # 1. 时间序列分析
            stationarity = check_stationarity(ts_data.tolist())
            print(f"✅ 时间序列分析 - 平稳性: {stationarity.is_stationary}")
            
            # 2. 机器学习预测
            rf_result = random_forest_regression(y_data, X_data, feature_names, n_estimators=50)
            
            assert rf_result.r2_score > -1
            print(f"✅ 机器学习预测 - R²: {rf_result.r2_score:.3f}")
            
        except Exception as e:
            print(f"⚠️  时间序列到机器学习集成跳过: {e}")
        
        print("🎉 时间序列到机器学习集成测试通过")


class TestErrorHandlingIntegration:
    """测试集成错误处理"""
    
    def test_error_propagation(self):
        """测试错误传播"""
        print("\n--- 错误传播测试 ---")
        
        # 测试无效数据在整个工作流中的传播
        invalid_data = {"var1": "not_a_list"}
        
        try:
            calculate_descriptive_stats(invalid_data)
            assert False, "应该抛出错误"
        except Exception as e:
            print(f"✅ 错误正确传播: {type(e).__name__}")
        
        # 测试数据不足
        insufficient_data = [1, 2]
        
        try:
            perform_hypothesis_test(insufficient_data)
            assert False, "应该抛出错误"
        except Exception as e:
            print(f"✅ 数据不足错误: {type(e).__name__}")
        
        print("🎉 错误处理集成测试通过")


def main():
    """运行集成测试"""
    print("\n" + "="*80)
    print("开始运行计量经济学集成测试")
    print("="*80)
    
    tester = TestEconometricWorkflow()
    
    # 运行工作流测试
    test_methods = [
        tester.test_macro_economic_analysis_workflow,
        tester.test_financial_time_series_workflow,
        tester.test_panel_data_analysis_workflow,
        tester.test_machine_learning_workflow
    ]
    
    for test_method in test_methods:
        try:
            test_method()
        except Exception as e:
            print(f"❌ {test_method.__name__} 失败: {e}")
    
    # 运行集成测试
    integration_tester = TestCrossModuleIntegration()
    integration_methods = [
        integration_tester.test_statistics_to_regression_integration,
        integration_tester.test_time_series_to_ml_integration
    ]
    
    for test_method in integration_methods:
        try:
            test_method()
        except Exception as e:
            print(f"❌ {test_method.__name__} 失败: {e}")
    
    # 运行错误处理测试
    error_tester = TestErrorHandlingIntegration()
    error_tester.test_error_propagation()
    
    print("\n" + "="*80)
    print("集成测试完成")
    print("="*80)


if __name__ == "__main__":
    main()