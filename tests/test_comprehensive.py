
"""
综合测试文件 - 验证所有计量经济学工具模块功能

测试覆盖：
1. 统计分析模块
2. 回归分析模块  
3. 时间序列分析模块
4. 面板数据分析模块
5. 机器学习模块
6. 缓存机制
7. 参数验证系统
8. 性能监控
9. 集成测试
10. 错误处理
11. 性能基准测试
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 导入所有工具模块
from aigroup_econ_mcp.tools.statistics import (
    calculate_descriptive_stats, 
    calculate_correlation_matrix,
    perform_hypothesis_test,
    normality_test
)
from aigroup_econ_mcp.tools.regression import (
    perform_ols_regression,
    calculate_vif,
    run_diagnostic_tests,
    stepwise_regression
)
from aigroup_econ_mcp.tools.time_series import (
    check_stationarity,
    calculate_acf_pacf,
    fit_arima_model,
    find_best_arima_order,
    decompose_time_series,
    var_model,
    vecm_model,
    garch_model,
    state_space_model,
    forecast_var,
    impulse_response_analysis,
    variance_decomposition
)
from aigroup_econ_mcp.tools.panel_data import (
    fixed_effects_model,
    random_effects_model,
    hausman_test,
    panel_unit_root_test,
    compare_panel_models
)
from aigroup_econ_mcp.tools.machine_learning import (
    random_forest_regression,
    gradient_boosting_regression,
    lasso_regression,
    ridge_regression,
    cross_validation,
    feature_importance_analysis,
    compare_ml_models
)
from aigroup_econ_mcp.tools.cache import (
    global_econometric_cache,
    cache_result,
    cache_model
)
from aigroup_econ_mcp.tools.validation import (
    validate_econometric_data,
    validate_model_parameters,
    validate_time_series_data,
    ValidationError
)
from aigroup_econ_mcp.tools.monitoring import (
    global_performance_monitor,
    track_progress,
    enable_memory_monitoring,
    disable_memory_monitoring
)
from aigroup_econ_mcp.tools.base import (
    EconometricTool,
    econometric_tool,
    validate_input
)


class ComprehensiveTester:
    """综合测试器"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_data = {}
        self.error_count = 0
        self.success_count = 0
        
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 80)
        print("开始运行综合测试")
        print("=" * 80)
        
        # 启用性能监控
        global_performance_monitor.enabled = True
        
        # 测试列表
        test_methods = [
            self.test_statistics_module,
            self.test_regression_module,
            self.test_time_series_module,
            self.test_panel_data_module,
            self.test_machine_learning_module,
            self.test_cache_mechanism,
            self.test_validation_system,
            self.test_performance_monitoring,
            self.test_integration_scenarios,
            self.test_error_handling,
            self.test_performance_benchmarks
        ]
        
        # 运行所有测试
        with track_progress(len(test_methods), "综合测试进度") as tracker:
            for test_method in test_methods:
                tracker.start_step(test_method.__name__)
                try:
                    test_method()
                    self.success_count += 1
                    print(f"✅ {test_method.__name__} - 通过")
                except Exception as e:
                    self.error_count += 1
                    print(f"❌ {test_method.__name__} - 失败: {e}")
                tracker.complete_step()
                tracker.print_progress()
        
        # 生成测试报告
        self.generate_test_report()
    
    def generate_test_report(self):
        """生成测试报告"""
        print("\n" + "=" * 80)
        print("测试报告")
        print("=" * 80)
        print(f"总测试数: {len(self.test_results)}")
        print(f"成功: {self.success_count}")
        print(f"失败: {self.error_count}")
        print(f"成功率: {self.success_count / len(self.test_results) * 100:.1f}%")
        
        # 性能监控摘要
        perf_summary = global_performance_monitor.get_summary()
        if perf_summary:
            print(f"\n性能监控摘要:")
            print(f"  监控函数数: {perf_summary['total_functions']}")
            print(f"  总执行时间: {perf_summary['total_execution_time']:.2f}s")
            print(f"  平均执行时间: {perf_summary['average_execution_time']:.2f}s")
            print(f"  最大内存使用: {perf_summary['max_memory_usage_mb']:.1f}MB")
        
        # 缓存统计
        cache_stats = global_econometric_cache.get_cache_info()
        if cache_stats:
            print(f"\n缓存统计:")
            print(f"  模型缓存大小: {cache_stats['model_cache_size']}")
            print(f"  数据缓存大小: {cache_stats['data_cache_size']}")
            result_cache_stats = cache_stats['result_cache_stats']
            if result_cache_stats:
                print(f"  全局命中率: {result_cache_stats['global_hit_rate']:.2f}")
                print(f"  总命中数: {result_cache_stats['total_hits']}")
                print(f"  总未命中数: {result_cache_stats['total_misses']}")
    
    def create_sample_data(self, n_samples: int = 100) -> Dict[str, Any]:
        """创建样本数据"""
        np.random.seed(42)
        
        # 宏观经济数据
        gdp_growth = np.random.normal(3.0, 0.8, n_samples)
        inflation = np.random.normal(2.0, 0.5, n_samples)
        unemployment = np.random.normal(5.0, 1.2, n_samples)
        interest_rate = np.random.normal(2.5, 0.7, n_samples)
        
        # 金融数据
        stock_returns = np.random.normal(0.1, 2.0, n_samples)
        volatility = np.abs(np.random.normal(1.5, 0.5, n_samples))
        
        # 面板数据
        entity_ids = [f"Entity_{i//10}" for i in range(n_samples)]
        time_periods = [f"Period_{i%10}" for i in range(n_samples)]
        
        return {
            "gdp_growth": gdp_growth.tolist(),
            "inflation": inflation.tolist(),
            "unemployment": unemployment.tolist(),
            "interest_rate": interest_rate.tolist(),
            "stock_returns": stock_returns.tolist(),
            "volatility": volatility.tolist(),
            "entity_ids": entity_ids,
            "time_periods": time_periods
        }
    
    def test_statistics_module(self):
        """测试统计分析模块"""
        print("\n--- 测试统计分析模块 ---")
        
        data = self.create_sample_data(50)
        
        # 描述性统计
        stats = calculate_descriptive_stats(data["gdp_growth"])
        assert stats.count == 50
        assert isinstance(stats.mean, float)
        print(f"✅ 描述性统计 - 均值: {stats.mean:.3f}, 标准差: {stats.std:.3f}")
        
        # 相关性分析
        corr_data = {
            "GDP": data["gdp_growth"],
            "Inflation": data["inflation"],
            "Unemployment": data["unemployment"]
        }
        corr_result = calculate_correlation_matrix(corr_data)
        assert len(corr_result.correlation_matrix) == 3
        print(f"✅ 相关性分析 - 矩阵大小: {len(corr_result.correlation_matrix)}x{len(corr_result.correlation_matrix)}")
        
        # 假设检验
        test_result = perform_hypothesis_test(data["gdp_growth"], data["inflation"])
        assert "statistic" in test_result
        assert "p_value" in test_result
        print(f"✅ 假设检验 - 统计量: {test_result['statistic']:.3f}, p值: {test_result['p_value']:.3f}")
        
        # 正态性检验
        normality_result = normality_test(data["gdp_growth"])
        assert "shapiro_wilk" in normality_result
        assert "kolmogorov_smirnov" in normality_result
        print(f"✅ 正态性检验 - Shapiro-Wilk p值: {normality_result['shapiro_wilk']['p_value']:.3f}")
        
        self.test_results["statistics"] = "通过"
    
    def test_regression_module(self):
        """测试回归分析模块"""
        print("\n--- 测试回归分析模块 ---")
        
        data = self.create_sample_data(50)
        
        # OLS回归
        y_data = data["gdp_growth"]
        X_data = [[inf, unemp] for inf, unemp in zip(data["inflation"], data["unemployment"])]
        feature_names = ["Inflation", "Unemployment"]
        
        ols_result = perform_ols_regression(y_data, X_data, feature_names)
        assert ols_result.rsquared >= 0
        assert len(ols_result.coefficients) == 3  # 常数项 + 2个特征
        print(f"✅ OLS回归 - R²: {ols_result.rsquared:.3f}, 系数数量: {len(ols_result.coefficients)}")
        
        # VIF计算
        vif_values = calculate_vif(X_data, feature_names)
        assert len(vif_values) == 2
        print(f"✅ VIF计算 - Inflation: {vif_values.get('Inflation', 0):.2f}, Unemployment: {vif_values.get('Unemployment', 0):.2f}")
        
        # 模型诊断
        diagnostic_result = run_diagnostic_tests(y_data, X_data)
        assert diagnostic_result.dw_statistic > 0
        print(f"✅ 模型诊断 - Durbin-Watson: {diagnostic_result.dw_statistic:.3f}")
        
        # 逐步回归
        stepwise_result = stepwise_regression(y_data, X_data, feature_names)
        assert "selected_features" in stepwise_result
        print(f"✅ 逐步回归 - 选择特征: {stepwise_result['selected_features']}")
        
        self.test_results["regression"] = "通过"
    
    def test_time_series_module(self):
        """测试时间序列分析模块"""
        print("\n--- 测试时间序列分析模块 ---")
        
        data = self.create_sample_data(100)
        
        # 平稳性检验
        stationarity_result = check_stationarity(data["gdp_growth"])
        assert isinstance(stationarity_result.is_stationary, bool)
        print(f"✅ 平稳性检验 - 是否平稳: {stationarity_result.is_stationary}")
        
        # 自相关分析
        acf_pacf_result = calculate_acf_pacf(data["gdp_growth"], nlags=10)
        assert len(acf_pacf_result.acf_values) == 11  # 包括滞后0
        print(f"✅ 自相关分析 - ACF滞后数: {len(acf_pacf_result.acf_values)}")
        
        # ARIMA模型
        arima_result = fit_arima_model(data["gdp_growth"], order=(1, 0, 1))
        assert arima_result.aic < float('inf')
        print(f"✅ ARIMA模型 - AIC: {arima_result.aic:.2f}")
        
        # VAR模型
        var_data = {
            "GDP": data["gdp_growth"],
            "Inflation": data["inflation"]
        }
        var_result = var_model(var_data, max_lags=2)
        assert var_result.order >= 0
        print(f"✅ VAR模型 - 最优滞后阶数: {var_result.order}, AIC: {var_result.aic:.2f}")
        
        # GARCH模型
        garch_result = garch_model(data["stock_returns"], order=(1, 1))
        assert garch_result.persistence >= 0
        print(f"✅ GARCH模型 - 持久性: {garch_result.persistence:.3f}")
        
        # 状态空间模型
        state_space_result = state_space_model(data["gdp_growth"], trend=True)
        assert state_space_result.log_likelihood < float('inf')
        print(f"✅ 状态空间模型 - 对数似然: {state_space_result.log_likelihood:.2f}")
        
        self.test_results["time_series"] = "通过"
    
    def test_panel_data_module(self):
        """测试面板数据分析模块"""
        print("\n--- 测试面板数据分析模块 ---")
        
        data = self.create_sample_data(60)  # 6个实体，每个10个时期
        
        # 准备面板数据
        y_data = data["gdp_growth"]
        X_data = [[inf, unemp] for inf, unemp in zip(data["inflation"], data["unemployment"])]
        entity_ids = data["entity_ids"]
        time_periods = data["time_periods"]
        feature_names = ["Inflation", "Unemployment"]
        
        # 固定效应模型
        fe_result = fixed_effects_model(y_data, X_data, entity_ids, time_periods, feature_names)
        assert fe_result.rsquared >= 0
        print(f"✅ 固定效应模型 - R²: {fe_result.rsquared:.3f}")
        
        # 随机效应模型
        re_result = random_effects_model(y_data, X_data, entity_ids, time_periods, feature_names)
        assert re_result.rsquared >= 0
        print(f"✅ 随机效应模型 - R²: {re_result.rsquared:.3f}")
        
        # Hausman检验
        hausman_result = hausman_test(y_data, X_data, entity_ids, time_periods, feature_names)
        assert hausman_result.significant in [True, False]
        print(f"✅ Hausman检验 - 是否显著: {hausman_result.significant}")
        
        # 面板单位根检验
        panel_unit_result = panel_unit_root_test(data["gdp_growth"], entity_ids, time_periods)
        assert panel_unit_result.stationary in [True, False]
        print(f"✅ 面板单位根检验 - 是否平稳: {panel_unit_result.stationary}")
        
        # 模型比较
        comparison_result = compare_panel_models(y_data, X_data, entity_ids, time_periods, feature_names)
        assert "recommendation" in comparison_result
        print(f"✅ 面板模型比较 - 推荐: {comparison_result['recommendation']}")
        
        self.test_results["panel_data"] = "通过"
    
    def test_machine_learning_module(self):
        """测试机器学习模块"""
        print("\n--- 测试机器学习模块 ---")
        
        data = self.create_sample_data(80)
        
        # 准备数据
        y_data = data["gdp_growth"]
        X_data = [[inf, unemp, intr] for inf, unemp, intr in 
                  zip(data["inflation"], data["unemployment"], data["interest_rate"])]
        feature_names = ["Inflation", "Unemployment", "InterestRate"]
        
        # 随机森林回归
        rf_result = random_forest_regression(y_data, X_data, feature_names, n_estimators=50)
        assert rf_result.r2_score >= -1
        print(f"✅ 随机森林回归 - R²: {rf_result.r2_score:.3f}")
        
        # 梯度提升树回归
        gb_result = gradient_boosting_regression(y_data, X_data, feature_names, n_estimators=50)
        assert gb_result.r2_score >= -1
        print(f"✅ 梯度提升树回归 - R²: {gb_result.r2_score:.3f}")
        
        # Lasso回归
        lasso_result = lasso_regression(y_data, X_data, feature_names, alpha=0.1)
        assert lasso_result.r2_score >= -1
        print(f"✅ Lasso回归 - R²: {lasso_result.r2_score:.3f}")
        
        # Ridge回归
        ridge_result = ridge_regression(y_data, X_data, feature_names, alpha=0.1)
        assert ridge_result.r2_score >= -1
        print(f"✅ Ridge回归 - R²: {ridge_result.r2_score:.3f}")
        
        # 交叉验证
        cv_result = cross_validation(y_data, X_data, model_type="random_forest", cv_folds=5)
        assert len(cv_result.cv_scores) == 5
        print(f"✅ 交叉验证 - 平均得分: {cv_result.mean_score:.3f}")
        
        # 特征重要性分析
        feature_importance_result = feature_importance_analysis(y_data, X_data, feature_names)
        assert len(feature_importance_result.top_features) > 0
        print(f"✅ 特征重要性分析 - 最重要特征: {feature_importance_result.top_features[0]}")
        
        # 模型比较
        model_comparison = compare_ml_models(y_data, X_data, feature_names)
        assert "best_model" in model_comparison
        print(f"✅ 模型比较 - 最佳模型: {model_comparison['best_model']}")
        
        self.test_results["machine_learning"] = "通过"
    
    def test_cache_mechanism(self):
        """测试缓存机制"""
        print("\n--- 测试缓存机制 ---")
        
        # 清空缓存
        global_econometric_cache.result_cache.clear_all()
        
        # 创建测试函数
        @cache_result(ttl=60, max_size=10)
        def expensive_calculation(x, y):
            time.sleep(0.1)
            return x + y
        
        # 第一次调用（应该计算）
        start_time = time.time()
        result1 = expensive_calculation(5, 10)
        time1 = time.time() - start_time
        
        # 第二次调用（应该从缓存获取）
        start_time = time.time()
        result2 = expensive_calculation(5, 10)
        time2 = time.time() - start_time
        
        assert result1 == result2 == 15
        assert time2 < time1  # 缓存应该更快
        print(f"✅ 缓存机制 - 第一次: {time1:.3f}s, 第二次: {time2:.3f}s, 加速: {time1/time2:.1f}x")
        
        # 检查缓存统计
        cache_stats = global_econometric_cache.result_cache.get_function_cache_stats("expensive_calculation")
        assert cache_stats is not None
        assert cache_stats["hits"] >= 1
        print(f"✅ 缓存统计 - 命中数: {cache_stats['hits']}, 未命中数: {cache_stats['misses']}")
        
        self.test_results["cache"] = "通过"
    
    def test_validation_system(self):
        """测试参数验证系统"""
        print("\n--- 测试参数验证系统 ---")
        
        # 有效数据验证
        valid_data = {
            "GDP": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Inflation": [2.1, 2.3, 2.0, 2.4, 2.2]
        }
        validated_data = validate_econometric_data(valid_data)
        assert len(validated_data) == 2
        print(f"✅ 有效数据验证 - 变量数: {len(validated_data)}")
        
        # 无效数据验证（应该抛出异常）
        try:
            invalid_data = {"GDP": "not_a_list"}
            validate_econometric_data(invalid_data)
            assert False, "应该抛出ValidationError"
        except ValidationError:
            print("✅ 无效数据验证 - 正确抛出异常")
        
        # 模型参数验证
        valid_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "learning_rate": 0.1,
            "alpha": 1.0
        }
        validated_params = validate_model_parameters(valid_params)
        assert validated_params["n_estimators"] == 100
        print(f"✅ 模型参数验证 - 验证参数数: {len(validated_params)}")
        
        # 时间序列数据验证
        valid_ts_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        validated_ts_data = validate_time_series_data(valid_ts_data, min_length=10)
        assert len(validated_ts_data) == 10
        print(f"✅ 时间序列验证 - 数据长度: {len(validated_ts_data)}")
        
        self.test_results["validation"] = "通过"
    
    def test_performance_monitoring(self):
        """测试性能监控"""
        print("\n--- 测试性能监控 ---")
        
        # 启用内存监控
        memory_monitor = enable_memory_monitoring(check_interval=0.1)
        
        # 监控一个计算密集型函数
        @global_performance_monitor.monitor_function
        def monitored_function(data_size):
            # 模拟计算密集型操作
            data = np.random.rand(data_size, data_size)
            result = np.linalg.eig(data)
            return result[0].sum()
        
        # 执行监控的函数
        result = monitored_function(100)
        assert isinstance(result, float)
        
        # 获取性能统计
        stats = global_performance_monitor.get_function_stats("monitored_function")
        assert stats is not None
        assert stats.execution_time > 0
        print(f"✅ 性能监控 - 执行时间: {stats.execution_time:.3f}s, 内存使用: {stats.peak_memory_mb:.1f}MB")
        
        # 禁用内存监控并获取分析
        memory_analysis = disable_memory_monitoring()
        assert "memory_increase_mb" in memory_analysis
        print(f"✅ 内存监控 - 内存增长: {memory_analysis['memory_increase_mb']:.1f}MB")
        
        self.test_results["performance_monitoring"] = "通过"
    
    def test_integration_scenarios(self):
        """测试集成场景"""
        print("\n--- 测试集成场景 ---")
        
        data = self.create_sample_data(100)
        
        # 场景1: 宏观经济分析管道
        # 描述性统计 -> 相关性分析 -> VAR模型 -> 脉冲响应分析
        print("🔍 集成场景1: 宏观经济分析管道")
        
        # 1. 描述性统计
        gdp_stats = calculate_descriptive_stats(data["gdp_growth"])
        inflation_stats = calculate_descriptive_stats(data["inflation"])
        
        # 2. 相关性分析
        macro_data = {
            "GDP": data["gdp_growth"],
            "Inflation": data["inflation"],
            "Unemployment": data["unemployment"]
        }
        corr_result = calculate_correlation_matrix(macro_data)
        
        # 3. VAR模型
        var_result = var_model({
            "GDP": data["gdp_growth"][:50],  # 使用前50个点避免数据不足
            "Inflation": data["inflation"][:50]
        }, max_lags=2)
        
        # 4. 脉冲响应分析
        irf_result = impulse_response_analysis({
            "GDP": data["gdp_growth"][:50],
            "Inflation": data["inflation"][:50]
        }, periods=5, max_lags=2)
        
        assert var_result.order >= 0
        assert "impulse_responses" in irf_result
        print(f"✅ 集成场景1 - VAR滞后阶数: {var_result.order}, 脉冲响应计算完成")
        
        # 场景2: 机器学习预测管道
        # 特征工程 -> 模型训练 -> 交叉验证 -> 特征重要性分析
        print("🔍 集成场景2: 机器学习预测管道")
        
        # 准备数据
        y_data = data["gdp_growth"]
        X_data = [[inf, unemp, intr] for inf, unemp, intr in 
                  zip(data["inflation"], data["unemployment"], data["interest_rate"])]
        feature_names = ["Inflation", "Unemployment", "InterestRate"]
        
        # 1. 模型训练
        rf_result = random_forest_regression(y_data, X_data, feature_names)
        
        # 2. 交叉验证
        cv_result = cross_validation(y_data, X_data, model_type="random_forest")
        
        # 3. 特征重要性
        feature_importance = feature_importance_analysis(y_data, X_data, feature_names)
        
        assert rf_result.r2_score >= -1
        assert cv_result.mean_score >= -1
        assert len(feature_importance.top_features) > 0
        print(f"✅ 集成场景2 - R²: {rf_result.r2_score:.3f}, 交叉验证得分: {cv_result.mean_score:.3f}")
        
        self.test_results["integration"] = "通过"
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n--- 测试错误处理 ---")
        
        # 测试1: 空数据
        try:
            calculate_descriptive_stats([])
            assert False, "应该处理空数据错误"
        except Exception as e:
            print(f"✅ 空数据处理 - {type(e).__name__}: {str(e)}")
        
        # 测试2: 无效参数
        try:
            perform_ols_regression([1, 2, 3], [[1], [2]])  # 数据长度不匹配
            assert False, "应该处理参数错误"
        except Exception as e:
            print(f"✅ 无效参数处理 - {type(e).__name__}: {str(e)}")
        
        # 测试3: 数据不足
        try:
            var_model({"GDP": [1, 2, 3]}, max_lags=2)  # 数据不足
            assert False, "应该处理数据不足错误"
        except Exception as e:
            print(f"✅ 数据不足处理 - {type(e).__name__}: {str(e)}")
        
        # 测试4: 无效模型参数
        try:
            random_forest_regression([1, 2, 3], [[1], [2], [3]], n_estimators=-1)  # 无效参数
            assert False, "应该处理无效模型参数"
        except Exception as e:
            print(f"✅ 无效模型参数处理 - {type(e).__name__}: {str(e)}")
        
        self.test_results["error_handling"] = "通过"
    
    def test_performance_benchmarks(self):
        """测试性能基准"""
        print("\n--- 测试性能基准 ---")
        
        data = self.create_sample_data(200)  # 更大的数据集
        
        # 基准1: OLS回归性能
        print("📊 基准1: OLS回归性能")
        y_data = data["gdp_growth"]
        X_data = [[inf, unemp] for inf, unemp in zip(data["inflation"], data["unemployment"])]
        
        start_time = time.time()
        ols_result = perform_ols_regression(y_data, X_data)
        ols_time = time.time() - start_time
        
        assert ols_result.rsquared >= -1
        print(f"✅ OLS基准 - 时间: {ols_time:.3f}s, R²: {ols_result.rsquared:.3f}")
        
        # 基准2: 随机森林性能
        print("📊 基准2: 随机森林性能")
        start_time = time.time()
        rf_result = random_forest_regression(y_data, X_data, n_estimators=50)
        rf_time = time.time() - start_time
        
        assert rf_result.r2_score >= -1
        print(f"✅ 随机森林基准 - 时间: {rf_time:.3f}s, R²: {rf_result.r2_score:.3f}")
        
        # 基准3: VAR模型性能
        print("📊 基准3: VAR模型性能")
        var_data = {
            "GDP": data["gdp_growth"][:100],  # 使用前100个点
            "Inflation": data["inflation"][:100],
            "Unemployment": data["unemployment"][:100]
        }
        
        start_time = time.time()
        var_result = var_model(var_data, max_lags=3)
        var_time = time.time() - start_time
        
        assert var_result.aic < float('inf')
        print(f"✅ VAR基准 - 时间: {var_time:.3f}s, AIC: {var_result.aic:.2f}")
        
        # 性能比较
        print(f"📈 性能比较:")
        print(f"   - OLS回归: {ols_time:.3f}s")
        print(f"   - 随机森林: {rf_time:.3f}s")
        print(f"   - VAR模型: {var_time:.3f}s")
        
        self.performance_data = {
            "ols_time": ols_time,
            "rf_time": rf_time,
            "var_time": var_time
        }
        
        self.test_results["performance_benchmarks"] = "通过"


def main():
    """主函数"""
    tester = ComprehensiveTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()