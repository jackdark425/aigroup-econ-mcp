"""
AIGroup 计量经济学 MCP 工具基本使用示例
展示如何使用各种计量经济学工具进行数据分析
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Any

# 模拟MCP客户端调用（实际使用时会通过MCP协议调用）
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

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
    forecast_arima
)


async def demo_descriptive_statistics():
    """描述性统计分析示例"""
    print("📊 描述性统计分析示例")
    print("=" * 50)

    # 模拟经济数据
    economic_data = {
        "GDP": [100, 110, 120, 115, 125, 130, 128, 135, 140, 145],
        "Inflation": [2.1, 2.3, 1.9, 2.4, 2.0, 2.2, 1.8, 2.5, 2.1, 2.3],
        "Unemployment": [4.5, 4.2, 4.0, 4.3, 4.1, 3.9, 4.0, 3.8, 4.2, 4.0]
    }

    # 计算描述性统计
    for var_name, data in economic_data.items():
        stats = calculate_descriptive_stats(data)
        print(f"\n{var_name} 的描述性统计:")
        print(f"  均值: {stats.mean:.4f}")
        print(f"  中位数: {stats.median:.4f}")
        print(f"  标准差: {stats.std:.4f}")
        print(f"  最小值: {stats.min:.4f}")
        print(f"  最大值: {stats.max:.4f}")
        print(f"  偏度: {stats.skewness:.4f}")
        print(f"  峰度: {stats.kurtosis:.4f}")

    # 相关性分析
    print("\n相关性分析:")
    corr_result = calculate_correlation_matrix(economic_data)
    for var1 in corr_result.correlation_matrix:
        for var2 in corr_result.correlation_matrix[var1]:
            if var1 != var2:
                corr = corr_result.correlation_matrix[var1][var2]
                print(f"  {var1} vs {var2}: {corr:.4f}")


async def demo_regression_analysis():
    """回归分析示例"""
    print("\n📈 回归分析示例")
    print("=" * 50)

    # 模拟数据：分析广告支出和价格对销售额的影响
    sales = [120, 135, 118, 142, 155, 160, 148, 175, 180, 185]
    advertising = [8, 9, 7.5, 10, 11, 12, 10.5, 13, 14, 15]
    price = [100, 98, 102, 97, 95, 94, 96, 93, 92, 91]
    seasonality = [1.0, 1.1, 0.9, 1.2, 1.3, 1.4, 1.1, 1.5, 1.6, 1.7]

    X_data = list(zip(advertising, price, seasonality))
    feature_names = ["advertising", "price", "seasonality"]

    # OLS回归
    print("\n1. OLS回归分析:")
    ols_result = perform_ols_regression(sales, X_data, feature_names)

    print(f"  R² = {ols_result.rsquared:.4f}")
    print(f"  调整R² = {ols_result.rsquared_adj:.4f}")
    print(f"  F统计量 = {ols_result.f_statistic:.4f} (p值 = {ols_result.f_pvalue:.4f})")
    print(f"  AIC = {ols_result.aic:.2f}, BIC = {ols_result.bic:.2f}")

    print("\n  回归系数:")
    for var, coef_info in ols_result.coefficients.items():
        print(f"    {var}: {coef_info['coef']:.4f} (p值 = {coef_info['p_value']:.4f})")

    # 逐步回归
    print("\n2. 逐步回归:")
    stepwise_result = stepwise_regression(sales, X_data, feature_names)
    print(f"  选择的变量: {stepwise_result['selected_features']}")
    print(f"  最终模型R² = {stepwise_result['model_summary']['rsquared']:.4f}")
    print(f"  最终模型AIC = {stepwise_result['model_summary']['aic']:.2f}")

    # 模型诊断
    print("\n3. 模型诊断:")
    diagnostics = run_diagnostic_tests(sales, X_data)
    print(f"  Jarque-Bera正态性检验: 统计量={diagnostics.jb_statistic:.4f}, p值={diagnostics.jb_pvalue:.4f}")
    print(f"  Breusch-Pagan异方差检验: 统计量={diagnostics.bp_statistic:.4f}, p值={diagnostics.bp_pvalue:.4f}")
    print(f"  Durbin-Watson序列相关检验: 统计量={diagnostics.dw_statistic:.4f}")
    print(f"  VIF值: {diagnostics.vif}")


async def demo_hypothesis_testing():
    """假设检验示例"""
    print("\n🧪 假设检验示例")
    print("=" * 50)

    # 生成两组样本数据
    np.random.seed(42)
    group1 = np.random.normal(100, 10, 50)  # 均值100，标准差10
    group2 = np.random.normal(105, 12, 50)  # 均值105，标准差12

    # 双样本t检验
    print("\n1. 双样本t检验:")
    t_result = perform_hypothesis_test(group1.tolist(), group2.tolist(), "t_test")
    print(f"  t统计量 = {t_result['statistic']:.4f}")
    print(f"  p值 = {t_result['p_value']:.4f}")
    print(f"  显著性 = {'是' if t_result['significant'] else '否'}")

    # 单样本t检验
    print("\n2. 单样本t检验 (检验均值是否等于100):")
    t1_result = perform_hypothesis_test(group1.tolist(), test_type="t_test")
    print(f"  t统计量 = {t1_result['statistic']:.4f}")
    print(f"  p值 = {t1_result['p_value']:.4f}")
    print(f"  显著性 = {'是' if t1_result['significant'] else '否'}")

    # 正态性检验
    print("\n3. 正态性检验:")
    normal_result = normality_test(group1.tolist())
    print(f"  Shapiro-Wilk检验: 统计量={normal_result['shapiro_wilk']['statistic']:.4f}, p值={normal_result['shapiro_wilk']['p_value']:.4f}")
    print(f"  正态性 = {'是' if normal_result['shapiro_wilk']['normal'] else '否'}")


async def demo_time_series_analysis():
    """时间序列分析示例"""
    print("\n⏰ 时间序列分析示例")
    print("=" * 50)

    # 模拟时间序列数据
    np.random.seed(42)
    # 模拟一个有趋势和季节性的时间序列
    n = 100
    trend = np.linspace(100, 200, n)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = np.random.normal(0, 5, n)
    ts_data = trend + seasonal + noise

    # 平稳性检验
    print("\n1. 平稳性检验:")
    stationarity = check_stationarity(ts_data.tolist())
    print(f"  ADF检验: 统计量={stationarity.adf_statistic:.4f}, p值={stationarity.adf_pvalue:.4f}")
    print(f"  KPSS检验: 统计量={stationarity.kpss_statistic:.4f}, p值={stationarity.kpss_pvalue:.4f}")
    print(f"  是否平稳: {'是' if stationarity.is_stationary else '否'}")

    # 自相关分析
    print("\n2. 自相关分析:")
    acf_pacf = calculate_acf_pacf(ts_data.tolist(), nlags=12)
    print(f"  ACF前6阶: {[f'{x:.4f}' for x in acf_pacf.acf_values[:6]]}")
    print(f"  PACF前6阶: {[f'{x:.4f}' for x in acf_pacf.pacf_values[:6]]}")

    # 差分后重新检验
    print("\n3. 一阶差分后的平稳性:")
    diff_data = np.diff(ts_data)
    stationarity_diff = check_stationarity(diff_data.tolist())
    print(f"  ADF检验: 统计量={stationarity_diff.adf_statistic:.4f}, p值={stationarity_diff.adf_pvalue:.4f}")
    print(f"  是否平稳: {'是' if stationarity_diff.is_stationary else '否'}")

    # ARIMA模型拟合
    print("\n4. ARIMA模型拟合:")
    try:
        # 寻找最佳模型
        best_model = find_best_arima_order(diff_data.tolist(), max_p=3, max_d=1, max_q=3)
        print(f"  最佳模型阶数: {best_model['best_order']}")
        print(f"  AIC: {best_model['best_aic']:.2f}")
        print(f"  BIC: {best_model['best_bic']:.2f}")

        # 预测
        forecast_result = forecast_arima(diff_data.tolist(), best_model['best_order'], steps=5)
        print(f"  5步预测: {[f'{x:.2f}' for x in forecast_result['forecast']]}")

    except Exception as e:
        print(f"  ARIMA模型拟合失败: {e}")


async def demo_structured_output():
    """结构化输出示例"""
    print("\n🔄 结构化输出示例")
    print("=" * 50)

    # 展示如何处理结构化输出
    economic_data = {
        "GDP": [100, 110, 120, 115, 125],
        "Investment": [15, 16, 17, 16.5, 18],
        "Consumption": [70, 75, 80, 78, 85]
    }

    # 计算相关性
    corr_result = calculate_correlation_matrix(economic_data)

    print("\n结构化输出示例:")
    print("相关性分析结果结构:")
    print(json.dumps(corr_result.model_dump(), indent=2, ensure_ascii=False))


async def main():
    """主演示函数"""
    print("🎯 AIGroup 计量经济学 MCP 工具演示")
    print("展示各种计量经济学分析功能")
    print("=" * 60)

    await demo_descriptive_statistics()
    await demo_regression_analysis()
    await demo_hypothesis_testing()
    await demo_time_series_analysis()
    await demo_structured_output()

    print("\n✅ 演示完成！")
    print("这些功能都可以在MCP服务器中通过工具调用使用。")


if __name__ == "__main__":
    asyncio.run(main())