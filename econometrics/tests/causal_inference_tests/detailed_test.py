"""
详细测试所有因果识别策略方法
"""

import numpy as np
import pandas as pd
from econometrics.causal_inference.causal_identification_strategy import *


def test_instrumental_variables():
    """测试工具变量法"""
    print("测试工具变量法...")
    np.random.seed(42)
    n = 100
    
    # 生成数据
    z = np.random.normal(0, 1, n)  # 工具变量
    e1 = np.random.normal(0, 1, n)
    x = 1 + 0.5 * z + e1  # 内生变量
    e2 = np.random.normal(0, 1, n)
    y = 2 + 1.5 * x + e2 + 0.3 * e1  # 结果变量，包含内生性
    
    try:
        result = instrumental_variables_2sls(
            y=y.tolist(),
            x=x.reshape(-1, 1).tolist(),
            instruments=z.reshape(-1, 1).tolist()
        )
        print(f"  系数: {result.estimate:.4f}")
        print(f"  标准误: {result.std_error:.4f}")
        print(f"  p值: {result.p_value:.4f}")
        print("  ✓ 工具变量法测试通过\n")
    except Exception as e:
        print(f"  ✗ 工具变量法测试失败: {e}\n")


def test_control_function():
    """测试控制函数法"""
    print("测试控制函数法...")
    np.random.seed(42)
    n = 100
    
    # 生成数据
    z1 = np.random.normal(0, 1, n)
    z2 = np.random.normal(0, 1, n)
    e1 = np.random.normal(0, 1, n)
    x = 1 + 0.5 * z1 + 0.3 * z2 + e1  # 内生变量
    e2 = np.random.normal(0, 1, n)
    y = 2 + 1.5 * x + e2 + 0.3 * e1  # 结果变量，包含内生性
    
    try:
        result = control_function_approach(
            y=y.tolist(),
            x=x.tolist(),
            z=np.column_stack([z1, z2]).tolist()
        )
        print(f"  系数: {result.estimate:.4f}")
        print(f"  标准误: {result.std_error:.4f}")
        print(f"  p值: {result.p_value:.4f}")
        print("  ✓ 控制函数法测试通过\n")
    except Exception as e:
        print(f"  ✗ 控制函数法测试失败: {e}\n")


def test_fixed_effects():
    """测试固定效应模型"""
    print("测试固定效应模型...")
    np.random.seed(42)
    n_entities = 10
    n_periods = 5
    n = n_entities * n_periods
    
    # 生成面板数据
    entity_ids = [f"entity_{i}" for i in range(n_entities) for _ in range(n_periods)]
    time_periods = [f"period_{t}" for _ in range(n_entities) for t in range(n_periods)]
    x = np.random.normal(0, 1, (n, 2)).tolist()
    
    # 因变量（包含个体固定效应）
    entity_effects = np.random.normal(0, 1, n_entities)
    y = []
    for i in range(n):
        entity_idx = i // n_periods
        y_value = 1 + 2 * x[i][0] + 1.5 * x[i][1] + entity_effects[entity_idx] + np.random.normal(0, 0.5)
        y.append(y_value)
    
    try:
        result = fixed_effects_model(
            y=y,
            x=x,
            entity_ids=entity_ids,
            time_periods=time_periods
        )
        print(f"  系数: {result.estimate:.4f}")
        print(f"  标准误: {result.std_error:.4f}")
        print(f"  p值: {result.p_value:.4f}")
        print("  ✓ 固定效应模型测试通过\n")
    except Exception as e:
        print(f"  ✗ 固定效应模型测试失败: {e}\n")


def test_random_effects():
    """测试随机效应模型"""
    print("测试随机效应模型...")
    np.random.seed(42)
    n_entities = 10
    n_periods = 5
    n = n_entities * n_periods
    
    # 生成面板数据
    entity_ids = [f"entity_{i}" for i in range(n_entities) for _ in range(n_periods)]
    time_periods = [f"period_{t}" for _ in range(n_entities) for t in range(n_periods)]
    x = np.random.normal(0, 1, (n, 2)).tolist()
    
    # 因变量（包含个体随机效应）
    entity_effects = np.random.normal(0, 1, n_entities)
    y = []
    for i in range(n):
        entity_idx = i // n_periods
        y_value = 1 + 2 * x[i][0] + 1.5 * x[i][1] + entity_effects[entity_idx] + np.random.normal(0, 0.5)
        y.append(y_value)
    
    try:
        result = random_effects_model(
            y=y,
            x=x,
            entity_ids=entity_ids,
            time_periods=time_periods
        )
        print(f"  系数: {result.estimate:.4f}")
        print(f"  标准误: {result.std_error:.4f}")
        print(f"  p值: {result.p_value:.4f}")
        print("  ✓ 随机效应模型测试通过\n")
    except Exception as e:
        print(f"  ✗ 随机效应模型测试失败: {e}\n")


def test_first_difference():
    """测试一阶差分模型"""
    print("测试一阶差分模型...")
    np.random.seed(42)
    n_entities = 10
    n_periods = 5
    n = n_entities * n_periods
    
    # 生成面板数据
    entity_ids = [f"entity_{i}" for i in range(n_entities) for _ in range(n_periods)]
    x = np.cumsum(np.random.normal(0, 1, n))  # 随时间累积的变量
    y = 2 + 1.5 * x + np.random.normal(0, 1, n)  # 因变量
    
    try:
        result = first_difference_model(
            y=y.tolist(),
            x=x.tolist(),
            entity_ids=entity_ids
        )
        print(f"  系数: {result.estimate:.4f}")
        print(f"  标准误: {result.std_error:.4f}")
        print(f"  p值: {result.p_value:.4f}")
        print("  ✓ 一阶差分模型测试通过\n")
    except Exception as e:
        print(f"  ✗ 一阶差分模型测试失败: {e}\n")


def test_hausman_test():
    """测试Hausman检验"""
    print("测试Hausman检验...")
    np.random.seed(42)
    n_entities = 10
    n_periods = 5
    n = n_entities * n_periods
    
    # 生成面板数据
    entity_ids = [f"entity_{i}" for i in range(n_entities) for _ in range(n_periods)]
    time_periods = [f"period_{t}" for _ in range(n_entities) for t in range(n_periods)]
    
    # 设计协变量
    x = np.random.normal(0, 1, (n, 2))
    
    # 添加与个体效应相关的协变量（用于触发内生性）
    entity_effects = np.random.normal(0, 1, n_entities)
    correlation_with_entity = 0.5  # 引入部分相关性
    x[:, 0] += correlation_with_entity * np.repeat(entity_effects, n_periods)
    
    # 因变量
    y = []
    for i in range(n):
        entity_idx = i // n_periods
        y_value = (1 + 2 * x[i, 0] + 1.5 * x[i, 1] + 
                  entity_effects[entity_idx] + np.random.normal(0, 0.5))
        y.append(y_value)
    
    try:
        result = hausman_test(
            y=y,
            x=x.tolist(),
            entity_ids=entity_ids,
            time_periods=time_periods
        )
        if hasattr(result, 'hausman_statistic') and result.hausman_statistic >= 0:
            print(f"  Hausman统计量: {result.hausman_statistic:.4f}")
            print(f"  p值: {result.p_value:.4f}")
            print(f"  解释: {result.interpretation}")
            print("  ✓ Hausman检验测试通过\n")
        else:
            print(f"  ✗ Hausman检验返回无效统计量: {result.hausman_statistic if hasattr(result, 'hausman_statistic') else 'None'}\n")
    except Exception as e:
        print(f"  ✗ Hausman检验测试失败: {type(e).__name__}: {e}\n")


def test_difference_in_differences():
    """测试双重差分法"""
    print("测试双重差分法...")
    np.random.seed(42)
    n = 200
    
    # 生成数据
    treatment = np.concatenate([np.zeros(100), np.ones(100)]).tolist()
    time_period = np.concatenate([np.zeros(50), np.ones(50), np.zeros(50), np.ones(50)]).tolist()
    
    # 结果变量
    outcome = []
    for i in range(n):
        if treatment[i] == 0 and time_period[i] == 0:
            outcome.append(np.random.normal(10, 1))
        elif treatment[i] == 0 and time_period[i] == 1:
            outcome.append(np.random.normal(10, 1))
        elif treatment[i] == 1 and time_period[i] == 0:
            outcome.append(np.random.normal(10, 1))
        else:  # treatment[i] == 1 and time_period[i] == 1
            outcome.append(np.random.normal(12, 1))  # 处理效应为2
    
    try:
        result = difference_in_differences(
            treatment=treatment,
            time_period=time_period,
            outcome=outcome
        )
        print(f"  DID估计: {result.estimate:.4f}")
        print(f"  标准误: {result.std_error:.4f}")
        print(f"  p值: {result.p_value:.4f}")
        print("  ✓ 双重差分法测试通过\n")
    except Exception as e:
        print(f"  ✗ 双重差分法测试失败: {e}\n")


def test_triple_difference():
    """测试三重差分法"""
    print("测试三重差分法...")
    np.random.seed(42)
    n = 400
    
    # 生成变量
    treatment_group = np.tile([0, 0, 1, 1], n//4).tolist()
    time_period = np.tile([0, 1, 0, 1], n//4).tolist()
    cohort_group = np.tile([0, 0, 0, 0, 1, 1, 1, 1], n//8).tolist()
    
    # 结果变量
    outcome = []
    for i in range(n):
        if treatment_group[i] == 1 and time_period[i] == 1 and cohort_group[i] == 1:
            outcome.append(np.random.normal(12, 1))  # 处理效应
        else:
            outcome.append(np.random.normal(10, 1))
    
    try:
        result = triple_difference(
            outcome=outcome,
            treatment_group=treatment_group,
            time_period=time_period,
            cohort_group=cohort_group
        )
        print(f"  DDD估计: {result.estimate:.4f}")
        print(f"  标准误: {result.std_error:.4f}")
        print(f"  p值: {result.p_value:.4f}")
        print("  ✓ 三重差分法测试通过\n")
    except Exception as e:
        print(f"  ✗ 三重差分法测试失败: {e}\n")


def test_regression_discontinuity():
    """测试断点回归设计"""
    print("测试断点回归设计...")
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
    
    try:
        result = regression_discontinuity(
            running_variable=running_variable,
            outcome=outcome,
            cutoff=cutoff,
            bandwidth=0.5
        )
        print(f"  RDD估计: {result.estimate:.4f}")
        print(f"  标准误: {result.std_error:.4f}")
        print(f"  p值: {result.p_value:.4f}")
        print("  ✓ 断点回归设计测试通过\n")
    except Exception as e:
        print(f"  ✗ 断点回归设计测试失败: {e}\n")


def test_propensity_score_matching():
    """测试倾向得分匹配"""
    print("测试倾向得分匹配...")
    np.random.seed(42)
    n = 200
    
    # 协变量
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    covariates = np.column_stack([x1, x2]).tolist()
    
    # 倾向得分
    pscore = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    treatment = (np.random.uniform(0, 1, n) < pscore).astype(int).tolist()
    
    # 结果变量
    outcome = (2 + 1.5 * np.array(treatment) + 0.8 * x1 + 0.5 * x2 + 
              np.random.normal(0, 1, n)).tolist()
    
    try:
        result = propensity_score_matching(
            treatment=treatment,
            outcome=outcome,
            covariates=covariates
        )
        print(f"  ATE: {result.ate:.4f}")
        print(f"  标准误: {result.std_error:.4f}")
        print(f"  p值: {result.p_value:.4f}")
        print("  ✓ 倾向得分匹配测试通过\n")
    except Exception as e:
        print(f"  ✗ 倾向得分匹配测试失败: {e}\n")


def test_mediation_analysis():
    """测试中介效应分析"""
    print("测试中介效应分析...")
    np.random.seed(42)
    n = 200
    
    # 处理变量
    treatment = np.random.normal(0, 1, n).tolist()
    
    # 协变量
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    covariates = np.column_stack([x1, x2]).tolist()
    
    # 中介变量
    mediator = (1 + 0.8 * np.array(treatment) + 0.3 * x1 + 0.2 * x2 + 
               np.random.normal(0, 1, n)).tolist()
    
    # 结果变量
    outcome = (2 + 1.2 * np.array(treatment) + 0.7 * np.array(mediator) + 
              0.4 * x1 + 0.3 * x2 + np.random.normal(0, 1, n)).tolist()
    
    try:
        result = mediation_analysis(
            outcome=outcome,
            treatment=treatment,
            mediator=mediator,
            covariates=covariates
        )
        print(f"  直接效应: {result.direct_effect:.4f}")
        print(f"  间接效应: {result.indirect_effect:.4f}")
        print(f"  总效应: {result.total_effect:.4f}")
        print("  ✓ 中介效应分析测试通过\n")
    except Exception as e:
        print(f"  ✗ 中介效应分析测试失败: {e}\n")


def test_moderation_analysis():
    """测试调节效应分析"""
    print("测试调节效应分析...")
    np.random.seed(42)
    n = 200
    
    # 预测变量
    predictor = np.random.normal(0, 1, n).tolist()
    
    # 调节变量
    moderator = np.random.normal(0, 1, n).tolist()
    
    # 协变量
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    covariates = np.column_stack([x1, x2]).tolist()
    
    # 结果变量
    outcome = (2 + 1.2 * np.array(predictor) + 0.8 * np.array(moderator) + 
              0.5 * np.array(predictor) * np.array(moderator) + 
              0.3 * x1 + 0.2 * x2 + np.random.normal(0, 1, n)).tolist()
    
    try:
        result = moderation_analysis(
            outcome=outcome,
            predictor=predictor,
            moderator=moderator,
            covariates=covariates
        )
        print(f"  主效应: {result.main_effect:.4f}")
        print(f"  调节效应: {result.moderator_effect:.4f}")
        print(f"  交互效应: {result.interaction_effect:.4f}")
        print("  ✓ 调节效应分析测试通过\n")
    except Exception as e:
        print(f"  ✗ 调节效应分析测试失败: {e}\n")


def main():
    """主测试函数"""
    print("开始全面测试所有因果识别策略方法...\n")
    
    test_instrumental_variables()
    test_control_function()
    test_fixed_effects()
    test_random_effects()
    test_first_difference()
    test_hausman_test()
    test_difference_in_differences()
    test_triple_difference()
    test_regression_discontinuity()
    test_propensity_score_matching()
    test_mediation_analysis()
    test_moderation_analysis()
    
    print("所有测试完成！")


if __name__ == "__main__":
    main()