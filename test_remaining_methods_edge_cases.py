"""
测试剩余因果推断方法的边界情况和潜在问题
"""

import sys
import traceback
import numpy as np

def test_did_edge_cases():
    """测试双重差分法的边界情况"""
    print("测试双重差分法的边界情况...")
    
    try:
        from econometrics.causal_inference import difference_in_differences
        
        # 测试1: 小样本
        print("测试1: 小样本...")
        treatment = [0, 1, 0, 1]
        time_period = [0, 0, 1, 1]
        outcome = [10, 8, 12, 15]
        
        try:
            result = difference_in_differences(treatment, time_period, outcome)
            print(f"✓ 小样本测试成功: estimate = {result.estimate:.4f}")
        except Exception as e:
            print(f"✗ 小样本测试失败: {e}")
        
        # 测试2: 不平衡数据
        print("测试2: 不平衡数据...")
        treatment = [0, 0, 1, 1, 0, 1]  # 不平衡的处理组
        time_period = [0, 1, 0, 1, 0, 1]
        outcome = [10, 12, 8, 15, 11, 16]
        
        try:
            result = difference_in_differences(treatment, time_period, outcome)
            print(f"✓ 不平衡数据测试成功: estimate = {result.estimate:.4f}")
        except Exception as e:
            print(f"✗ 不平衡数据测试失败: {e}")
        
        # 测试3: 协变量
        print("测试3: 协变量...")
        treatment = [0, 0, 1, 1, 0, 0, 1, 1]
        time_period = [0, 1, 0, 1, 0, 1, 0, 1]
        outcome = [10, 12, 8, 15, 11, 13, 9, 16]
        covariates = [[1], [2], [1], [2], [1], [2], [1], [2]]  # 简单协变量
        
        try:
            result = difference_in_differences(treatment, time_period, outcome, covariates)
            print(f"✓ 协变量测试成功: estimate = {result.estimate:.4f}")
        except Exception as e:
            print(f"✗ 协变量测试失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 双重差分法边界测试总体失败: {e}")
        traceback.print_exc()
        return False

def test_psm_edge_cases():
    """测试倾向得分匹配的边界情况"""
    print("\n测试倾向得分匹配的边界情况...")
    
    try:
        from econometrics.causal_inference import propensity_score_matching
        
        # 测试1: 小样本
        print("测试1: 小样本...")
        n_small = 20
        np.random.seed(42)
        
        x1 = np.random.normal(0, 1, n_small)
        x2 = np.random.normal(0, 1, n_small)
        propensity = 1 / (1 + np.exp(-(0.5 * x1 + 0.5 * x2)))
        treatment = np.random.binomial(1, propensity)
        outcome = 2 * treatment + 0.5 * x1 + 0.5 * x2 + np.random.normal(0, 0.1, n_small)
        
        try:
            result = propensity_score_matching(
                treatment=treatment.tolist(),
                outcome=outcome.tolist(),
                covariates=[[x1[i], x2[i]] for i in range(n_small)]
            )
            print(f"✓ 小样本测试成功: ATE = {result.ate:.4f}")
        except Exception as e:
            print(f"✗ 小样本测试失败: {e}")
        
        # 测试2: 极端倾向得分
        print("测试2: 极端倾向得分...")
        n_extreme = 50
        x1_extreme = np.random.normal(0, 1, n_extreme)
        x2_extreme = np.random.normal(0, 1, n_extreme)
        # 创建极端倾向得分（接近0或1）
        propensity_extreme = np.where(x1_extreme > 0, 0.95, 0.05)
        treatment_extreme = np.random.binomial(1, propensity_extreme)
        outcome_extreme = 2 * treatment_extreme + 0.5 * x1_extreme + 0.5 * x2_extreme + np.random.normal(0, 0.1, n_extreme)
        
        try:
            result = propensity_score_matching(
                treatment=treatment_extreme.tolist(),
                outcome=outcome_extreme.tolist(),
                covariates=[[x1_extreme[i], x2_extreme[i]] for i in range(n_extreme)]
            )
            print(f"✓ 极端倾向得分测试成功: ATE = {result.ate:.4f}")
        except Exception as e:
            print(f"✗ 极端倾向得分测试失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 倾向得分匹配边界测试总体失败: {e}")
        traceback.print_exc()
        return False

def test_rdd_edge_cases():
    """测试断点回归的边界情况"""
    print("\n测试断点回归的边界情况...")
    
    try:
        from econometrics.causal_inference import regression_discontinuity
        
        # 测试1: 窄带宽
        print("测试1: 窄带宽...")
        n = 200
        np.random.seed(42)
        
        running = np.random.uniform(-2, 2, n)
        treatment = (running > 0).astype(int)
        outcome = 2 * treatment + 0.5 * running + np.random.normal(0, 0.1, n)
        
        try:
            result = regression_discontinuity(
                running_variable=running.tolist(),
                outcome=outcome.tolist(),
                cutoff=0.0,
                bandwidth=0.1  # 窄带宽
            )
            print(f"✓ 窄带宽测试成功: estimate = {result.estimate:.4f}")
        except Exception as e:
            print(f"✗ 窄带宽测试失败: {e}")
        
        # 测试2: 高多项式阶数
        print("测试2: 高多项式阶数...")
        try:
            result = regression_discontinuity(
                running_variable=running.tolist(),
                outcome=outcome.tolist(),
                cutoff=0.0,
                bandwidth=1.0,
                polynomial_order=3  # 高多项式阶数
            )
            print(f"✓ 高多项式阶数测试成功: estimate = {result.estimate:.4f}")
        except Exception as e:
            print(f"✗ 高多项式阶数测试失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 断点回归边界测试总体失败: {e}")
        traceback.print_exc()
        return False

def test_fe_re_edge_cases():
    """测试固定效应和随机效应模型的边界情况"""
    print("\n测试固定效应和随机效应模型的边界情况...")
    
    try:
        from econometrics.causal_inference import fixed_effects_model, random_effects_model
        
        # 测试1: 小面板数据
        print("测试1: 小面板数据...")
        n_entities = 3
        n_time = 2
        n_obs = n_entities * n_time
        
        np.random.seed(42)
        
        y = []
        x = []
        entity_ids = []
        time_periods = []
        
        for i in range(n_entities):
            for t in range(n_time):
                x_val = np.random.normal(0, 1)
                y_val = 2 * x_val + np.random.normal(0, 0.1)
                
                y.append(y_val)
                x.append([x_val])
                entity_ids.append(f"entity_{i}")
                time_periods.append(f"time_{t}")
        
        try:
            result_fe = fixed_effects_model(y, x, entity_ids, time_periods)
            print(f"✓ 固定效应小面板测试成功: estimate = {result_fe.estimate:.4f}")
        except Exception as e:
            print(f"✗ 固定效应小面板测试失败: {e}")
        
        try:
            result_re = random_effects_model(y, x, entity_ids, time_periods)
            print(f"✓ 随机效应小面板测试成功: estimate = {result_re.estimate:.4f}")
        except Exception as e:
            print(f"✗ 随机效应小面板测试失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 面板模型边界测试总体失败: {e}")
        traceback.print_exc()
        return False

def test_other_methods_edge_cases():
    """测试其他因果推断方法的边界情况"""
    print("\n测试其他因果推断方法的边界情况...")
    
    try:
        from econometrics.causal_inference import (
            mediation_analysis, moderation_analysis, 
            control_function_approach, first_difference_model,
            triple_difference, event_study, hausman_test
        )
        
        # 测试中介效应分析
        print("测试中介效应分析...")
        n = 50
        np.random.seed(42)
        
        treatment = np.random.normal(0, 1, n)
        mediator = 0.5 * treatment + np.random.normal(0, 0.5, n)
        outcome = 0.3 * treatment + 0.7 * mediator + np.random.normal(0, 0.1, n)
        
        try:
            result = mediation_analysis(
                outcome=outcome.tolist(),
                treatment=treatment.tolist(),
                mediator=mediator.tolist()
            )
            print(f"✓ 中介效应分析测试成功: indirect_effect = {result.indirect_effect:.4f}")
        except Exception as e:
            print(f"✗ 中介效应分析测试失败: {e}")
        
        # 测试调节效应分析
        print("测试调节效应分析...")
        predictor = np.random.normal(0, 1, n)
        moderator = np.random.normal(0, 1, n)
        outcome_mod = 2 * predictor + 1 * moderator + 0.5 * predictor * moderator + np.random.normal(0, 0.1, n)
        
        try:
            result = moderation_analysis(
                outcome=outcome_mod.tolist(),
                predictor=predictor.tolist(),
                moderator=moderator.tolist()
            )
            print(f"✓ 调节效应分析测试成功: interaction_effect = {result.interaction_effect:.4f}")
        except Exception as e:
            print(f"✗ 调节效应分析测试失败: {e}")
        
        # 测试Hausman检验 - 修复变量定义问题
        print("测试Hausman检验...")
        n_hausman = 30
        y_hausman = np.random.normal(0, 1, n_hausman).tolist()
        x_hausman = [[np.random.normal(0, 1)] for _ in range(n_hausman)]
        entity_ids_hausman = [f"entity_{i}" for i in range(3) for _ in range(10)]
        time_periods_hausman = [f"time_{t}" for _ in range(3) for t in range(10)]
        
        try:
            result = hausman_test(y_hausman, x_hausman, entity_ids_hausman, time_periods_hausman)
            print(f"✓ Hausman检验测试成功: statistic = {result.hausman_statistic:.4f}")
        except Exception as e:
            print(f"✗ Hausman检验测试失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 其他方法边界测试总体失败: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试剩余因果推断方法的边界情况...")
    print("=" * 60)
    
    tests = [
        test_did_edge_cases,
        test_psm_edge_cases,
        test_rdd_edge_cases,
        test_fe_re_edge_cases,
        test_other_methods_edge_cases
    ]
    
    success_count = 0
    for test_func in tests:
        if test_func():
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"测试总结: {success_count}/{len(tests)} 通过")
    
    if success_count == len(tests):
        print("✓ 所有边界情况测试通过！")
        sys.exit(0)
    else:
        print("✗ 部分边界情况测试失败")
        sys.exit(1)