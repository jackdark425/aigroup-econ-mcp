"""
全面测试因果推断模块的类型注解和参数验证
"""

import sys
import traceback
import numpy as np
import pandas as pd

def test_instrumental_variables():
    """测试工具变量法"""
    print("测试工具变量法...")
    try:
        from econometrics.causal_inference import instrumental_variables_2sls
        
        # 生成测试数据
        n = 100
        np.random.seed(42)
        
        # 工具变量
        z = np.random.normal(0, 1, n)
        # 内生变量（与误差项相关）
        u = np.random.normal(0, 1, n)
        x = 0.5 * z + 0.5 * u + np.random.normal(0, 0.1, n)
        # 因变量
        y = 2 * x + u + np.random.normal(0, 0.1, n)
        
        # 转换为列表格式
        y_list = y.tolist()
        x_list = [[xi] for xi in x]
        z_list = [[zi] for zi in z]
        
        result = instrumental_variables_2sls(
            y=y_list,
            x=x_list,
            instruments=z_list,
            feature_names=['x1'],
            instrument_names=['z1']
        )
        
        print(f"✓ IV方法测试成功: estimate = {result.estimate:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ IV方法测试失败: {e}")
        traceback.print_exc()
        return False

def test_propensity_score_matching():
    """测试倾向得分匹配"""
    print("测试倾向得分匹配...")
    try:
        from econometrics.causal_inference import propensity_score_matching
        
        # 生成测试数据
        n = 200
        np.random.seed(42)
        
        # 协变量
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        
        # 倾向得分（逻辑函数）
        propensity = 1 / (1 + np.exp(-(0.5 * x1 + 0.5 * x2)))
        treatment = np.random.binomial(1, propensity)
        
        # 结果变量
        outcome = 2 * treatment + 0.5 * x1 + 0.5 * x2 + np.random.normal(0, 0.1, n)
        
        result = propensity_score_matching(
            treatment=treatment.tolist(),
            outcome=outcome.tolist(),
            covariates=[[x1[i], x2[i]] for i in range(n)]
        )
        
        print(f"✓ PSM方法测试成功: ATE = {result.ate:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ PSM方法测试失败: {e}")
        traceback.print_exc()
        return False

def test_regression_discontinuity():
    """测试断点回归"""
    print("测试断点回归...")
    try:
        from econometrics.causal_inference import regression_discontinuity
        
        # 生成测试数据
        n = 500
        np.random.seed(42)
        
        # 运行变量
        running = np.random.uniform(-2, 2, n)
        # 处理变量（运行变量大于0）
        treatment = (running > 0).astype(int)
        # 结果变量（有断点效应）
        outcome = 2 * treatment + 0.5 * running + np.random.normal(0, 0.1, n)
        
        result = regression_discontinuity(
            running_variable=running.tolist(),
            outcome=outcome.tolist(),
            cutoff=0.0,
            bandwidth=1.0
        )
        
        print(f"✓ RDD方法测试成功: estimate = {result.estimate:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ RDD方法测试失败: {e}")
        traceback.print_exc()
        return False

def test_synthetic_control():
    """测试合成控制法"""
    print("测试合成控制法...")
    try:
        from econometrics.causal_inference import synthetic_control_method
        
        # 生成测试数据
        n_units = 5  # 4个对照单元 + 1个处理单元
        n_time = 20
        treatment_period = 10
        
        np.random.seed(42)
        
        # 生成结果变量（面板数据）
        outcome = []
        for unit in range(n_units):
            base_trend = np.linspace(10, 20, n_time)
            noise = np.random.normal(0, 0.5, n_time)
            if unit == 0:  # 处理单元
                # 处理后效应
                effect = np.concatenate([np.zeros(treatment_period), 
                                       np.ones(n_time - treatment_period) * 3])
                outcome.extend(base_trend + effect + noise)
            else:  # 对照单元
                outcome.extend(base_trend + noise)
        
        result = synthetic_control_method(
            outcome=outcome,
            treatment_period=treatment_period,
            treated_unit="unit_0",
            donor_units=["unit_1", "unit_2", "unit_3", "unit_4"],
            time_periods=[f"t_{i}" for i in range(n_time)]
        )
        
        print(f"✓ 合成控制法测试成功: treatment_effect = {result.treatment_effect:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ 合成控制法测试失败: {e}")
        traceback.print_exc()
        return False

def test_fixed_effects():
    """测试固定效应模型"""
    print("测试固定效应模型...")
    try:
        from econometrics.causal_inference import fixed_effects_model
        
        # 生成面板数据
        n_entities = 10
        n_time = 5
        n_obs = n_entities * n_time
        
        np.random.seed(42)
        
        # 实体固定效应
        entity_effects = np.random.normal(0, 1, n_entities)
        
        y = []
        x = []
        entity_ids = []
        time_periods = []
        
        for i in range(n_entities):
            for t in range(n_time):
                x_val = np.random.normal(0, 1)
                y_val = 2 * x_val + entity_effects[i] + np.random.normal(0, 0.1)
                
                y.append(y_val)
                x.append([x_val])
                entity_ids.append(f"entity_{i}")
                time_periods.append(f"time_{t}")
        
        result = fixed_effects_model(
            y=y,
            x=x,
            entity_ids=entity_ids,
            time_periods=time_periods
        )
        
        print(f"✓ 固定效应模型测试成功: estimate = {result.estimate:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ 固定效应模型测试失败: {e}")
        traceback.print_exc()
        return False

def test_mediation_analysis():
    """测试中介效应分析"""
    print("测试中介效应分析...")
    try:
        from econometrics.causal_inference import mediation_analysis
        
        # 生成测试数据
        n = 200
        np.random.seed(42)
        
        treatment = np.random.normal(0, 1, n)
        mediator = 0.5 * treatment + np.random.normal(0, 0.5, n)
        outcome = 0.3 * treatment + 0.7 * mediator + np.random.normal(0, 0.1, n)
        
        result = mediation_analysis(
            outcome=outcome.tolist(),
            treatment=treatment.tolist(),
            mediator=mediator.tolist()
        )
        
        print(f"✓ 中介效应分析测试成功: direct_effect = {result.direct_effect:.4f}, indirect_effect = {result.indirect_effect:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ 中介效应分析测试失败: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """测试错误处理"""
    print("测试错误处理...")
    try:
        from econometrics.causal_inference import difference_in_differences
        
        # 测试空数据
        try:
            result = difference_in_differences([], [], [])
            print("✗ 空数据测试失败：应该抛出异常")
            return False
        except Exception:
            print("✓ 空数据正确处理")
        
        # 测试长度不匹配
        try:
            result = difference_in_differences([0, 1], [0, 1, 0], [1, 2, 3])
            print("✗ 长度不匹配测试失败：应该抛出异常")
            return False
        except Exception:
            print("✓ 长度不匹配正确处理")
            
        return True
        
    except Exception as e:
        print(f"✗ 错误处理测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始全面测试因果推断模块...")
    print("=" * 60)
    
    tests = [
        test_instrumental_variables,
        test_propensity_score_matching,
        test_regression_discontinuity,
        test_synthetic_control,
        test_fixed_effects,
        test_mediation_analysis,
        test_error_handling
    ]
    
    success_count = 0
    for test_func in tests:
        if test_func():
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"测试总结: {success_count}/{len(tests)} 通过")
    
    if success_count == len(tests):
        print("✓ 所有测试通过！")
        sys.exit(0)
    else:
        print("✗ 部分测试失败")
        sys.exit(1)