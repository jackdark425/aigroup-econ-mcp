"""
测试因果推断模块的导入和基本功能
"""

import sys
import traceback

def test_imports():
    """测试所有因果推断方法的导入"""
    methods_to_test = [
        'instrumental_variables_2sls',
        'difference_in_differences', 
        'regression_discontinuity',
        'fixed_effects_model',
        'random_effects_model',
        'propensity_score_matching',
        'synthetic_control_method',
        'mediation_analysis',
        'moderation_analysis',
        'hausman_test',
        'control_function_approach',
        'first_difference_model',
        'triple_difference',
        'event_study'
    ]
    
    print("=" * 60)
    print("测试因果推断模块导入")
    print("=" * 60)
    
    success_count = 0
    failed_methods = []
    
    for method in methods_to_test:
        try:
            exec(f"from econometrics.causal_inference import {method}")
            print(f"✓ {method} 导入成功")
            success_count += 1
        except Exception as e:
            print(f"✗ {method} 导入失败: {e}")
            failed_methods.append((method, str(e)))
    
    print(f"\n导入结果: {success_count}/{len(methods_to_test)} 成功")
    
    if failed_methods:
        print("\n失败的导入:")
        for method, error in failed_methods:
            print(f"  - {method}: {error}")
    
    return success_count == len(methods_to_test)

def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 60)
    print("测试基本功能")
    print("=" * 60)
    
    try:
        from econometrics.causal_inference import difference_in_differences, DIDResult
        
        # 测试DID方法
        treatment = [0, 0, 1, 1, 0, 0, 1, 1]
        time_period = [0, 1, 0, 1, 0, 1, 0, 1]
        outcome = [10, 12, 8, 15, 11, 13, 9, 16]
        
        result = difference_in_differences(treatment, time_period, outcome)
        print(f"✓ DID方法测试成功: estimate = {result.estimate:.4f}")
        
    except Exception as e:
        print(f"✗ DID方法测试失败: {e}")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("开始测试因果推断模块...")
    
    import_success = test_imports()
    functionality_success = test_basic_functionality()
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if import_success and functionality_success:
        print("✓ 所有测试通过！")
        sys.exit(0)
    else:
        print("✗ 部分测试失败")
        sys.exit(1)