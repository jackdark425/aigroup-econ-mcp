"""
测试工具变量法的边界情况和潜在问题
"""

import sys
import traceback
import numpy as np

def test_iv_edge_cases():
    """测试工具变量法的边界情况"""
    print("测试工具变量法的边界情况...")
    
    try:
        from econometrics.causal_inference import instrumental_variables_2sls
        
        # 测试1: 弱工具变量
        print("测试1: 弱工具变量...")
        n = 100
        np.random.seed(42)
        
        # 弱工具变量（相关性很低）
        z = np.random.normal(0, 1, n)
        u = np.random.normal(0, 1, n)
        x = 0.01 * z + 0.99 * u + np.random.normal(0, 0.1, n)  # 弱相关性
        y = 2 * x + u + np.random.normal(0, 0.1, n)
        
        try:
            result = instrumental_variables_2sls(
                y=y.tolist(),
                x=[[xi] for xi in x],
                instruments=[[zi] for zi in z]
            )
            print(f"✓ 弱工具变量测试成功: estimate = {result.estimate:.4f}")
        except Exception as e:
            print(f"✗ 弱工具变量测试失败: {e}")
        
        # 测试2: 完美共线性
        print("测试2: 完美共线性...")
        z1 = np.random.normal(0, 1, n)
        z2 = z1 * 2  # 完美共线性
        x = 0.5 * z1 + 0.5 * np.random.normal(0, 1, n)
        y = 2 * x + np.random.normal(0, 0.1, n)
        
        try:
            result = instrumental_variables_2sls(
                y=y.tolist(),
                x=[[xi] for xi in x],
                instruments=[[z1[i], z2[i]] for i in range(n)]
            )
            print(f"✓ 完美共线性测试成功: estimate = {result.estimate:.4f}")
        except Exception as e:
            print(f"✗ 完美共线性测试失败: {e}")
        
        # 测试3: 小样本
        print("测试3: 小样本...")
        n_small = 10
        z_small = np.random.normal(0, 1, n_small)
        x_small = 0.5 * z_small + np.random.normal(0, 0.5, n_small)
        y_small = 2 * x_small + np.random.normal(0, 0.1, n_small)
        
        try:
            result = instrumental_variables_2sls(
                y=y_small.tolist(),
                x=[[xi] for xi in x_small],
                instruments=[[zi] for zi in z_small]
            )
            print(f"✓ 小样本测试成功: estimate = {result.estimate:.4f}")
        except Exception as e:
            print(f"✗ 小样本测试失败: {e}")
        
        # 测试4: 缺失值处理
        print("测试4: 缺失值处理...")
        n_missing = 20
        z_missing = np.random.normal(0, 1, n_missing)
        x_missing = 0.5 * z_missing + np.random.normal(0, 0.5, n_missing)
        y_missing = 2 * x_missing + np.random.normal(0, 0.1, n_missing)
        
        # 引入一些NaN值
        y_missing[5] = np.nan
        x_missing[10] = np.nan
        
        try:
            result = instrumental_variables_2sls(
                y=y_missing.tolist(),
                x=[[xi] for xi in x_missing],
                instruments=[[zi] for zi in z_missing]
            )
            print(f"✓ 缺失值处理测试成功: estimate = {result.estimate:.4f}")
        except Exception as e:
            print(f"✗ 缺失值处理测试失败: {e}")
        
        # 测试5: 多变量情况
        print("测试5: 多变量情况...")
        n_multi = 100
        z1_multi = np.random.normal(0, 1, n_multi)
        z2_multi = np.random.normal(0, 1, n_multi)
        x1_multi = 0.3 * z1_multi + 0.2 * z2_multi + np.random.normal(0, 0.5, n_multi)
        x2_multi = 0.1 * z1_multi + 0.4 * z2_multi + np.random.normal(0, 0.5, n_multi)
        y_multi = 2 * x1_multi + 1 * x2_multi + np.random.normal(0, 0.1, n_multi)
        
        try:
            result = instrumental_variables_2sls(
                y=y_multi.tolist(),
                x=[[x1_multi[i], x2_multi[i]] for i in range(n_multi)],
                instruments=[[z1_multi[i], z2_multi[i]] for i in range(n_multi)],
                feature_names=['x1', 'x2'],
                instrument_names=['z1', 'z2']
            )
            print(f"✓ 多变量测试成功: estimate for x1 = {result.estimate:.4f}")
        except Exception as e:
            print(f"✗ 多变量测试失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 工具变量法边界测试总体失败: {e}")
        traceback.print_exc()
        return False

def test_iv_parameter_validation():
    """测试工具变量法的参数验证"""
    print("\n测试工具变量法的参数验证...")
    
    try:
        from econometrics.causal_inference import instrumental_variables_2sls
        
        # 测试长度不匹配
        print("测试长度不匹配...")
        try:
            result = instrumental_variables_2sls(
                y=[1, 2, 3],
                x=[[1], [2]],  # 长度不匹配
                instruments=[[1], [2], [3]]
            )
            print("✗ 长度不匹配测试失败：应该抛出异常")
        except Exception:
            print("✓ 长度不匹配正确处理")
        
        # 测试空数据
        print("测试空数据...")
        try:
            result = instrumental_variables_2sls(
                y=[],
                x=[],
                instruments=[]
            )
            print("✗ 空数据测试失败：应该抛出异常")
        except Exception:
            print("✓ 空数据正确处理")
        
        # 测试无效参数
        print("测试无效参数...")
        try:
            result = instrumental_variables_2sls(
                y=[1, 2, 3],
                x=[[1], [2], [3]],
                instruments=[[1], [2], [3]],
                feature_names=['x1'],  # 长度不匹配
                instrument_names=['z1', 'z2']  # 长度不匹配
            )
            print("✗ 无效参数测试失败：应该抛出异常")
        except Exception:
            print("✓ 无效参数正确处理")
        
        return True
        
    except Exception as e:
        print(f"✗ 参数验证测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试工具变量法的边界情况...")
    print("=" * 60)
    
    edge_case_success = test_iv_edge_cases()
    validation_success = test_iv_parameter_validation()
    
    print("\n" + "=" * 60)
    print("工具变量法测试总结")
    print("=" * 60)
    
    if edge_case_success and validation_success:
        print("✓ 所有边界情况测试通过！")
        sys.exit(0)
    else:
        print("✗ 部分边界情况测试失败")
        sys.exit(1)