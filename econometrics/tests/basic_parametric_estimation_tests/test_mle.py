"""
MLE模型测试脚本
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from econometrics.basic_parametric_estimation.mle.mle_model import mle_estimation, MLEResult


def test_mle_normal():
    """测试正态分布MLE"""
    print("测试正态分布MLE...")
    
    # 生成正态分布测试数据
    np.random.seed(42)
    data = np.random.normal(5, 2, 100).tolist()  # 均值5，标准差2
    
    # 执行MLE估计
    result = mle_estimation(data, distribution="normal")
    
    # 验证结果类型
    assert isinstance(result, MLEResult), "结果应为MLEResult类型"
    
    # 验证参数数量
    assert len(result.parameters) == 2, "正态分布应有2个参数（均值和标准差）"
    
    # 验证参数名称
    assert result.param_names == ["mu", "sigma"], "参数名称应为['mu', 'sigma']"
    
    # 验证统计量合理性
    assert result.n_obs == 100, "观测数量应为100"
    assert result.log_likelihood < 0, "对数似然值应为负数"
    
    print("  估计参数:", result.parameters)
    print("  对数似然值:", result.log_likelihood)
    print("  AIC:", result.aic)
    print("  正态分布MLE测试通过")


def test_mle_poisson():
    """测试泊松分布MLE"""
    print("测试泊松分布MLE...")
    
    # 生成泊松分布测试数据
    np.random.seed(42)
    data = np.random.poisson(3, 100).tolist()  # 均值3
    
    # 执行MLE估计
    result = mle_estimation(data, distribution="poisson")
    
    # 验证结果类型
    assert isinstance(result, MLEResult), "结果应为MLEResult类型"
    
    # 验证参数数量
    assert len(result.parameters) == 1, "泊松分布应有1个参数"
    
    # 验证参数名称
    assert result.param_names == ["lambda"], "参数名称应为['lambda']"
    
    print("  估计参数:", result.parameters)
    print("  泊松分布MLE测试通过")


def test_mle_exponential():
    """测试指数分布MLE"""
    print("测试指数分布MLE...")
    
    # 生成指数分布测试数据
    np.random.seed(42)
    data = np.random.exponential(2, 100).tolist()  # 均值2
    
    # 执行MLE估计
    result = mle_estimation(data, distribution="exponential")
    
    # 验证结果类型
    assert isinstance(result, MLEResult), "结果应为MLEResult类型"
    
    # 验证参数数量
    assert len(result.parameters) == 1, "指数分布应有1个参数"
    
    # 验证参数名称
    assert result.param_names == ["lambda"], "参数名称应为['lambda']"
    
    print("  估计参数:", result.parameters)
    print("  指数分布MLE测试通过")


def test_mle_errors():
    """测试MLE错误处理"""
    print("测试MLE错误处理...")
    
    # 测试空数据
    try:
        mle_estimation([])
        assert False, "应该抛出ValueError异常"
    except ValueError:
        print("  空数据错误处理正确")
    
    # 测试不支持的分布类型
    try:
        mle_estimation([1, 2, 3], distribution="unsupported")
        assert False, "应该抛出ValueError异常"
    except ValueError:
        print("  不支持的分布类型错误处理正确")
    
    # 测试负值的指数分布数据
    try:
        mle_estimation([-1, 0, 1], distribution="exponential")
        assert False, "应该抛出ValueError异常"
    except ValueError:
        print("  负值指数分布数据错误处理正确")
    
    print("  MLE错误处理测试通过")


if __name__ == "__main__":
    print("开始测试MLE模型...")
    test_mle_normal()
    test_mle_poisson()
    test_mle_exponential()
    test_mle_errors()
    print("所有MLE测试通过!")