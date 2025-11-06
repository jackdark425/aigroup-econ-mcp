"""
统计推断技术模块
提供重采样、模拟和渐近推断方法
"""

from .bootstrap_methods import (
    bootstrap_inference,
    BootstrapResult
)

from .permutation_test import (
    permutation_test,
    PermutationTestResult
)

__all__ = [
    'bootstrap_inference',
    'BootstrapResult',
    'permutation_test',
    'PermutationTestResult'
]