"""
模型诊断测试 (Diagnostic Tests) 模块

包括各种统计检验方法：
- 异方差检验（White、Breusch-Pagan）
- 自相关检验（Durbin-Watson、Ljung-Box）
- 正态性检验（Jarque-Bera）
- 多重共线性诊断（VIF）
- 内生性检验（Durbin-Wu-Hausman）
"""

from .diagnostic_tests_model import (
    DiagnosticTestsResult,
    diagnostic_tests
)

__all__ = [
    "DiagnosticTestsResult",
    "diagnostic_tests"
]