"""
模型选择 (Model Selection) 模块

包括：
- 信息准则（AIC/BIC/HQIC）
- 交叉验证（K折、留一法）
- 格兰杰因果检验
"""

from .model_selection_model import (
    ModelSelectionResult,
    model_selection_criteria
)

__all__ = [
    "ModelSelectionResult",
    "model_selection_criteria"
]