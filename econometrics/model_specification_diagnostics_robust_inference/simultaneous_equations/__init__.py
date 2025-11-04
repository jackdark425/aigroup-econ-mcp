"""
联立方程模型 (Simultaneous Equations Models) 模块

处理双向因果关系的模型方法
"""

from .simultaneous_equations_model import (
    SimultaneousEquationsResult,
    two_stage_least_squares
)

__all__ = [
    "SimultaneousEquationsResult",
    "two_stage_least_squares"
]