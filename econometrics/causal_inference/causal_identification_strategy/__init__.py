"""
因果识别策略模块
"""

from .control_function import ControlFunctionResult, control_function_approach
from .difference_in_differences import DIDResult, difference_in_differences
from .event_study import EventStudyResult, event_study
from .first_difference import FirstDifferenceResult, first_difference_model
from .fixed_effects import FixedEffectsResult, fixed_effects_model
from .hausman_specification import HausmanResult, hausman_test
from .instrumental_variables import IVResult, instrumental_variables_2sls
from .mediation_analysis import MediationResult, mediation_analysis
from .moderation_analysis import ModerationResult, moderation_analysis
from .propensity_score_matching import PSMMatchResult, propensity_score_matching
from .random_effects import RandomEffectsResult, random_effects_model
from .regression_discontinuity import RDDResult, regression_discontinuity
from .synthetic_control import SyntheticControlResult, synthetic_control_method
from .triple_difference import TripeDifferenceResult, triple_difference

__all__ = [
    "instrumental_variables_2sls",
    "difference_in_differences",
    "regression_discontinuity",
    "fixed_effects_model",
    "random_effects_model",
    "control_function_approach",
    "first_difference_model",
    "triple_difference",
    "event_study",
    "synthetic_control_method",
    "propensity_score_matching",
    "mediation_analysis",
    "moderation_analysis",
    "hausman_test",
    "IVResult",
    "DIDResult",
    "RDDResult",
    "FixedEffectsResult",
    "RandomEffectsResult",
    "ControlFunctionResult",
    "FirstDifferenceResult",
    "TripeDifferenceResult",
    "EventStudyResult",
    "SyntheticControlResult",
    "PSMMatchResult",
    "MediationResult",
    "ModerationResult",
    "HausmanResult",
]
