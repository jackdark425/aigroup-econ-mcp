
"""
工具描述模块
提供各种计量经济学工具的描述信息
"""

from typing import Dict, Any

# 基础工具描述
DESCRIPTIVE_STATISTICS = {
    "name": "descriptive_statistics",
    "description": "计算描述性统计量，包括均值、中位数、标准差、分位数等",
    "parameters": {
        "data": {
            "type": "object",
            "description": "数值数据字典，键为变量名，值为数值列表"
        }
    }
}

OLS_REGRESSION = {
    "name": "ols_regression",
    "description": "普通最小二乘法回归分析",
    "parameters": {
        "y_data": {
            "type": "array",
            "items": {"type": "number"},
            "description": "因变量数据"
        },
        "x_data": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"}
            },
            "description": "自变量数据矩阵"
        },
        "feature_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "特征名称列表"
        },
        "constant": {
            "type": "boolean",
            "description": "是否包含常数项，默认为True"
        },
        "confidence_level": {
            "type": "number",
            "description": "置信水平，默认为0.95"
        }
    }
}

HYPOTHESIS_TESTING = {
    "name": "hypothesis_testing",
    "description": "假设检验，包括t检验、z检验、卡方检验等",
    "parameters": {
        "test_type": {
            "type": "string",
            "description": "检验类型: t_test, z_test, chi_square_test"
        },
        "data1": {
            "type": "array",
            "items": {"type": "number"},
            "description": "第一组数据"
        },
        "data2": {
            "type": "array",
            "items": {"type": "number"},
            "description": "第二组数据（仅在两样本检验时需要）"
        },
        "alpha": {
            "type": "number",
            "description": "显著性水平，默认为0.05"
        }
    }
}

# 新增的基础与参数估计工具描述
BASIC_PARAMETRIC_ESTIMATION_OLS = {
    "name": "basic_parametric_estimation_ols",
    "description": "普通最小二乘法(OLS)参数估计，用于线性回归模型参数估计",
    "parameters": {
        "y_data": {
            "type": "array",
            "items": {"type": "number"},
            "description": "因变量数据"
        },
        "x_data": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"}
            },
            "description": "自变量数据矩阵"
        },
        "feature_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "特征名称列表"
        },
        "constant": {
            "type": "boolean",
            "description": "是否包含常数项，默认为True"
        },
        "confidence_level": {
            "type": "number",
            "description": "置信水平，默认为0.95"
        }
    }
}

BASIC_PARAMETRIC_ESTIMATION_MLE = {
    "name": "basic_parametric_estimation_mle",
    "description": "最大似然估计(MLE)，用于估计统计模型的参数",
    "parameters": {
        "data": {
            "type": "array",
            "items": {"type": "number"},
            "description": "用于估计的数据"
        },
        "distribution": {
            "type": "string",
            "description": "假设的分布类型: normal(正态分布), poisson(泊松分布), exponential(指数分布)"
        },
        "initial_params": {
            "type": "array",
            "items": {"type": "number"},
            "description": "初始参数值"
        },
        "confidence_level": {
            "type": "number",
            "description": "置信水平，默认为0.95"
        }
    }
}

BASIC_PARAMETRIC_ESTIMATION_GMM = {
    "name": "basic_parametric_estimation_gmm",
    "description": "广义矩估计(GMM)，用于参数估计，特别适用于存在内生性的情况",
    "parameters": {
        "y_data": {
            "type": "array",
            "items": {"type": "number"},
            "description": "因变量数据"
        },
        "x_data": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"}
            },
            "description": "自变量数据矩阵"
        },
        "instruments": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"}
            },
            "description": "工具变量数据矩阵"
        },
        "feature_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "特征名称列表"
        },
        "constant": {
            "type": "boolean",
            "description": "是否包含常数项，默认为True"
        },
        "confidence_level": {
            "type": "number",
            "description": "置信水平，默认为0.95"
        }
    }
}

# 工具描述映射
TOOL_DESCRIPTIONS = {
    "descriptive_statistics": DESCRIPTIVE_STATISTICS,
    "ols_regression": OLS_REGRESSION,
    "hypothesis_testing": HYPOTHESIS_TESTING,
    "basic_parametric_estimation_ols": BASIC_PARAMETRIC_ESTIMATION_OLS,
    "basic_parametric_estimation_mle": BASIC_PARAMETRIC_ESTIMATION_MLE,
    "basic_parametric_estimation_gmm": BASIC_PARAMETRIC_ESTIMATION_GMM
}

def get_tool_description(tool_name: str) -> Dict[str, Any]:
    """
    获取工具描述
    
    Args:
        tool_name: 工具名称
        
    Returns:
        Dict[str, Any]: 工具描述
    """
    return TOOL_DESCRIPTIONS.get(tool_name, {})

def get_field_description(tool_name: str, field_name: str) -> str:
    """
    获取字段描述
    
    Args:
        tool_name: 工具名称
        field_name: 字段名称
        
    Returns:
        str: 字段描述
    """
    tool_desc = TOOL_DESCRIPTIONS.get(tool_name, {})
    return tool_desc.get("parameters", {}).get(field_name, {}).get("description", "")
