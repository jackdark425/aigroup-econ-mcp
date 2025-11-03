"""
工具描述模块 - 优化版
统一管理所有MCP工具的描述信息，为大模型提供详细、结构化的工具说明
包含使用示例、参数说明、适用场景等信息，提升大模型调用体验
"""

from typing import Dict, Any, List, Optional
from pydantic import Field


class ToolDescription:
    """工具描述类 - 优化版"""
    
    def __init__(self, name: str, description: str, field_descriptions: Dict[str, str] = None,
                 examples: Optional[List[str]] = None, use_cases: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.field_descriptions = field_descriptions or {}
        self.examples = examples or []
        self.use_cases = use_cases or []
    
    def get_field_description(self, field_name: str, default: str = "") -> str:
        """获取字段描述"""
        return self.field_descriptions.get(field_name, default)
    
    def get_full_description(self) -> str:
        """获取完整描述，包含示例和用例"""
        full_desc = self.description
        if self.examples:
            full_desc += "\n\n使用示例:\n" + "\n".join(f"- {example}" for example in self.examples)
        if self.use_cases:
            full_desc += "\n\n适用场景:\n" + "\n".join(f"- {use_case}" for use_case in self.use_cases)
        return full_desc


# ============================================================================
# 基础统计工具描述 (5个)
# ============================================================================

DESCRIPTIVE_STATISTICS = ToolDescription(
    name="descriptive_statistics",
    description="""计算描述性统计量

支持三种输入方式：file_path(文件路径)、file_content(文件内容)、data(直接数据)
支持格式：CSV、JSON、TXT（单列/多列/键值对三种格式）""",
    field_descriptions={
        "file_path": "CSV/JSON/TXT文件路径",
        "file_content": "CSV/JSON/TXT文件内容",
        "file_format": "文件格式(csv/json/txt/auto)",
        "data": "数据字典"
    }
)

OLS_REGRESSION = ToolDescription(
    name="ols_regression",
    description="""OLS回归分析
    
支持文件输入或直接数据输入。文件格式示例：
CSV: 最后一列为因变量，其余列为自变量""",
    field_descriptions={
        "file_path": "CSV/JSON/TXT文件路径",
        "file_content": "CSV/JSON/TXT文件内容",
        "file_format": "文件格式",
        "y_data": "因变量(直接输入)",
        "x_data": "自变量(直接输入)",
        "feature_names": "特征名称"
    }
)

HYPOTHESIS_TESTING = ToolDescription(
    name="hypothesis_testing",
    description="假设检验 - 支持文件或直接数据输入",
    field_descriptions={
        "file_path": "文件路径",
        "file_content": "文件内容",
        "file_format": "文件格式",
        "data": "第一组数据",
        "data2": "第二组数据",
        "test_type": "检验类型(t_test/adf)"
    }
)

TIME_SERIES_ANALYSIS = ToolDescription(
    name="time_series_analysis",
    description="时间序列分析 - 支持文件或直接数据输入",
    field_descriptions={
        "file_path": "文件路径",
        "file_content": "文件内容",
        "file_format": "文件格式",
        "data": "时间序列数据"
    }
)

CORRELATION_ANALYSIS = ToolDescription(
    name="correlation_analysis",
    description="相关性分析 - 支持文件或直接数据输入",
    field_descriptions={
        "file_path": "文件路径",
        "file_content": "文件内容",
        "file_format": "文件格式",
        "data": "多变量数据",
        "method": "相关系数类型"
    }
)


# ============================================================================
# 面板数据工具描述 (4个)
# ============================================================================

PANEL_FIXED_EFFECTS = ToolDescription(
    name="panel_fixed_effects",
    description="固定效应模型 - 支持文件输入",
    field_descriptions={
        "file_path": "CSV文件路径。CSV格式要求：必须包含实体ID列(列名含entity_id/id/entity/firm/company/country/region之一)和时间列(列名含time_period/time/date/year/month/period/quarter之一)",
        "file_content": "CSV/JSON/TXT文件内容字符串",
        "file_format": "文件格式(csv/json/txt/auto)。面板数据推荐使用CSV格式",
        "y_data": "因变量数据列表，数值格式",
        "x_data": "自变量数据矩阵，二维列表格式",
        "entity_ids": "实体ID列表，字符串格式，如 ['A', 'B', 'C', ...]",
        "time_periods": "时间周期列表，字符串格式，如 ['2020', '2021', '2022', ...]",
        "feature_names": "自变量名称列表，如 ['Investment', 'Employment', 'R&D']",
        "entity_effects": "是否包含实体固定效应，默认True",
        "time_effects": "是否包含时间固定效应，默认False"
    }
)

PANEL_RANDOM_EFFECTS = ToolDescription(
    name="panel_random_effects",
    description="随机效应模型 - 支持文件输入",
    field_descriptions={
        "file_path": "CSV文件路径。CSV格式要求：必须包含实体ID列和时间列",
        "file_content": "CSV/JSON/TXT文件内容字符串",
        "file_format": "文件格式(csv/json/txt/auto)",
        "y_data": "因变量数据列表，数值格式",
        "x_data": "自变量数据矩阵，二维列表格式",
        "entity_ids": "实体ID列表，字符串格式",
        "time_periods": "时间周期列表，字符串格式",
        "feature_names": "自变量名称列表",
        "entity_effects": "是否包含实体随机效应，默认True",
        "time_effects": "是否包含时间随机效应，默认False"
    }
)

PANEL_HAUSMAN_TEST = ToolDescription(
    name="panel_hausman_test",
    description="Hausman检验 - 支持文件输入",
    field_descriptions={
        "file_path": "CSV文件路径。CSV格式要求：必须包含实体ID列和时间列",
        "file_content": "CSV/JSON/TXT文件内容字符串",
        "file_format": "文件格式(csv/json/txt/auto)",
        "y_data": "因变量数据列表",
        "x_data": "自变量数据矩阵",
        "entity_ids": "实体ID列表",
        "time_periods": "时间周期列表",
        "feature_names": "自变量名称列表"
    }
)

PANEL_UNIT_ROOT_TEST = ToolDescription(
    name="panel_unit_root_test",
    description="面板单位根检验 - 支持文件输入",
    field_descriptions={
        "file_path": "CSV文件路径。CSV格式要求：必须包含实体ID列和时间列。数据量要求：至少3个实体，每个实体至少5个时间点",
        "file_content": "CSV/JSON/TXT文件内容字符串",
        "file_format": "文件格式(csv/json/txt/auto)",
        "data": "面板数据（从文件解析后的数据）",
        "y_data": "需要检验的变量数据",
        "x_data": "自变量数据（通常不用）",
        "entity_ids": "实体ID列表",
        "time_periods": "时间周期列表",
        "feature_names": "变量名称列表",
        "test_type": "检验类型(levinlin/ips/adf)"
    }
)


# ============================================================================
# 高级时间序列工具描述 (5个)
# ============================================================================

VAR_MODEL_ANALYSIS = ToolDescription(
    name="var_model_analysis",
    description="VAR模型分析 - 支持文件输入",
    field_descriptions={
        "file_path": "CSV/JSON/TXT文件路径，包含多个时间序列变量",
        "file_content": "CSV/JSON/TXT文件内容字符串",
        "file_format": "文件格式(csv/json/txt/auto)",
        "data": "多变量时间序列数据字典",
        "max_lags": "最大滞后阶数，默认5",
        "ic": "信息准则(aic/bic/hqic)，默认aic"
    }
)

VECM_MODEL_ANALYSIS = ToolDescription(
    name="vecm_model_analysis",
    description="VECM模型分析 - 支持文件输入",
    field_descriptions={
        "file_path": "CSV/JSON/TXT文件路径，包含协整的时间序列变量",
        "file_content": "CSV/JSON/TXT文件内容字符串",
        "file_format": "文件格式(csv/json/txt/auto)",
        "data": "多变量时间序列数据字典",
        "coint_rank": "协整秩，默认1",
        "deterministic": "确定性项(co/ci/lo/li)，默认co",
        "max_lags": "最大滞后阶数，默认5"
    }
)

GARCH_MODEL_ANALYSIS = ToolDescription(
    name="garch_model_analysis",
    description="GARCH模型分析 - 支持文件输入",
    field_descriptions={
        "file_path": "CSV/JSON/TXT文件路径，包含单个时间序列",
        "file_content": "CSV/JSON/TXT文件内容字符串",
        "file_format": "文件格式(csv/json/txt/auto)",
        "data": "时间序列数据",
        "order": "GARCH阶数(p,q)，默认(1,1)",
        "dist": "误差分布(normal/t/skewt)，默认normal"
    }
)

STATE_SPACE_MODEL_ANALYSIS = ToolDescription(
    name="state_space_model_analysis",
    description="状态空间模型分析 - 支持文件输入",
    field_descriptions={
        "file_path": "CSV/JSON/TXT文件路径，包含时间序列数据",
        "file_content": "CSV/JSON/TXT文件内容字符串",
        "file_format": "文件格式(csv/json/txt/auto)",
        "data": "时间序列数据",
        "state_dim": "状态维度，默认1",
        "observation_dim": "观测维度，默认1",
        "trend": "是否包含趋势项，默认True",
        "seasonal": "是否包含季节项，默认False",
        "period": "季节周期，默认12"
    }
)

VARIANCE_DECOMPOSITION_ANALYSIS = ToolDescription(
    name="variance_decomposition_analysis",
    description="方差分解分析 - 支持文件输入",
    field_descriptions={
        "file_path": "CSV/JSON/TXT文件路径，包含多个时间序列变量",
        "file_content": "CSV/JSON/TXT文件内容字符串",
        "file_format": "文件格式(csv/json/txt/auto)",
        "data": "多变量时间序列数据字典",
        "periods": "预测期数，默认10",
        "max_lags": "最大滞后阶数，默认5"
    }
)


# ============================================================================
# 机器学习工具描述 (6个)
# ============================================================================

RANDOM_FOREST_REGRESSION_ANALYSIS = ToolDescription(
    name="random_forest_regression_analysis",
    description="随机森林回归 - 支持文件输入",
    field_descriptions={
        "file_path": "CSV/JSON/TXT文件路径，最后一列为因变量",
        "file_content": "CSV/JSON/TXT文件内容字符串",
        "file_format": "文件格式(csv/json/txt/auto)",
        "y_data": "因变量数据",
        "x_data": "自变量数据矩阵",
        "feature_names": "特征名称列表",
        "n_estimators": "树的数量，默认100",
        "max_depth": "树的最大深度，默认None（不限制）"
    }
)

GRADIENT_BOOSTING_REGRESSION_ANALYSIS = ToolDescription(
    name="gradient_boosting_regression_analysis",
    description="梯度提升树回归 - 支持文件输入",
    field_descriptions={
        "file_path": "CSV/JSON/TXT文件路径，最后一列为因变量",
        "file_content": "CSV/JSON/TXT文件内容字符串",
        "file_format": "文件格式(csv/json/txt/auto)",
        "y_data": "因变量数据",
        "x_data": "自变量数据矩阵",
        "feature_names": "特征名称列表",
        "n_estimators": "提升轮数，默认100",
        "learning_rate": "学习率，默认0.1",
        "max_depth": "树的最大深度，默认3"
    }
)

LASSO_REGRESSION_ANALYSIS = ToolDescription(
    name="lasso_regression_analysis",
    description="Lasso回归 - 支持文件输入",
    field_descriptions={
        "file_path": "CSV/JSON/TXT文件路径，最后一列为因变量",
        "file_content": "CSV/JSON/TXT文件内容字符串",
        "file_format": "文件格式(csv/json/txt/auto)",
        "y_data": "因变量数据",
        "x_data": "自变量数据矩阵",
        "feature_names": "特征名称列表",
        "alpha": "正则化强度，默认1.0"
    }
)

RIDGE_REGRESSION_ANALYSIS = ToolDescription(
    name="ridge_regression_analysis",
    description="Ridge回归 - 支持文件输入",
    field_descriptions={
        "file_path": "CSV/JSON/TXT文件路径，最后一列为因变量",
        "file_content": "CSV/JSON/TXT文件内容字符串",
        "file_format": "文件格式(csv/json/txt/auto)",
        "y_data": "因变量数据",
        "x_data": "自变量数据矩阵",
        "feature_names": "特征名称列表",
        "alpha": "正则化强度，默认1.0"
    }
)

CROSS_VALIDATION_ANALYSIS = ToolDescription(
    name="cross_validation_analysis",
    description="交叉验证 - 支持文件输入",
    field_descriptions={
        "file_path": "CSV/JSON/TXT文件路径，最后一列为因变量",
        "file_content": "CSV/JSON/TXT文件内容字符串",
        "file_format": "文件格式(csv/json/txt/auto)",
        "y_data": "因变量数据",
        "x_data": "自变量数据矩阵",
        "feature_names": "特征名称列表",
        "model_type": "模型类型(random_forest/gradient_boosting/lasso/ridge)，默认random_forest",
        "cv_folds": "交叉验证折数，默认5",
        "scoring": "评分指标(r2/mse/mae)，默认r2"
    }
)

FEATURE_IMPORTANCE_ANALYSIS_TOOL = ToolDescription(
    name="feature_importance_analysis_tool",
    description="特征重要性分析 - 支持文件输入",
    field_descriptions={
        "file_path": "CSV/JSON/TXT文件路径，最后一列为因变量",
        "file_content": "CSV/JSON/TXT文件内容字符串",
        "file_format": "文件格式(csv/json/txt/auto)",
        "y_data": "因变量数据",
        "x_data": "自变量数据矩阵",
        "feature_names": "特征名称列表",
        "method": "分析方法(random_forest/gradient_boosting/permutation)，默认random_forest",
        "top_k": "返回前K个重要特征，默认5"
    }
)


# ============================================================================
# 辅助函数
# ============================================================================

def get_tool_description(tool_name: str) -> str:
    """获取工具描述"""
    tool_map = {
        "descriptive_statistics": DESCRIPTIVE_STATISTICS,
        "ols_regression": OLS_REGRESSION,
        "hypothesis_testing": HYPOTHESIS_TESTING,
        "time_series_analysis": TIME_SERIES_ANALYSIS,
        "correlation_analysis": CORRELATION_ANALYSIS,
        "panel_fixed_effects": PANEL_FIXED_EFFECTS,
        "panel_random_effects": PANEL_RANDOM_EFFECTS,
        "panel_hausman_test": PANEL_HAUSMAN_TEST,
        "panel_unit_root_test": PANEL_UNIT_ROOT_TEST,
        "var_model_analysis": VAR_MODEL_ANALYSIS,
        "vecm_model_analysis": VECM_MODEL_ANALYSIS,
        "garch_model_analysis": GARCH_MODEL_ANALYSIS,
        "state_space_model_analysis": STATE_SPACE_MODEL_ANALYSIS,
        "variance_decomposition_analysis": VARIANCE_DECOMPOSITION_ANALYSIS,
        "random_forest_regression_analysis": RANDOM_FOREST_REGRESSION_ANALYSIS,
        "gradient_boosting_regression_analysis": GRADIENT_BOOSTING_REGRESSION_ANALYSIS,
        "lasso_regression_analysis": LASSO_REGRESSION_ANALYSIS,
        "ridge_regression_analysis": RIDGE_REGRESSION_ANALYSIS,
        "cross_validation_analysis": CROSS_VALIDATION_ANALYSIS,
        "feature_importance_analysis_tool": FEATURE_IMPORTANCE_ANALYSIS_TOOL
    }
    
    tool = tool_map.get(tool_name)
    if tool:
        return tool.get_full_description()
    return ""


def get_field_description(tool_name: str, field_name: str, default: str = "") -> str:
    """获取字段描述"""
    tool_map = {
        "descriptive_statistics": DESCRIPTIVE_STATISTICS,
        "ols_regression": OLS_REGRESSION,
        "hypothesis_testing": HYPOTHESIS_TESTING,
        "time_series_analysis": TIME_SERIES_ANALYSIS,
        "correlation_analysis": CORRELATION_ANALYSIS,
        "panel_fixed_effects": PANEL_FIXED_EFFECTS,
        "panel_random_effects": PANEL_RANDOM_EFFECTS,
        "panel_hausman_test": PANEL_HAUSMAN_TEST,
        "panel_unit_root_test": PANEL_UNIT_ROOT_TEST,
        "var_model_analysis": VAR_MODEL_ANALYSIS,
        "vecm_model_analysis": VECM_MODEL_ANALYSIS,
        "garch_model_analysis": GARCH_MODEL_ANALYSIS,
        "state_space_model_analysis": STATE_SPACE_MODEL_ANALYSIS,
        "variance_decomposition_analysis": VARIANCE_DECOMPOSITION_ANALYSIS,
        "random_forest_regression_analysis": RANDOM_FOREST_REGRESSION_ANALYSIS,
        "gradient_boosting_regression_analysis": GRADIENT_BOOSTING_REGRESSION_ANALYSIS,
        "lasso_regression_analysis": LASSO_REGRESSION_ANALYSIS,
        "ridge_regression_analysis": RIDGE_REGRESSION_ANALYSIS,
        "cross_validation_analysis": CROSS_VALIDATION_ANALYSIS,
        "feature_importance_analysis_tool": FEATURE_IMPORTANCE_ANALYSIS_TOOL
    }
    
    tool = tool_map.get(tool_name)
    if tool:
        return tool.get_field_description(field_name, default)
    return default