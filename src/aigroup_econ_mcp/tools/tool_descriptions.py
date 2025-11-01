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
# 基础统计工具描述 (5个) - 优化版
# ============================================================================

DESCRIPTIVE_STATISTICS = ToolDescription(
    name="descriptive_statistics",
    description="""计算描述性统计量

功能说明:
- 计算数据的均值、标准差、最小值、最大值、中位数、四分位数等
- 支持数值型数据的全面统计分析
- 自动识别异常值和数据分布特征

输入方式优先级:
1. file_path: 文件路径 (推荐使用，支持CSV/JSON格式)
2. file_content: 文件内容字符串 (适合小文件)
3. data: 直接传入数据字典 (适合内存数据)

输出包含:
- 基本统计量: 均值、标准差、最小值、最大值
- 分布统计: 中位数、四分位数、偏度、峰度
- 数据质量: 缺失值统计、异常值检测""",
    field_descriptions={
        "file_path": "CSV/JSON文件路径。支持相对路径和绝对路径，文件应包含数值型数据列",
        "file_content": "CSV/JSON文件内容字符串。适合小文件直接传输，避免路径依赖",
        "file_format": "文件格式: csv/json/auto。auto模式自动检测文件格式",
        "data": "数据字典格式: {'变量名1': [值1, 值2, ...], '变量名2': [值1, 值2, ...]}"
    },
    examples=[
        "分析GDP数据的分布特征",
        "计算股票收益率的描述性统计",
        "评估消费者收入数据的集中趋势和离散程度"
    ],
    use_cases=[
        "数据探索性分析(EDA)",
        "数据质量评估",
        "变量分布特征分析",
        "异常值检测"
    ]
)

OLS_REGRESSION = ToolDescription(
    name="ols_regression",
    description="""OLS回归分析

功能说明:
- 执行普通最小二乘法(OLS)回归分析
- 估计回归系数、标准误、t统计量、p值
- 计算模型拟合优度(R²、调整R²)
- 进行模型显著性检验(F检验)

数据格式要求:
- 文件输入: CSV文件最后一列为因变量，其余列为自变量
- 直接输入: 提供y_data(因变量)和x_data(自变量矩阵)
- 支持特征名称自定义

模型输出:
- 回归系数估计值和统计显著性
- 模型拟合优度指标
- 残差分析和诊断统计量
- 置信区间和预测区间""",
    field_descriptions={
        "file_path": "CSV/JSON文件路径。CSV格式: 最后一列为因变量，其余列为自变量",
        "file_content": "CSV/JSON文件内容字符串。JSON格式: {'y': [因变量], 'x1': [自变量1], ...}",
        "file_format": "文件格式: csv/json/auto。推荐使用CSV格式",
        "y_data": "因变量数据列表，如 [1.2, 3.4, 5.6, ...]",
        "x_data": "自变量数据矩阵，如 [[1, 2], [3, 4], [5, 6], ...]",
        "feature_names": "自变量名称列表，如 ['GDP', 'Population', 'Investment']"
    },
    examples=[
        "分析GDP增长与投资、消费的关系",
        "预测房价与面积、位置、房龄的关系",
        "研究教育水平对收入的影响"
    ],
    use_cases=[
        "经济变量关系分析",
        "商业预测建模",
        "政策效果评估",
        "影响因素识别"
    ]
)

HYPOTHESIS_TESTING = ToolDescription(
    name="hypothesis_testing",
    description="""假设检验分析

功能说明:
- t检验: 比较两组数据的均值差异
- ADF检验: 时间序列平稳性检验
- 支持单样本、双样本检验
- 自动计算检验统计量和p值

检验类型说明:
- t_test: 学生t检验，用于均值比较
- adf: Augmented Dickey-Fuller检验，用于时间序列平稳性

数据要求:
- 单样本检验: 只需提供data参数
- 双样本检验: 提供data和data2参数
- 时间序列检验: 使用adf检验类型""",
    field_descriptions={
        "file_path": "数据文件路径。支持单变量或多变量数据",
        "file_content": "文件内容字符串。适合小数据集",
        "file_format": "文件格式: csv/json/auto",
        "data": "第一组数据或单样本数据，数值列表格式",
        "data2": "第二组数据(双样本检验时使用)，数值列表格式",
        "test_type": "检验类型: t_test(均值检验)/adf(平稳性检验)"
    },
    examples=[
        "检验两组学生的考试成绩是否有显著差异",
        "检验GDP时间序列是否平稳",
        "比较新旧营销策略的效果差异"
    ],
    use_cases=[
        "A/B测试结果验证",
        "时间序列平稳性分析",
        "均值差异显著性检验",
        "实验效果评估"
    ]
)

TIME_SERIES_ANALYSIS = ToolDescription(
    name="time_series_analysis",
    description="""时间序列分析

功能说明:
- 时间序列趋势分析和季节性分解
- 自相关函数(ACF)和偏自相关函数(PACF)分析
- 平稳性检验和单位根检验
- 时间序列模型诊断

分析内容:
- 趋势成分: 长期变化趋势
- 季节成分: 周期性波动
- 残差成分: 随机波动
- 自相关性: 时间依赖性分析""",
    field_descriptions={
        "file_path": "时间序列数据文件路径。支持单变量时间序列",
        "file_content": "文件内容字符串。数据应按时间顺序排列",
        "file_format": "文件格式: csv/json/auto",
        "data": "时间序列数据，数值列表格式，如 [100, 105, 110, 115, ...]"
    },
    examples=[
        "分析GDP季度数据的趋势和季节性",
        "分解股票价格的趋势和波动成分",
        "检验销售数据的自相关性"
    ],
    use_cases=[
        "经济周期分析",
        "销售预测建模",
        "金融市场分析",
        "季节性调整"
    ]
)

CORRELATION_ANALYSIS = ToolDescription(
    name="correlation_analysis",
    description="""相关性分析

功能说明:
- 计算变量间的相关系数矩阵
- 支持Pearson、Spearman、Kendall相关系数
- 可视化相关矩阵(可选)
- 显著性检验和置信区间

相关系数类型:
- pearson: 皮尔逊相关系数，衡量线性关系
- spearman: 斯皮尔曼等级相关系数，衡量单调关系
- kendall: 肯德尔等级相关系数，稳健性更好

应用场景:
- 变量关系探索
- 多重共线性检测
- 特征选择辅助""",
    field_descriptions={
        "file_path": "多变量数据文件路径。每列代表一个变量",
        "file_content": "文件内容字符串。支持多变量数据",
        "file_format": "文件格式: csv/json/auto",
        "data": "多变量数据字典，如 {'GDP': [值], 'Population': [值], ...}",
        "method": "相关系数类型: pearson(默认)/spearman/kendall"
    },
    examples=[
        "分析经济指标间的相关性",
        "研究股票收益率的相关性结构",
        "探索消费者行为变量间的关系"
    ],
    use_cases=[
        "变量关系探索",
        "多重共线性检测",
        "投资组合相关性分析",
        "市场联动性研究"
    ]
)


# ============================================================================
# 面板数据工具描述 (4个) - 优化版
# ============================================================================

PANEL_FIXED_EFFECTS = ToolDescription(
    name="panel_fixed_effects",
    description="""固定效应模型

功能说明:
- 处理面板数据的固定效应模型
- 控制个体固定效应和/或时间固定效应
- 消除不随时间变化的个体异质性
- 提供稳健的标准误估计

模型特点:
- 实体效应: 控制个体间不可观测的固定差异
- 时间效应: 控制时间趋势和宏观冲击
- 双向固定效应: 同时控制实体和时间效应

数据格式要求:
- 必须包含实体ID列和时间列
- 实体ID列名识别: entity_id, id, entity, firm, company, country, region
- 时间列名识别: time_period, time, date, year, month, period, quarter

适用场景:
- 个体间存在不可观测的固定差异
- 需要控制个体特异性因素
- 数据存在明显的个体效应""",
    field_descriptions={
        "file_path": "CSV文件路径。必须包含实体ID列和时间列，支持自动识别列名",
        "file_content": "文件内容字符串。CSV格式，包含实体ID、时间和变量列",
        "file_format": "文件格式: csv/auto。面板数据推荐使用CSV格式",
        "y_data": "因变量数据列表，数值格式",
        "x_data": "自变量数据矩阵，二维列表格式",
        "entity_ids": "实体ID列表，字符串格式，如 ['A', 'B', 'C', ...]",
        "time_periods": "时间周期列表，字符串格式，如 ['2020', '2021', '2022', ...]",
        "feature_names": "自变量名称列表，如 ['Investment', 'Employment', 'R&D']",
        "entity_effects": "是否包含实体固定效应，默认True",
        "time_effects": "是否包含时间固定效应，默认False"
    },
    examples=[
        "分析不同公司的研发投入对利润的影响(控制公司固定效应)",
        "研究各国教育支出对经济增长的影响(控制国家固定效应)",
        "评估政策改革效果(控制个体和时间双向效应)"
    ],
    use_cases=[
        "公司财务面板数据分析",
        "国家宏观经济面板研究",
        "政策评估和效果分析",
        "个体异质性控制"
    ]
)

PANEL_RANDOM_EFFECTS = ToolDescription(
    name="panel_random_effects",
    description="""随机效应模型

功能说明:
- 处理面板数据的随机效应模型
- 假设个体效应与解释变量不相关
- 更有效地利用数据信息
- 适用于大样本面板数据

模型特点:
- 个体效应被视为随机变量
- 允许个体间存在相关性
- 比固定效应模型更高效
- 可以估计不随时间变化的变量

与固定效应比较:
- 随机效应假设更强(个体效应与解释变量不相关)
- 固定效应更稳健但损失信息
- 可通过Hausman检验选择合适模型

数据要求:
- 与固定效应模型相同的数据格式
- 需要实体ID和时间信息""",
    field_descriptions={
        "file_path": "CSV文件路径。必须包含实体ID列和时间列，支持自动识别",
        "file_content": "文件内容字符串。CSV格式的面板数据",
        "file_format": "文件格式: csv/auto",
        "y_data": "因变量数据列表，数值格式",
        "x_data": "自变量数据矩阵，二维列表格式",
        "entity_ids": "实体ID列表，字符串格式",
        "time_periods": "时间周期列表，字符串格式",
        "feature_names": "自变量名称列表",
        "entity_effects": "是否包含实体随机效应，默认True",
        "time_effects": "是否包含时间随机效应，默认False"
    },
    examples=[
        "分析企业特征对绩效的影响(假设企业效应随机)",
        "研究家庭特征对消费行为的影响",
        "评估地区特征对经济发展的影响"
    ],
    use_cases=[
        "大样本面板数据分析",
        "个体效应与解释变量相关性较弱的情况",
        "需要估计不随时间变化变量的影响",
        "效率优先的分析场景"
    ]
)

PANEL_HAUSMAN_TEST = ToolDescription(
    name="panel_hausman_test",
    description="""Hausman检验

功能说明:
- 检验固定效应模型与随机效应模型的选择
- 基于模型估计差异的统计检验
- 帮助选择合适的面板数据模型
- 提供检验统计量和p值

检验原理:
- 零假设: 随机效应模型是合适的
- 备择假设: 固定效应模型更合适
- 检验统计量服从卡方分布

决策规则:
- p值 < 0.05: 拒绝零假设，选择固定效应模型
- p值 >= 0.05: 不拒绝零假设，选择随机效应模型

数据要求:
- 与固定效应和随机效应模型相同的数据格式""",
    field_descriptions={
        "file_path": "CSV文件路径。必须包含实体ID列和时间列",
        "file_content": "文件内容字符串。面板数据格式",
        "file_format": "文件格式: csv/auto",
        "y_data": "因变量数据列表",
        "x_data": "自变量数据矩阵",
        "entity_ids": "实体ID列表",
        "time_periods": "时间周期列表",
        "feature_names": "自变量名称列表"
    },
    examples=[
        "检验企业面板数据应该使用固定效应还是随机效应模型",
        "选择国家面板数据的合适模型形式",
        "确定个体效应是否与解释变量相关"
    ],
    use_cases=[
        "面板数据模型选择",
        "固定效应与随机效应比较",
        "模型设定检验",
        "实证研究模型验证"
    ]
)

PANEL_UNIT_ROOT_TEST = ToolDescription(
    name="panel_unit_root_test",
    description="""面板单位根检验

功能说明:
- 检验面板数据中变量的平稳性
- 支持多种面板单位根检验方法
- 检测面板数据的非平稳性
- 为面板协整分析提供基础

检验类型:
- levinlin: Levin-Lin-Chu检验，假设共同单位根过程
- 其他方法: 支持多种面板单位根检验

数据要求:
- 至少3个实体，每个实体至少5个时间点
- 平衡或不平衡面板数据
- 包含实体ID和时间信息

应用意义:
- 平稳性是面板数据建模的前提
- 非平稳数据可能导致伪回归
- 为面板协整分析做准备""",
    field_descriptions={
        "file_path": "CSV文件路径。必须包含实体ID列和时间列，数据量要求: 至少3个实体，每个实体至少5个时间点",
        "file_content": "文件内容字符串。面板数据格式，满足最小数据量要求",
        "file_format": "文件格式: csv/auto",
        "data": "时间序列数据，单变量格式",
        "y_data": "因变量数据(从面板数据转换)",
        "x_data": "自变量数据(从面板数据转换，通常忽略)",
        "entity_ids": "实体ID列表，用于识别不同个体",
        "time_periods": "时间周期列表，用于时间维度识别",
        "feature_names": "特征名称(从面板数据转换，通常忽略)",
        "test_type": "检验类型: levinlin(默认)/其他面板单位根检验方法"
    },
    examples=[
        "检验各国GDP面板数据是否平稳",
        "检测公司股价面板数据的单位根",
        "验证宏观经济面板变量的平稳性"
    ],
    use_cases=[
        "面板数据平稳性检验",
        "面板协整分析前提检验",
        "面板数据建模前的诊断",
        "非平稳面板数据处理"
    ]
)


# ============================================================================
# 高级时间序列工具描述 (5个)
# ============================================================================

VAR_MODEL_ANALYSIS = ToolDescription(
    name="var_model_analysis",
    description="""VAR模型分析

功能说明:
- 向量自回归(VAR)模型分析
- 分析多个时间序列变量间的动态关系
- 估计变量间的相互影响和滞后效应
- 进行脉冲响应分析和预测

模型特点:
- 多变量时间序列建模
- 变量间相互影响分析
- 滞后效应估计
- 动态系统建模

信息准则类型:
- aic: Akaike信息准则
- bic: Bayesian信息准则
- hqic: Hannan-Quinn信息准则

应用场景:
- 宏观经济变量联动分析
- 金融市场波动传导
- 多变量预测建模""",
    field_descriptions={
        "file_path": "多变量时间序列文件路径。支持CSV/JSON格式",
        "file_content": "文件内容字符串。多变量时间序列数据",
        "file_format": "文件格式: csv/json/auto",
        "data": "多变量时间序列数据字典，如 {'GDP': [值], 'CPI': [值], ...}",
        "max_lags": "最大滞后阶数，用于模型阶数选择，默认5",
        "ic": "信息准则类型: aic(默认)/bic/hqic，用于最优滞后阶数选择"
    },
    examples=[
        "分析GDP、CPI、利率等宏观经济变量的动态关系",
        "研究股票市场、债券市场、外汇市场的联动效应",
        "预测多变量经济系统的未来走势"
    ],
    use_cases=[
        "宏观经济政策分析",
        "金融市场联动研究",
        "多变量预测建模",
        "动态系统分析"
    ]
)

VECM_MODEL_ANALYSIS = ToolDescription(
    name="vecm_model_analysis",
    description="""VECM模型分析

功能说明:
- 向量误差修正模型(VECM)分析
- 处理非平稳时间序列的协整关系
- 估计长期均衡关系和短期调整机制
- 分析变量间的长期和短期动态

模型特点:
- 基于VAR模型的协整扩展
- 长期均衡关系建模
- 短期调整机制分析
- 误差修正项估计

确定性项类型:
- co: 常数项在协整关系中
- c: 常数项在VAR中
- ct: 常数项和趋势项
- none: 无确定性项

应用场景:
- 非平稳时间序列的长期关系分析
- 经济变量的均衡关系研究
- 协整系统的动态调整分析""",
    field_descriptions={
        "file_path": "多变量时间序列文件路径。支持非平稳时间序列",
        "file_content": "文件内容字符串。多变量非平稳时间序列",
        "file_format": "文件格式: csv/json/auto",
        "data": "多变量时间序列数据字典",
        "coint_rank": "协整秩，表示协整关系数量，默认1",
        "deterministic": "确定性项类型: co(默认)/c/ct/none",
        "max_lags": "最大滞后阶数，用于模型估计，默认5"
    },
    examples=[
        "分析GDP和消费的长期均衡关系",
        "研究汇率和利率的协整关系",
        "估计股票价格和交易量的误差修正机制"
    ],
    use_cases=[
        "非平稳时间序列建模",
        "长期均衡关系分析",
        "协整系统动态研究",
        "经济变量均衡分析"
    ]
)

GARCH_MODEL_ANALYSIS = ToolDescription(
    name="garch_model_analysis",
    description="GARCH模型分析 - 支持文件输入",
    field_descriptions={
        "file_path": "文件路径",
        "file_content": "文件内容",
        "file_format": "文件格式",
        "data": "时间序列数据",
        "order": "GARCH模型阶数",
        "dist": "误差分布类型"
    }
)

STATE_SPACE_MODEL_ANALYSIS = ToolDescription(
    name="state_space_model_analysis",
    description="状态空间模型分析 - 支持文件输入",
    field_descriptions={
        "file_path": "文件路径",
        "file_content": "文件内容",
        "file_format": "文件格式",
        "data": "时间序列数据",
        "state_dim": "状态维度",
        "observation_dim": "观测维度",
        "trend": "是否包含趋势项",
        "seasonal": "是否包含季节项",
        "period": "季节周期"
    }
)

VARIANCE_DECOMPOSITION_ANALYSIS = ToolDescription(
    name="variance_decomposition_analysis",
    description="方差分解分析 - 支持文件输入",
    field_descriptions={
        "file_path": "文件路径",
        "file_content": "文件内容",
        "file_format": "文件格式",
        "data": "多变量时间序列数据",
        "periods": "分解期数",
        "max_lags": "最大滞后阶数"
    }
)


# ============================================================================
# 机器学习工具描述 (6个)
# ============================================================================

RANDOM_FOREST_REGRESSION_ANALYSIS = ToolDescription(
    name="random_forest_regression_analysis",
    description="随机森林回归 - 支持文件输入",
    field_descriptions={
        "file_path": "文件路径",
        "file_content": "文件内容",
        "file_format": "文件格式",
        "y_data": "因变量数据",
        "x_data": "自变量数据",
        "feature_names": "特征名称",
        "n_estimators": "树的数量",
        "max_depth": "最大深度"
    }
)

GRADIENT_BOOSTING_REGRESSION_ANALYSIS = ToolDescription(
    name="gradient_boosting_regression_analysis",
    description="梯度提升树回归 - 支持文件输入",
    field_descriptions={
        "file_path": "文件路径",
        "file_content": "文件内容",
        "file_format": "文件格式",
        "y_data": "因变量数据",
        "x_data": "自变量数据",
        "feature_names": "特征名称",
        "n_estimators": "树的数量",
        "learning_rate": "学习率",
        "max_depth": "最大深度"
    }
)

LASSO_REGRESSION_ANALYSIS = ToolDescription(
    name="lasso_regression_analysis",
    description="Lasso回归 - 支持文件输入",
    field_descriptions={
        "file_path": "文件路径",
        "file_content": "文件内容",
        "file_format": "文件格式",
        "y_data": "因变量数据",
        "x_data": "自变量数据",
        "feature_names": "特征名称",
        "alpha": "正则化参数"
    }
)

RIDGE_REGRESSION_ANALYSIS = ToolDescription(
    name="ridge_regression_analysis",
    description="Ridge回归 - 支持文件输入",
    field_descriptions={
        "file_path": "文件路径",
        "file_content": "文件内容",
        "file_format": "文件格式",
        "y_data": "因变量数据",
        "x_data": "自变量数据",
        "feature_names": "特征名称",
        "alpha": "正则化参数"
    }
)

CROSS_VALIDATION_ANALYSIS = ToolDescription(
    name="cross_validation_analysis",
    description="交叉验证 - 支持文件输入",
    field_descriptions={
        "file_path": "文件路径",
        "file_content": "文件内容",
        "file_format": "文件格式",
        "y_data": "因变量数据",
        "x_data": "自变量数据",
        "feature_names": "特征名称",
        "model_type": "模型类型",
        "cv_folds": "交叉验证折数",
        "scoring": "评分指标"
    }
)

FEATURE_IMPORTANCE_ANALYSIS_TOOL = ToolDescription(
    name="feature_importance_analysis_tool",
    description="特征重要性分析 - 支持文件输入",
    field_descriptions={
        "file_path": "文件路径",
        "file_content": "文件内容",
        "file_format": "文件格式",
        "y_data": "因变量数据",
        "x_data": "自变量数据",
        "feature_names": "特征名称",
        "method": "分析方法",
        "top_k": "显示前K个重要特征"
    }
)


# ============================================================================
# 工具描述映射
# ============================================================================

TOOL_DESCRIPTIONS: Dict[str, ToolDescription] = {
    # 基础统计工具
    "descriptive_statistics": DESCRIPTIVE_STATISTICS,
    "ols_regression": OLS_REGRESSION,
    "hypothesis_testing": HYPOTHESIS_TESTING,
    "time_series_analysis": TIME_SERIES_ANALYSIS,
    "correlation_analysis": CORRELATION_ANALYSIS,
    
    # 面板数据工具
    "panel_fixed_effects": PANEL_FIXED_EFFECTS,
    "panel_random_effects": PANEL_RANDOM_EFFECTS,
    "panel_hausman_test": PANEL_HAUSMAN_TEST,
    "panel_unit_root_test": PANEL_UNIT_ROOT_TEST,
    
    # 高级时间序列工具
    "var_model_analysis": VAR_MODEL_ANALYSIS,
    "vecm_model_analysis": VECM_MODEL_ANALYSIS,
    "garch_model_analysis": GARCH_MODEL_ANALYSIS,
    "state_space_model_analysis": STATE_SPACE_MODEL_ANALYSIS,
    "variance_decomposition_analysis": VARIANCE_DECOMPOSITION_ANALYSIS,
    
    # 机器学习工具
    "random_forest_regression_analysis": RANDOM_FOREST_REGRESSION_ANALYSIS,
    "gradient_boosting_regression_analysis": GRADIENT_BOOSTING_REGRESSION_ANALYSIS,
    "lasso_regression_analysis": LASSO_REGRESSION_ANALYSIS,
    "ridge_regression_analysis": RIDGE_REGRESSION_ANALYSIS,
    "cross_validation_analysis": CROSS_VALIDATION_ANALYSIS,
    "feature_importance_analysis_tool": FEATURE_IMPORTANCE_ANALYSIS_TOOL,
}


def get_tool_description(tool_name: str) -> ToolDescription:
    """获取工具描述"""
    if tool_name not in TOOL_DESCRIPTIONS:
        raise ValueError(f"未知的工具名称: {tool_name}")
    return TOOL_DESCRIPTIONS[tool_name]


def get_all_tool_names() -> List[str]:
    """获取所有工具名称"""
    return list(TOOL_DESCRIPTIONS.keys())


def get_field_description(tool_name: str, field_name: str, default: str = "") -> str:
    """获取指定工具的字段描述"""
    tool_desc = get_tool_description(tool_name)
    return tool_desc.get_field_description(field_name, default)


# 导出主要类和函数
__all__ = [
    "ToolDescription",
    "TOOL_DESCRIPTIONS",
    "get_tool_description",
    "get_all_tool_names",
    "get_field_description",
    
    # 基础统计工具
    "DESCRIPTIVE_STATISTICS",
    "OLS_REGRESSION",
    "HYPOTHESIS_TESTING",
    "TIME_SERIES_ANALYSIS",
    "CORRELATION_ANALYSIS",
    
    # 面板数据工具
    "PANEL_FIXED_EFFECTS",
    "PANEL_RANDOM_EFFECTS",
    "PANEL_HAUSMAN_TEST",
    "PANEL_UNIT_ROOT_TEST",
    
    # 高级时间序列工具
    "VAR_MODEL_ANALYSIS",
    "VECM_MODEL_ANALYSIS",
    "GARCH_MODEL_ANALYSIS",
    "STATE_SPACE_MODEL_ANALYSIS",
    "VARIANCE_DECOMPOSITION_ANALYSIS",
    
    # 机器学习工具
    "RANDOM_FOREST_REGRESSION_ANALYSIS",
    "GRADIENT_BOOSTING_REGRESSION_ANALYSIS",
    "LASSO_REGRESSION_ANALYSIS",
    "RIDGE_REGRESSION_ANALYSIS",
    "CROSS_VALIDATION_ANALYSIS",
    "FEATURE_IMPORTANCE_ANALYSIS_TOOL",
]