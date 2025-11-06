# aigroup-econ-mcp - 专业计量经济学MCP工具

🎯 **66项专业计量经济学分析工具** - 提供完整计量功能覆盖，支持CSV/JSON/TXT/Excel多种数据格式

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![MCP](https://img.shields.io/badge/MCP-1.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/Version-2.0.1-orange.svg)
![Tools](https://img.shields.io/badge/Tools-66-brightgreen.svg)

## 📋 目录

- [🚀 快速开始](#-快速开始)
- [✨ 核心功能](#-核心功能)
- [🔧 工具列表](#-工具列表)
- [📁 文件输入支持](#-文件输入支持)
- [⚙️ 安装配置](#️-安装配置)
- [📚 使用示例](#-使用示例)
- [🔍 故障排除](#-故障排除)
- [🏗️ 项目架构](#️-项目架构)
- [🤝 贡献指南](#-贡献指南)
- [📄 许可证](#-许可证)

## 🚀 快速开始

### 一键启动（推荐）

```bash
# 使用uvx快速启动（无需安装）
uvx aigroup-econ-mcp
```

### Roo-Code、通义灵码、Claude code配置

MCP设置中添加：

```json
{
  "mcpServers": {
    "aigroup-econ-mcp": {
      "command": "uvx",
      "args": ["aigroup-econ-mcp"],
      "alwaysAllow": [
        "basic_parametric_estimation_ols", "basic_parametric_estimation_mle", "basic_parametric_estimation_gmm",
        "causal_difference_in_differences", "causal_instrumental_variables", "causal_propensity_score_matching",
        "causal_fixed_effects", "causal_random_effects", "causal_regression_discontinuity",
        "causal_synthetic_control", "causal_event_study", "causal_triple_difference",
        "causal_mediation_analysis", "causal_moderation_analysis", "causal_control_function",
        "causal_first_difference", "ml_random_forest", "ml_gradient_boosting",
        "ml_support_vector_machine", "ml_neural_network", "ml_kmeans_clustering",
        "ml_hierarchical_clustering", "ml_double_machine_learning", "ml_causal_forest",
        "micro_logit", "micro_probit", "micro_multinomial_logit",
        "micro_poisson", "micro_negative_binomial", "micro_tobit",
        "micro_heckman", "model_diagnostic_tests", "generalized_least_squares",
        "weighted_least_squares", "robust_errors_regression", "model_selection_criteria",
        "regularized_regression", "simultaneous_equations_model", "time_series_arima_model",
        "time_series_exponential_smoothing", "time_series_garch_model", "time_series_unit_root_tests",
        "time_series_var_svar_model", "time_series_cointegration_analysis", "panel_data_dynamic_model",
        "panel_data_diagnostics", "panel_var_model", "structural_break_tests",
        "time_varying_parameter_models"
      ]
    }
  }
}
```

## ✨ 核心功能 - 66项专业工具

### 1. 基础参数估计 (3项)

解决建立变量间的基础参数化关系并进行估计的问题。

- **普通最小二乘法 (OLS)** - `basic_parametric_estimation_ols`
- **最大似然估计 (MLE)** - `basic_parametric_estimation_mle`
- **广义矩估计 (GMM)** - `basic_parametric_estimation_gmm`

### 2. 因果识别策略 (13项)

在非实验数据中，识别变量间的因果关系（解决内生性问题）。

- **双重差分法 (DID)** - `causal_difference_in_differences`
- **工具变量法 (IV/2SLS)** - `causal_instrumental_variables`
- **倾向得分匹配 (PSM)** - `causal_propensity_score_matching`
- **固定效应模型** - `causal_fixed_effects`
- **随机效应模型** - `causal_random_effects`
- **回归断点设计 (RDD)** - `causal_regression_discontinuity`
- **合成控制法** - `causal_synthetic_control`
- **事件研究法** - `causal_event_study`
- **三重差分法 (DDD)** - `causal_triple_difference`
- **中介效应分析** - `causal_mediation_analysis`
- **调节效应分析** - `causal_moderation_analysis`
- **控制函数法** - `causal_control_function`
- **一阶差分模型** - `causal_first_difference`

### 3. 分解分析 (3项)

分析变量差异的来源和构成。

- **Oaxaca-Blinder分解** - `decomposition_oaxaca_blinder`
- **方差分解 (ANOVA)** - `decomposition_variance_anova`
- **时间序列分解** - `decomposition_time_series`

### 4. 机器学习方法 (8项)

处理高维数据、复杂模式识别、预测以及为因果推断提供辅助工具。

- **随机森林** - `ml_random_forest`
- **梯度提升机** - `ml_gradient_boosting`
- **支持向量机** - `ml_support_vector_machine`
- **神经网络** - `ml_neural_network`
- **K均值聚类** - `ml_kmeans_clustering`
- **层次聚类** - `ml_hierarchical_clustering`
- **双重机器学习** - `ml_double_machine_learning`
- **因果森林** - `ml_causal_forest`

### 5. 微观计量模型 (7项)

针对因变量或数据结构的固有特性进行建模。

- **Logit模型** - `micro_logit`
- **Probit模型** - `micro_probit`
- **多项Logit** - `micro_multinomial_logit`
- **泊松回归** - `micro_poisson`
- **负二项回归** - `micro_negative_binomial`
- **Tobit模型** - `micro_tobit`
- **Heckman选择模型** - `micro_heckman`

### 6. 缺失数据处理 (2项)

处理数据缺失问题，保证分析的完整性。

- **简单插补** - `missing_data_simple_imputation`
- **多重插补 (MICE)** - `missing_data_multiple_imputation`

### 7. 模型规范、诊断与稳健推断 (7项)

当基础模型的理想假设不成立时，修正模型或调整推断；对模型进行诊断和选择。

- **模型诊断检验** - `model_diagnostic_tests`
- **广义最小二乘法 (GLS)** - `generalized_least_squares`
- **加权最小二乘法 (WLS)** - `weighted_least_squares`
- **稳健标准误回归** - `robust_errors_regression`
- **模型选择准则** - `model_selection_criteria`
- **正则化回归** - `regularized_regression`
- **联立方程模型** - `simultaneous_equations_model`

### 8. 非参数方法 (4项)

不依赖特定函数形式的灵活建模方法。

- **核回归** - `nonparametric_kernel_regression`
- **分位数回归** - `nonparametric_quantile_regression`
- **样条回归** - `nonparametric_spline_regression`
- **广义可加模型 (GAM)** - `nonparametric_gam_model`

### 9. 空间计量经济学 (6项)

分析具有空间依赖性的数据。

- **空间权重矩阵** - `spatial_weights_matrix`
- **Moran's I检验** - `spatial_morans_i_test`
- **Geary's C检验** - `spatial_gearys_c_test`
- **局部Moran's I (LISA)** - `spatial_local_moran_lisa`
- **空间回归模型** - `spatial_regression_model`
- **地理加权回归 (GWR)** - `spatial_gwr_model`

### 10. 统计推断 (2项)

基于重采样的统计推断方法。

- **Bootstrap方法** - `inference_bootstrap`
- **置换检验** - `inference_permutation_test`

### 11. 时间序列与面板数据 (11项)

分析具有时间维度数据的动态依赖、预测和非平稳性。

- **ARIMA模型** - `time_series_arima_model`
- **指数平滑法** - `time_series_exponential_smoothing`
- **GARCH波动率模型** - `time_series_garch_model`
- **单位根检验** - `time_series_unit_root_tests`
- **VAR/SVAR模型** - `time_series_var_svar_model`
- **协整分析** - `time_series_cointegration_analysis`
- **动态面板模型** - `panel_data_dynamic_model`
- **面板数据诊断** - `panel_data_diagnostics`
- **面板VAR模型** - `panel_var_model`
- **结构突变检验** - `structural_break_tests`
- **时变参数模型** - `time_varying_parameter_models`

## 🔧 完整工具列表 (66项)

### 基础参数估计 (3项)

| 工具                                | 功能         | 主要参数                      | 输出                              |
| ----------------------------------- | ------------ | ----------------------------- | --------------------------------- |
| `basic_parametric_estimation_ols` | OLS回归分析  | y_data, x_data, file_path     | R²、系数、t统计量、p值、置信区间 |
| `basic_parametric_estimation_mle` | 最大似然估计 | data, file_path, distribution | 参数估计、标准误、置信区间        |
| `basic_parametric_estimation_gmm` | 广义矩估计   | y_data, x_data, instruments   | GMM系数、J统计量、p值             |

### 因果推断 (13项)

| 工具                                 | 功能         | 主要参数                               | 输出                    |
| ------------------------------------ | ------------ | -------------------------------------- | ----------------------- |
| `causal_difference_in_differences` | 双重差分法   | treatment, time_period, outcome        | 处理效应、时间效应      |
| `causal_instrumental_variables`    | 工具变量法   | y_data, x_data, instruments            | 2SLS系数、弱工具检验    |
| `causal_propensity_score_matching` | 倾向得分匹配 | treatment, outcome, covariates         | 处理效应、匹配统计      |
| `causal_fixed_effects`             | 固定效应模型 | y_data, x_data, entity_ids             | R²、系数、F统计量      |
| `causal_random_effects`            | 随机效应模型 | y_data, x_data, entity_ids             | R²、系数、随机效应方差 |
| `causal_regression_discontinuity`  | 回归断点设计 | running_variable, outcome, cutoff      | 局部平均处理效应        |
| `causal_synthetic_control`         | 合成控制法   | outcome, treatment_period, donor_units | 合成权重、处理效应      |
| `causal_event_study`               | 事件研究法   | outcome, treatment, event_time         | 动态处理效应            |
| `causal_triple_difference`         | 三重差分法   | outcome, treatment_group, cohort_group | 三重差分效应            |
| `causal_mediation_analysis`        | 中介效应分析 | outcome, treatment, mediator           | 直接效应、间接效应      |
| `causal_moderation_analysis`       | 调节效应分析 | outcome, predictor, moderator          | 交互效应、条件效应      |
| `causal_control_function`          | 控制函数法   | y_data, x_data, z_data                 | 控制函数估计            |
| `causal_first_difference`          | 一阶差分模型 | y_data, x_data, entity_ids             | 差分系数、标准误        |

### 机器学习 (8项)

| 工具                           | 功能         | 主要参数                           | 输出                       |
| ------------------------------ | ------------ | ---------------------------------- | -------------------------- |
| `ml_random_forest`           | 随机森林     | X_data, y_data, problem_type       | R²、特征重要性、预测精度  |
| `ml_gradient_boosting`       | 梯度提升机   | X_data, y_data, algorithm          | R²、特征重要性、预测精度  |
| `ml_support_vector_machine`  | 支持向量机   | X_data, y_data, kernel             | R²、支持向量、预测精度    |
| `ml_neural_network`          | 神经网络     | X_data, y_data, hidden_layer_sizes | R²、网络权重、预测精度    |
| `ml_kmeans_clustering`       | K均值聚类    | X_data, n_clusters                 | 聚类中心、簇标签、轮廓系数 |
| `ml_hierarchical_clustering` | 层次聚类     | X_data, n_clusters, linkage        | 聚类树、簇标签             |
| `ml_double_machine_learning` | 双重机器学习 | X_data, y_data, d_data             | 处理效应、置信区间         |
| `ml_causal_forest`           | 因果森林     | X_data, y_data, w_data             | 异质性处理效应、特征重要性 |

### 微观计量 (7项)

| 工具                        | 功能            | 主要参数                      | 输出                       |
| --------------------------- | --------------- | ----------------------------- | -------------------------- |
| `micro_logit`             | Logit回归       | X_data, y_data                | 伪R²、系数、OR值、p值     |
| `micro_probit`            | Probit回归      | X_data, y_data                | 伪R²、系数、边际效应、p值 |
| `micro_multinomial_logit` | 多项Logit       | X_data, y_data                | 伪R²、系数、相对风险比    |
| `micro_poisson`           | 泊松回归        | X_data, y_data                | 伪R²、系数、发生率比      |
| `micro_negative_binomial` | 负二项回归      | X_data, y_data, distr         | 伪R²、系数、过度离散参数  |
| `micro_tobit`             | Tobit模型       | X_data, y_data, bounds        | 系数、边际效应、p值        |
| `micro_heckman`           | Heckman选择模型 | X_select_data, Z_data, s_data | 选择方程、结果方程系数     |

### 模型规范与诊断 (7项)

| 工具                             | 功能         | 主要参数                    | 输出                            |
| -------------------------------- | ------------ | --------------------------- | ------------------------------- |
| `model_diagnostic_tests`       | 模型诊断检验 | y_data, x_data              | 异方差、自相关、正态性、VIF检验 |
| `generalized_least_squares`    | GLS回归      | y_data, x_data, sigma       | GLS系数、标准误、置信区间       |
| `weighted_least_squares`       | WLS回归      | y_data, x_data, weights     | WLS系数、权重统计               |
| `robust_errors_regression`     | 稳健标准误   | y_data, x_data, cov_type    | 稳健标准误、检验统计量          |
| `model_selection_criteria`     | 模型选择     | y_data, x_data, cv_folds    | AIC、BIC、HQIC、交叉验证        |
| `regularized_regression`       | 正则化回归   | y_data, x_data, method      | 正则化系数、特征选择            |
| `simultaneous_equations_model` | 联立方程模型 | y_data, x_data, instruments | 2SLS系数、方程系统              |

### 时间序列与面板数据 (11项)

| 工具                                   | 功能         | 主要参数                   | 输出                       |
| -------------------------------------- | ------------ | -------------------------- | -------------------------- |
| `time_series_arima_model`            | ARIMA模型    | data, order                | 模型系数、预测值、置信区间 |
| `time_series_exponential_smoothing`  | 指数平滑     | data, trend, seasonal      | 平滑参数、预测值           |
| `time_series_garch_model`            | GARCH模型    | data, order                | 波动率参数、条件方差       |
| `time_series_unit_root_tests`        | 单位根检验   | data, test_type            | 检验统计量、平稳性判断     |
| `time_series_var_svar_model`         | VAR/SVAR模型 | data, model_type, lags     | 系数矩阵、脉冲响应         |
| `time_series_cointegration_analysis` | 协整分析     | data, analysis_type        | 协整向量、秩检验           |
| `panel_data_dynamic_model`           | 动态面板模型 | y_data, x_data, entity_ids | GMM系数、标准误            |
| `panel_data_diagnostics`             | 面板诊断     | test_type, residuals       | Hausman检验、F检验、LM检验 |
| `panel_var_model`                    | 面板VAR模型  | data, entity_ids, lags     | 面板VAR系数、脉冲响应      |
| `structural_break_tests`             | 结构突变检验 | data, test_type            | 断点检测、检验统计量       |
| `time_varying_parameter_models`      | 时变参数模型 | y_data, x_data, model_type | 参数轨迹、机制转换         |

> **注意**: 所有工具均支持CSV/JSON/TXT/Excel格式输入，可通过 `file_path`、`file_content`或直接数据参数调用。**输出支持JSON/Markdown/TXT多种格式**。

## 📁 文件输入支持

### 支持的文件格式

#### 1. CSV文件（推荐）

- **格式**: 逗号、制表符、分号分隔
- **表头**: 自动识别（第一行非数值为表头）
- **特点**: 最通用，易于编辑和查看

```
GDP,CPI,失业率
3.2,2.1,4.5
2.8,2.3,4.2
3.5,1.9,4.0
```

#### 2. JSON文件

- **字典格式**: `{"变量名": [数据], ...}`
- **数组格式**: `[{"变量1": 值, ...}, ...]`
- **嵌套格式**: `{"data": {...}, "metadata": {...}}`

```json
{
  "GDP": [3.2, 2.8, 3.5],
  "CPI": [2.1, 2.3, 1.9],
  "失业率": [4.5, 4.2, 4.0]
}
```

#### 3. Excel文件

- **格式**: .xlsx 或 .xls
- **表头**: 第一行作为变量名
- **工作表**: 自动读取第一个工作表，或指定sheet名称
- **特点**: 支持复杂数据结构，保留格式

```
# Excel文件示例结构
# Sheet1:
#   A列: GDP, B列: CPI, C列: 失业率
#   第1行: 3.2, 2.1, 4.5
#   第2行: 2.8, 2.3, 4.2
#   第3行: 3.5, 1.9, 4.0
```

#### 4. TXT文件

- **单列数值**: 每行一个数值

```
100.5
102.3
101.8
103.5
```

- **多列数值**: 空格或制表符分隔

```
GDP CPI 失业率
3.2 2.1 4.5
2.8 2.3 4.2
3.5 1.9 4.0
```

- **键值对格式**: 变量名: 值列表

```
GDP: 3.2 2.8 3.5 2.9
CPI: 2.1 2.3 1.9 2.4
失业率: 4.5 4.2 4.0 4.3
```

### 使用方式

#### 方式1：直接数据输入（程序化调用）

```
{
  "data": {
    "GDP增长率": [3.2, 2.8, 3.5, 2.9],
    "通货膨胀率": [2.1, 2.3, 1.9, 2.4]
  }
}
```

 方式2：文件内容输入（字符串）

```
{
  "file_content": "GDP,CPI\n3.2,2.1\n2.8,2.3\n3.5,1.9",
  "file_format": "csv"
}
```

#### 方式3：文件路径输入（推荐✨）

```
{
  "file_path": "./data/economic_data.csv"
}
```

或使用Excel文件：

```
{
  "file_path": "./data/panel_data.xlsx"
}
```

### 输出格式支持

所有工具支持多种输出格式，通过 `output_format` 参数指定：

- **json** (默认) - 结构化JSON格式，便于程序处理
- **markdown** - Markdown表格格式，适合文档展示
- **html** - HTML表格格式，适合网页展示
- **latex** - LaTeX表格格式，适合学术论文
- **text** - 纯文本格式，简洁易读

```
{
  "file_path": "./data/economic_data.csv",
  "output_format": "json"
}
```

### 自动格式检测

系统会智能检测文件格式：

1. 文件扩展名（.csv/.json/.txt/.xlsx/.xls）
2. 文件内容特征（逗号、JSON结构、纯数值、Excel二进制）
3. 建议使用 `"file_format": "auto"` 让系统自动识别

## ⚙️ 安装配置

### 跨平台兼容性

✅ **完全跨平台支持** - 支持 Windows、macOS、Linux 系统
✅ **纯Python实现** - 无平台特定依赖
✅ **ARM架构支持** - 兼容 Apple Silicon (M1/M2/M3)

### 方式1：uvx安装（推荐）

```
# 直接运行最新版本
uvx aigroup-econ-mcp

# 指定版本
uvx aigroup-econ-mcp@2.0.0
```

### 方式2：pip安装

```
# 安装包
pip install aigroup-econ-mcp

# 运行服务
aigroup-econ-mcp
```

### macOS 特定说明

```
# 如果遇到权限问题，使用用户安装
pip install --user aigroup-econ-mcp

# 或者使用虚拟环境
python -m venv econ_env
source econ_env/bin/activate
pip install aigroup-econ-mcp
```

### 依赖说明

- **核心依赖**: pandas >= 1.5.0, numpy >= 1.21.0, scipy >= 1.7.0
- **统计分析**: statsmodels >= 0.13.0
- **面板数据**: linearmodels >= 7.0
- **机器学习**: scikit-learn >= 1.0.0, xgboost >= 1.7.0, joblib >= 1.2.0
- **时间序列**: arch >= 6.0.0
- **空间计量**: libpysal >= 4.7.0, esda >= 2.4.0, spreg >= 1.4.0
- **可视化**: matplotlib >= 3.5.0
- **轻量级**: 无需torch或其他重型框架

## 📚 使用示例

### 示例1：OLS回归分析

```
# 使用文件路径
result = await basic_parametric_estimation_ols(
    file_path="./data/economic_indicators.csv"
)

# 使用直接数据输入
result = await basic_parametric_estimation_ols(
    y_data=[12, 13, 15, 18, 20],
    x_data=[[100, 50], [120, 48], [110, 52], [130, 45], [125, 47]],
    feature_names=["广告支出", "价格"]
)
```

### 示例2：因果推断 - 双重差分法

```
result = await causal_difference_in_differences(
    treatment=[0, 0, 1, 1],
    time_period=[0, 1, 0, 1],
    outcome=[10, 12, 11, 15],
    output_format="json"
)
```

### 示例3：机器学习 - 随机森林

```
result = await ml_random_forest(
    X_data=[[100, 50, 3], [120, 48, 3], [110, 52, 4], [130, 45, 3]],
    y_data=[12, 13, 15, 18],
    feature_names=["广告支出", "价格", "竞争对手数"],
    problem_type="regression",
    n_estimators=100
)
```

### 示例4：时间序列 - ARIMA模型

```
result = await time_series_arima_model(
    data=[100.5, 102.3, 101.8, 103.5, 104.2],
    order=(1, 1, 1),
    forecast_steps=5
)
```

### 示例5：微观计量 - Logit回归

```
result = await micro_logit(
    X_data=[[1.5, 2.5], [1.7, 2.7], [1.9, 2.9], [2.1, 3.1]],
    y_data=[0, 0, 1, 1],
    feature_names=["收入", "教育年限"]
)
```

## 🔍 故障排除

### 常见问题

#### Q: uvx安装卡住

```
# 清除缓存重试
uvx --no-cache aigroup-econ-mcp
```

#### Q: 工具返回错误

- ✅ 检查数据格式（CSV/JSON/TXT）
- ✅ 确保没有缺失值（NaN）
- ✅ 验证数据类型（所有数值必须是浮点数）
- ✅ 查看详细错误信息

#### Q: MCP服务连接失败

- ✅ 检查网络连接
- ✅ 确保Python版本 >= 3.8
- ✅ 查看VSCode输出面板的详细日志
- ✅ 尝试重启RooCode

### 数据要求

| 分析类型   | 最小样本量 | 推荐样本量  | 特殊要求          |
| ---------- | ---------- | ----------- | ----------------- |
| 描述性统计 | 5          | 20+         | 无缺失值          |
| OLS回归    | 变量数+2   | 30+         | 无多重共线性      |
| 时间序列   | 10         | 40+         | 时间顺序，等间隔  |
| 面板数据   | 实体数×3  | 实体数×10+ | 平衡或非平衡面板  |
| 机器学习   | 20         | 100+        | 训练集/测试集分割 |

## 🏗️ 项目架构

### 核心模块结构

```
aigroup-econ-mcp/
├── econometrics/              # 核心计量经济学算法
│   ├── basic_parametric_estimation/    # 基础参数估计（3个模型）
│   ├── causal_inference/               # 因果推断（13个方法）
│   ├── advanced_methods/               # 机器学习（8个模型）
│   ├── specific_data_modeling/         # 微观+时序（18个模型）
│   └── model_specification_diagnostics_robust_inference/  # 模型规范（7个工具）
├── tools/                     # MCP工具适配器
│   ├── mcp_tool_groups/              # 工具组定义
│   │   ├── basic_parametric_tools.py         # 基础参数估计工具
│   │   ├── causal_inference_tools.py         # 因果推断工具
│   │   ├── machine_learning_tools.py         # 机器学习工具
│   │   ├── microecon_tools.py                # 微观计量工具
│   │   ├── model_specification_tools.py      # 模型规范工具
│   │   └── time_series_tools.py              # 时间序列工具
│   ├── data_loader.py                # 数据加载器
│   └── output_formatter.py           # 输出格式化
└── server.py                  # MCP服务器入口
```

### 设计特点

- **🎯 十一大工具组** - 基础参数估计(3) + 因果推断(13) + 分解分析(3) + 机器学习(8) + 微观计量(7) + 缺失数据处理(2) + 模型规范诊断(7) + 非参数方法(4) + 空间计量(6) + 统计推断(2) + 时序面板(11) = 66项工具
- **🔄 统一接口** - 所有工具支持CSV/JSON/TXT/Excel四种格式输入
- **📊 多格式输出** - 支持JSON/Markdown/TXT三种输出格式
- **⚡ 异步处理** - 基于asyncio的异步设计，支持并发请求
- **🛡️ 错误处理** - 统一的错误处理和详细的错误信息
- **📝 完整文档** - 每个工具都有详细的参数说明和使用示例
- **🧪 全面测试** - 单元测试和集成测试覆盖

### 新增特性

- 🎯 **66项专业工具** - 完整覆盖计量经济学核心方法
- ✨ **11大工具组** - 基础参数估计(3) + 因果推断(13) + 分解分析(3) + 机器学习(8) + 微观计量(7) + 缺失数据处理(2) + 模型规范诊断(7) + 非参数方法(4) + 空间计量(6) + 统计推断(2) + 时序面板(11)
- 🔬 **13种因果方法** - DID、IV、PSM、RDD、合成控制等完整因果推断工具链
- 📊 **8种机器学习** - 随机森林、梯度提升、神经网络、聚类、因果森林等
- ⚙️ **7种微观模型** - Logit、Probit、Tobit、Heckman等离散选择和受限因变量模型
- 📈 **11种时序模型** - ARIMA、GARCH、VAR、协整、动态面板等时间序列工具
- ✨ **多格式输入** - 支持CSV/JSON/TXT/Excel(.xlsx/.xls)四种输入格式
- 📊 **多格式输出** - 支持JSON/Markdown/TXT三种输出格式
- 📝 **完善参数描述** - 所有66个工具的MCP参数都有详细说明
- 🔍 **智能格式检测** - 自动识别CSV/JSON/TXT/Excel格式
- 📂 **文件路径支持** - 支持直接传入文件路径（.txt/.csv/.json/.xlsx/.xls）

## 🤝 贡献指南

### 开发环境设置

```
# 克隆项目
git clone https://github.com/jackdark425/aigroup-econ-mcp
cd aigroup-econ-mcp

# 安装所有依赖（包括新添加的空间计量、生存分析等包）
uv sync

# 安装开发依赖
uv add --dev pytest pytest-asyncio black isort mypy ruff

# 运行测试
uv run pytest

# 代码格式化
uv run black src/
uv run isort src/
```

### 提交贡献

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 代码规范

- 遵循PEP 8编码规范
- 使用类型注解（Type Hints）
- 添加单元测试（覆盖率>80%）
- 更新相关文档和示例

## 📄 许可证

MIT License - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- **Model Context Protocol (MCP)** - 模型上下文协议框架
- **Roo-Code** - 强大的AI编程助手
- **statsmodels** - 专业的统计分析库
- **pandas** - 高效的数据处理库
- **scikit-learn** - 全面的机器学习库
- **linearmodels** - 面板数据分析专用库
- **计量经济学社区** - 提供方法参考和实现指导
- **开源社区** - 所有依赖库的开发者们

## 📞 支持

- 💬 **GitHub Issues**: [提交问题](https://github.com/jackdark425/aigroup-econ-mcp/issues)
- 📧 **邮箱**: jackdark425@gmail.com
- 📚 **文档**: 查看[详细文档](https://github.com/jackdark425/aigroup-econ-mcp/tree/main/docs)
- 🌟 **Star项目**: 如果觉得有用，请给个⭐️

## 📈 工具统计

**总计 66 项专业工具**:

- 基础参数估计: 3项
- 因果推断: 13项
- 分解分析: 3项
- 机器学习: 8项
- 微观计量: 7项
- 缺失数据处理: 2项
- 模型规范诊断: 7项
- 非参数方法: 4项
- 空间计量: 6项
- 统计推断: 2项
- 时间序列与面板数据: 11项

---

**立即开始**: `uvx aigroup-econ-mcp` 🚀

让AI大模型成为你的专业计量经济学分析助手！66项专业工具，一站式解决方案！
