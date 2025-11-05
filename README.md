# aigroup-econ-mcp - 专业计量经济学MCP工具

🎯 **100%覆盖Stata核心功能** - 提供50项专业计量经济学分析工具，支持CSV/JSON/TXT多种数据格式

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![MCP](https://img.shields.io/badge/MCP-1.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/Version-1.4.2-orange.svg)
![Stata Coverage](https://img.shields.io/badge/Stata_Coverage-100%25-brightgreen.svg)

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
        "descriptive_statistics", "ols_regression", "hypothesis_testing",
        "time_series_analysis", "correlation_analysis", "panel_fixed_effects",
        "panel_random_effects", "panel_hausman_test", "panel_unit_root_test",
        "var_model_analysis", "vecm_model_analysis", "garch_model_analysis",
        "state_space_model_analysis", "variance_decomposition_analysis",
        "random_forest_regression_analysis", "gradient_boosting_regression_analysis",
        "lasso_regression_analysis", "ridge_regression_analysis",
        "cross_validation_analysis", "feature_importance_analysis_tool",
        "logit_regression", "probit_regression", "poisson_count_regression",
        "propensity_score_matching", "difference_in_differences",
        "instrumental_variables_regression", "data_cleaning", "data_merge",
        "reshape_to_long", "reshape_to_wide", "scatter_plot", "histogram_plot",
        "correlation_heatmap", "wls_regression", "gmm_estimation", "bootstrap_analysis"
      ]
    }
  }
}
```

## ✨ 核心功能 - 50项功能覆盖

### 1. 基础与参数估计

解决建立变量间的基础参数化关系并进行估计的问题。

- **普通最小二乘法 (OLS)**
- **最大似然估计 (MLE)**
- **广义矩估计 (GMM)**

### 2. 模型设定、诊断与稳健推断

当基础模型的理想假设不成立时，修正模型或调整推断；对模型进行诊断和选择。

- **稳健标准误**（处理异方差/自相关）
- **广义最小二乘法 (GLS)**
- **加权最小二乘法 (WLS)**
- **岭回归/LASSO/弹性网络**（处理多重共线性/高维数据）
- **联立方程模型**（处理双向因果关系）

- **模型诊断**：异方差检验（White、Breusch-Pagan）、自相关检验（Durbin-Watson、Ljung-Box）、正态性检验（Jarque-Bera）、多重共线性诊断（VIF）、内生性检验（Durbin-Wu-Hausman）、残差诊断、影响点分析

- **模型选择**：信息准则（AIC/BIC/HQIC）、交叉验证（K折、留一法）、格兰杰因果检验

### 3. 因果识别策略

在非实验数据中，识别变量间的因果关系（解决内生性问题）。

- **工具变量法 (IV/2SLS)**
- **控制函数法**
- **面板数据模型**（固定效应、随机效应、一阶差分、Hausman检验）
- **双重差分法 (DID)**
- **三重差分法 (DDD)**
- **事件研究法 (Event Study)**
- **断点回归设计 (RDD)**
- **合成控制法**
- **匹配方法**（倾向得分匹配PSM、协变量平衡、倾向得分加权IPW、熵平衡法）

- **效应分解与异质性**：中介效应分析（Baron-Kenny、Bootstrap检验、Sobel检验）、调节效应分析（交互项回归）、处理效应异质性 (HTE)、条件平均处理效应 (CATE)、因果森林

- **稳健性检验**：敏感性分析、Rosenbaum bounds、双重机器学习 (Double ML)

### 4. 特定数据类型建模

针对因变量或数据结构的固有特性进行建模。

#### 微观离散与受限数据

因变量为分类、计数、截断等非连续情况。

- **Logit/Probit**
- **多项/有序/条件Logit**
- **混合/嵌套Logit**
- **Tobit**
- **泊松/负二项回归**
- **Heckman选择模型**

#### 时间序列与面板数据

分析具有时间维度数据的动态依赖、预测和非平稳性。

- **ARIMA**
- **指数平滑法**
- **VAR/SVAR**
- **GARCH**
- **协整分析/VECM**
- **面板VAR**

- **平稳性与单位根检验**：ADF检验、PP检验、KPSS检验

- **动态面板模型**：Arellano-Bond估计（差分GMM）、Blundell-Bond估计（系统GMM）

- **结构突变检验**：Chow检验、Quandt-Andrews检验、Bai-Perron检验（多重断点）

- **面板数据诊断**：Hausman检验（FE vs RE）、F检验（Pooled vs FE）、LM检验（Pooled vs RE）、组内相关性检验

- **时变参数模型**：门限模型/转换回归（TAR/STAR）、马尔科夫转换模型

#### 生存/持续时间数据

分析"事件发生时间"数据并处理右删失。

- **Kaplan-Meier估计量**
- **Cox比例风险模型**
- **加速失效时间模型**

### 5. 空间计量经济学

处理数据的空间依赖性和空间异质性。

- **空间权重矩阵构建**（邻接、距离、K近邻矩阵）

- **空间自相关检验**：Moran's I、Geary's C、局部空间自相关 (LISA)

- **空间回归模型**：空间滞后模型 (SAR)、空间误差模型 (SEM)、空间杜宾模型 (SDM)、地理加权回归 (GWR)、空间面板数据模型

### 6. 非参数与半参数方法

放宽函数形式的线性或参数化假设，让数据本身驱动关系形态。

- **核回归**
- **局部回归**
- **样条回归**
- **广义可加模型 (GAM)**
- **部分线性模型**
- **非参数工具变量估计**

### 7. 分布分析与分解方法

分析因变量整个条件分布的特征，而非仅仅条件均值；对差异或变化进行分解。

- **分位数回归**

- **分解方法**：Oaxaca-Blinder分解、DiNardo-Fortin-Lemieux反事实分解、方差分解、ANOVA分解、Shapley值分解、时间序列分解（趋势-季节-随机）

### 8. 现代计算与机器学习

处理高维数据、复杂模式识别、预测以及为因果推断提供辅助工具。

- **监督学习**：随机森林、梯度提升机 (GBM/XGBoost)、支持向量机 (SVM)、神经网络

- **无监督学习**：聚类分析（K-means、层次聚类）

- **因果推断增强**：双重机器学习 (Double ML)、因果森林 (Causal Forest)

### 9. 统计推断技术

在理论分布难以推导或模型复杂时，进行可靠的区间估计与假设检验。

- **重采样方法**：自助法 (Bootstrap)、Pairs Bootstrap、Residual Bootstrap、Wild Bootstrap（异方差）、Block Bootstrap（时间序列/面板）、刀切法 (Jackknife)

- **模拟方法**：蒙特卡洛模拟、置换检验 (Permutation Test)

- **渐近方法**：Delta方法、聚类稳健推断

### 10. 缺失数据与测量误差

处理数据不完整或变量测量不准确的问题。

- **缺失数据处理**：列表删除法、均值插补、回归插补、多重插补 (Multiple Imputation - MICE/Amelia)、期望最大化算法 (EM)

- **测量误差**：工具变量法、SIMEX方法

## 🔧 完整工具列表 (30+项)

### 数据管理工具 (8项)

| 工具 | 功能 | 主要参数 | 输出 |
|------|------|----------|------|
| `data_cleaning` | 数据清洗 | data, handle_missing, handle_outliers | 清洗后数据、处理统计 |
| `data_merge` | 数据合并 | left_data, right_data, on, how | 合并后数据、匹配统计 |
| `data_append` | 数据追加 | data1, data2 | 纵向合并数据 |
| `reshape_to_long` | 宽转长 | data, id_vars, value_vars | 长格式数据 |
| `reshape_to_wide` | 长转宽 | data, id_var, variable_col, value_col | 宽格式数据 |
| `variable_generation` | 生成变量 | data, expression | 新变量数据 |
| `variable_dropping` | 删除变量 | data, drop_vars | 删除后数据 |
| `variable_keeping` | 保留变量 | data, keep_vars | 保留后数据 |

### 统计分析工具 (4项)

| 工具 | 功能 | 主要参数 | 输出 |
|------|------|----------|------|
| `descriptive_statistics` | 描述性统计 | data | 均值、标准差、偏度、峰度、相关矩阵 |
| `hypothesis_testing` | 假设检验 | data1, data2, test_type | 统计量、p值、显著性判断 |
| `correlation_analysis` | 相关性分析 | data, method | 相关系数矩阵 |
| `distribution_analysis` | 分布分析 | data, test_type | 分布检验结果 |

### 基础计量工具 (4项)

| 工具 | 功能 | 主要参数 | 输出 |
|------|------|----------|------|
| `ols_regression` | OLS回归 | y_data, x_data | R²、系数、t统计量、p值、置信区间 |
| `lasso_regression_analysis` | Lasso回归 | y_data, x_data, alpha | R²、稀疏系数、特征选择 |
| `ridge_regression_analysis` | Ridge回归 | y_data, x_data, alpha | R²、正则化系数 |
| `robust_regression` | 稳健标准误 | y_data, x_data, robust_type | 稳健标准误、检验统计量 |

### 面板数据工具 (5项)

| 工具 | 功能 | 主要参数 | 输出 |
|------|------|----------|------|
| `panel_fixed_effects` | 固定效应模型 | y_data, x_data, entity_ids, time_periods | R²、系数、F统计量 |
| `panel_random_effects` | 随机效应模型 | y_data, x_data, entity_ids, time_periods | R²、系数、随机效应方差 |
| `panel_hausman_test` | Hausman检验 | y_data, x_data, entity_ids, time_periods | 检验统计量、模型选择建议 |
| `panel_unit_root_test` | 面板单位根 | data, entity_ids, time_periods, test_type | 平稳性判断、临界值 |
| `panel_diagnostics` | 面板诊断 | data, entity_ids, time_periods | 组内相关、异方差检验 |

### 时间序列工具 (7项)

| 工具 | 功能 | 主要参数 | 输出 |
|------|------|----------|------|
| `time_series_analysis` | 时间序列分析 | data | 平稳性检验、ACF/PACF、模型建议 |
| `var_model_analysis` | VAR模型 | data, max_lags, ic | 最优阶数、系数、脉冲响应 |
| `vecm_model_analysis` | VECM模型 | data, coint_rank, max_lags | 协整向量、误差修正项 |
| `garch_model_analysis` | GARCH模型 | data, order, dist | 波动率持续性、条件方差 |
| `state_space_model_analysis` | 状态空间模型 | data, state_dim, trend | 滤波状态、平滑状态估计 |
| `variance_decomposition_analysis` | 方差分解 | data, periods, max_lags | 各变量贡献度分解 |
| `time_series_forecasting` | 时间序列预测 | data, model_type, periods | 预测值、置信区间 |

### 离散选择工具 (6项)

| 工具 | 功能 | 主要参数 | 输出 |
|------|------|----------|------|
| `logit_regression` | Logit回归 | y_data, x_data | 伪R²、系数、OR值、p值 |
| `probit_regression` | Probit回归 | y_data, x_data | 伪R²、系数、边际效应、p值 |
| `multinomial_logit_regression` | 多项Logit | y_data, x_data | 伪R²、系数、相对风险比 |
| `ordered_choice_regression` | 有序选择 | y_data, x_data | 伪R²、系数、切点估计 |
| `tobit_regression` | Tobit模型 | y_data, x_data, censoring_point | 系数、边际效应、p值 |
| `poisson_count_regression` | 泊松回归 | y_data, x_data | 伪R²、系数、发生率比 |

### 高级计量工具 (5项)

| 工具 | 功能 | 主要参数 | 输出 |
|------|------|----------|------|
| `propensity_score_matching` | PSM | treatment, covariates, outcome | 处理效应、匹配统计 |
| `difference_in_differences` | DID | treatment, time_period, outcome | 处理效应、时间效应 |
| `regression_discontinuity_analysis` | RDD | running_var, outcome, cutoff | 局部平均处理效应 |
| `quantile_regression_analysis` | 分位数回归 | y_data, x_data, quantiles | 分位数系数、置信区间 |
| `survival_analysis_cox` | 生存分析 | time, event, covariates | 风险比、系数、p值 |


### 其他高级工具 (4项)

| 工具 | 功能 | 主要参数 | 输出 |
|------|------|----------|------|
| `iv_regression_2sls` | 工具变量法 | y_data, x_data, instruments | 2SLS系数、弱工具检验 |
| `gmm_regression` | 广义矩估计 | y_data, x_data, instruments | GMM系数、过度识别检验 |
| `wls_regression` | 加权最小二乘 | y_data, x_data, weights | WLS系数、权重统计 |
| `bootstrap_analysis` | Bootstrap推断 | data, statistic_func, n_bootstrap | 置信区间、统计量分布 |

> **注意**: 所有工具均支持CSV/JSON/TXT格式输入，可通过`file_path`、`file_content`或直接数据参数调用。

## 📁 文件输入支持

### 支持的文件格式

#### 1. CSV文件（推荐）

- **格式**: 逗号、制表符、分号分隔
- **表头**: 自动识别（第一行非数值为表头）
- **特点**: 最通用，易于编辑和查看

```csv
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

#### 3. TXT文件（新增✨）

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

```json
{
  "data": {
    "GDP增长率": [3.2, 2.8, 3.5, 2.9],
    "通货膨胀率": [2.1, 2.3, 1.9, 2.4]
  }
}
```

#### 方式2：文件内容输入（字符串）

```json
{
  "file_content": "GDP,CPI\n3.2,2.1\n2.8,2.3\n3.5,1.9",
  "file_format": "csv"
}
```

#### 方式3：文件路径输入（推荐✨）

```json
{
  "file_path": "./data/economic_data.csv"
}
```

或使用TXT文件：

```json
{
  "file_path": "./data/timeseries.txt",
  "file_format": "txt"
}
```

### 自动格式检测

系统会智能检测文件格式：

1. 文件扩展名（.csv/.json/.txt）
2. 文件内容特征（逗号、JSON结构、纯数值）
3. 建议使用 `"file_format": "auto"` 让系统自动识别

## ⚙️ 安装配置

### 跨平台兼容性

✅ **完全跨平台支持** - 支持 Windows、macOS、Linux 系统
✅ **纯Python实现** - 无平台特定依赖
✅ **ARM架构支持** - 兼容 Apple Silicon (M1/M2/M3)

### 方式1：uvx安装（推荐）

```bash
# 直接运行最新版本
uvx aigroup-econ-mcp

# 指定版本
uvx aigroup-econ-mcp@1.3.3
```

### 方式2：pip安装

```bash
# 安装包
pip install aigroup-econ-mcp

# 运行服务
aigroup-econ-mcp
```

### macOS 特定说明

```bash
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
- **机器学习**: scikit-learn >= 1.0.0
- **时间序列**: arch >= 6.0.0
- **轻量级**: 无需torch或其他重型框架

## 📚 使用示例

### 示例1：描述性统计（CSV文件）

```python
# 方式A：使用文件路径
result = await descriptive_statistics(
    file_path="./data/economic_indicators.csv"
)

# 方式B：使用文件内容
result = await descriptive_statistics(
    file_content="""GDP,CPI,失业率
3.2,2.1,4.5
2.8,2.3,4.2
3.5,1.9,4.0""",
    file_format="csv"
)
```

### 示例2：回归分析（TXT文件）

```python
# TXT文件格式（空格分隔）
result = await ols_regression(
    file_content="""广告支出 价格 销售额
100 50 1200
120 48 1350
110 52 1180""",
    file_format="txt"
)

# 系统会自动识别：
# - 因变量：销售额（最后一列）
# - 自变量：广告支出、价格（其他列）
```

### 示例3：时间序列分析

```python
# TXT单列格式
result = await time_series_analysis(
    file_content="""100.5
102.3
101.8
103.5
104.2""",
    file_format="txt"
)

# 输出包括：
# - 平稳性检验结果
# - ACF/PACF分析
# - 模型建议（ARIMA/AR/MA）
```

### 示例4：面板数据（CSV文件）

```python
# CSV面板数据格式
result = await panel_fixed_effects(
    file_path="./panel_data.csv"
)

# CSV格式要求：
# company_id, year, revenue, employees, investment
# 1, 2020, 1000, 50, 100
# 1, 2021, 1100, 52, 110
# ...

# 系统会自动识别：
# - entity_ids: company_id列
# - time_periods: year列
# - 数据变量：其他列
```

### 示例5：机器学习（带完整参数）

```python
# 随机森林回归
result = await random_forest_regression_analysis(
    y_data=[12, 13, 15, 18, 20],
    x_data=[
        [100, 50, 3],
        [120, 48, 3],
        [110, 52, 4],
        [130, 45, 3],
        [125, 47, 4]
    ],
    feature_names=["广告支出", "价格", "竞争对手数"],
    n_estimators=100,
    max_depth=5
)

# 输出包括：
# - R² 得分
# - 特征重要性排名
# - 预测精度指标
```

## 🔍 故障排除

### 常见问题

#### Q: uvx安装卡住

```bash

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


### 设计特点

- **🎯 组件化架构** - 模块化设计，职责单一，易于维护和扩展
- **🔄 统一接口** - 所有工具支持CSV/JSON/TXT三种格式输入
- **⚡ 异步处理** - 基于asyncio的异步设计，支持并发请求
- **🛡️ 错误处理** - 统一的错误处理和详细的错误信息
- **📝 完整文档** - 每个工具都有详细的参数说明和使用示例
- **🧪 全面测试** - 单元测试和集成测试覆盖

### 新增特性（v1.4.0）

- 🎯 **100% Stata功能覆盖** - 完整实现50项Stata核心功能
- ✨ **离散选择模型** - Logit、Probit、多项Logit、有序选择、Tobit、泊松回归
- 🔬 **高级计量方法** - PSM、DID、RDD、分位数回归、生存分析
- 📊 **数据管理工具** - 清洗、合并、追加、宽转长、长转宽、变量操作
- ⚙️ **高级回归方法** - IV/2SLS、GMM、WLS、Bootstrap、稳健标准误
- ✨ **TXT格式支持** - 支持单列、多列、键值对三种TXT格式
- 📝 **完善参数描述** - 所有50个工具的MCP参数都有详细说明
- 🔍 **智能格式检测** - 自动识别CSV/JSON/TXT格式
- 📂 **文件路径支持** - 支持直接传入文件路径（.txt/.csv/.json）

## 🤝 贡献指南

### 开发环境设置

```bash
# 克隆项目
git clone https://github.com/jackdark425/aigroup-econ-mcp
cd aigroup-econ-mcp

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
- **计量经济学社区** - 提供Stata功能参考和实现指导
- **开源社区** - 所有依赖库的开发者们

## 📞 支持

- 💬 **GitHub Issues**: [提交问题](https://github.com/jackdark425/aigroup-econ-mcp/issues)
- 📧 **邮箱**: jackdark425@gmail.com
- 📚 **文档**: 查看[详细文档](https://github.com/jackdark425/aigroup-econ-mcp/tree/main/docs)
- 🌟 **Star项目**: 如果觉得有用，请给个⭐️

## 📈 

---

**立即开始**: `uvx aigroup-econ-mcp` 🚀

让AI大模型成为你的专业计量经济学分析助手！50项Stata功能，一站式解决方案！
