# aigroup-econ-mcp - 专业计量经济学MCP工具

🎯 专为Roo-Code设计的计量经济学MCP服务 - 提供统计分析、回归建模、时间序列分析，支持CSV/JSON/TXT多种数据格式

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![MCP](https://img.shields.io/badge/MCP-1.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/Version-1.2.0-orange.svg)

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

### Roo-Code配置

在RooCode的MCP设置中添加：

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
        "cross_validation_analysis", "feature_importance_analysis_tool"
      ]
    }
  }
}
```

## ✨ 核心功能

### 📊 统计分析
- **描述性统计** - 均值、方差、偏度、峰度、中位数、四分位数
- **假设检验** - t检验、F检验、卡方检验、ADF检验、KPSS检验
- **相关性分析** - Pearson、Spearman、Kendall相关系数

### 📈 回归建模
- **OLS回归** - 普通最小二乘法回归，完整诊断统计
- **正则化回归** - Lasso、Ridge回归，自动特征选择
- **模型诊断** - 残差分析、异方差检验、多重共线性检测

### ⏰ 时间序列分析
- **平稳性检验** - ADF、KPSS检验，趋势强度分析
- **自相关分析** - ACF、PACF函数计算和解释
- **VAR/VECM模型** - 向量自回归/误差修正模型
- **GARCH模型** - 波动率建模和预测
- **状态空间模型** - 卡尔曼滤波和平滑

### 🏢 面板数据分析
- **固定效应模型** - 控制个体/时间固定效应
- **随机效应模型** - 处理随机效应
- **Hausman检验** - 模型选择检验
- **面板单位根检验** - Levin-Lin-Chu等多种检验方法

### 🤖 机器学习集成
- **随机森林** - 非线性关系建模，特征重要性分析
- **梯度提升** - 高精度预测，支持XGBoost风格
- **特征重要性** - 基于多种方法的变量选择
- **交叉验证** - K折交叉验证，模型稳定性评估

## 🔧 工具列表

### 基础统计工具（支持CSV/JSON/TXT格式）
| 工具 | 功能 | 主要参数 | 输出 |
|------|------|---------|------|
| `descriptive_statistics` | 描述性统计 | data或file_content | 均值、标准差、偏度、峰度、相关矩阵 |
| `ols_regression` | OLS回归 | y_data, x_data或file_content | R²、系数、t统计量、p值、置信区间 |
| `hypothesis_testing` | 假设检验 | data1, data2, test_type | 统计量、p值、显著性判断 |
| `correlation_analysis` | 相关性分析 | data, method | 相关系数矩阵 |

### 时间序列工具（支持CSV/JSON/TXT格式）
| 工具 | 功能 | 主要参数 | 输出 |
|------|------|---------|------|
| `time_series_analysis` | 时间序列分析 | data | 平稳性检验、ACF/PACF、模型建议 |
| `var_model_analysis` | VAR模型 | data, max_lags, ic | 最优阶数、系数、脉冲响应 |
| `vecm_model_analysis` | VECM模型 | data, coint_rank | 协整向量、误差修正项 |
| `garch_model_analysis` | GARCH模型 | data, order | 波动率持续性、条件方差 |
| `state_space_model_analysis` | 状态空间模型 | data, state_dim | 滤波状态、平滑状态估计 |
| `variance_decomposition_analysis` | 方差分解 | data, periods | 各变量贡献度分解 |

### 面板数据工具（支持CSV/TXT格式）
| 工具 | 功能 | 主要参数 | 输出 |
|------|------|---------|------|
| `panel_fixed_effects` | 固定效应模型 | y_data, x_data, entity_ids, time_periods | R²、系数、F统计量 |
| `panel_random_effects` | 随机效应模型 | y_data, x_data, entity_ids, time_periods | R²、系数、随机效应方差 |
| `panel_hausman_test` | Hausman检验 | y_data, x_data, entity_ids, time_periods | 检验统计量、模型选择建议 |
| `panel_unit_root_test` | 面板单位根 | data, entity_ids, time_periods | 平稳性判断、临界值 |

### 机器学习工具（支持CSV/JSON/TXT格式）
| 工具 | 功能 | 主要参数 | 输出 |
|------|------|---------|------|
| `random_forest_regression_analysis` | 随机森林 | y_data, x_data, n_estimators | R²、MSE、特征重要性 |
| `gradient_boosting_regression_analysis` | 梯度提升 | y_data, x_data, learning_rate | R²、MSE、特征重要性 |
| `lasso_regression_analysis` | Lasso回归 | y_data, x_data, alpha | R²、稀疏系数、特征选择 |
| `ridge_regression_analysis` | Ridge回归 | y_data, x_data, alpha | R²、正则化系数 |
| `cross_validation_analysis` | 交叉验证 | y_data, x_data, cv_folds | 平均得分、标准差、稳定性 |
| `feature_importance_analysis_tool` | 特征重要性 | y_data, x_data, method | 特征排名、重要性分数 |

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
```txt
100.5
102.3
101.8
103.5
```

- **多列数值**: 空格或制表符分隔
```txt
GDP CPI 失业率
3.2 2.1 4.5
2.8 2.3 4.2
3.5 1.9 4.0
```

- **键值对格式**: 变量名: 值列表
```txt
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

### 方式1：uvx安装（推荐）
```bash
# 直接运行最新版本
uvx aigroup-econ-mcp

# 指定版本
uvx aigroup-econ-mcp@1.1.0
```

### 方式2：pip安装
```bash
# 安装包
pip install aigroup-econ-mcp

# 运行服务
aigroup-econ-mcp
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

| 分析类型 | 最小样本量 | 推荐样本量 | 特殊要求 |
|---------|-----------|-----------|---------|
| 描述性统计 | 5 | 20+ | 无缺失值 |
| OLS回归 | 变量数+2 | 30+ | 无多重共线性 |
| 时间序列 | 10 | 40+ | 时间顺序，等间隔 |
| 面板数据 | 实体数×3 | 实体数×10+ | 平衡或非平衡面板 |
| 机器学习 | 20 | 100+ | 训练集/测试集分割 |

## 🏗️ 项目架构

### 模块结构
```
src/aigroup_econ_mcp/
├── server.py                    # MCP服务器核心
├── cli.py                       # 命令行入口
├── config.py                    # 配置管理
└── tools/                       # 工具模块
    ├── base.py                  # 基础工具类
    ├── statistics.py            # 统计分析工具
    ├── regression.py            # 回归分析工具
    ├── time_series.py           # 时间序列工具
    ├── panel_data.py            # 面板数据工具
    ├── machine_learning.py      # 机器学习工具
    ├── file_parser.py           # 文件解析器（CSV/JSON/TXT）
    ├── data_loader.py           # 数据加载器
    ├── tool_registry.py         # 工具注册中心
    ├── tool_handlers.py         # 业务处理器
    └── tool_descriptions.py     # 工具描述和文档
```

### 设计特点
- **🎯 组件化架构** - 模块化设计，职责单一，易于维护和扩展
- **🔄 统一接口** - 所有工具支持CSV/JSON/TXT三种格式输入
- **⚡ 异步处理** - 基于asyncio的异步设计，支持并发请求
- **🛡️ 错误处理** - 统一的错误处理和详细的错误信息
- **📝 完整文档** - 每个工具都有详细的参数说明和使用示例
- **🧪 全面测试** - 单元测试和集成测试覆盖

### 新增特性（v1.1.0）
- ✨ **TXT格式支持** - 支持单列、多列、键值对三种TXT格式
- 📝 **完善参数描述** - 所有20个工具的MCP参数都有详细说明
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

## 📞 支持

- 💬 **GitHub Issues**: [提交问题](https://github.com/jackdark425/aigroup-econ-mcp/issues)
- 📧 **邮箱**: jackdark425@gmail.com
- 📚 **文档**: 查看[详细文档](https://github.com/jackdark425/aigroup-econ-mcp/tree/main/docs)
- 🌟 **Star项目**: 如果觉得有用，请给个⭐️

## 📈 版本历史

### v1.2.0 (2024-11-03)
- ✨ 新增TXT格式支持（单列、多列、键值对）
- 📝 完善20个工具的MCP参数描述
- 🔍 改进文件格式自动检测
- 📂 支持文件路径直接输入

### v1.0.1 (2024-10)
- 🐛 修复状态空间模型bug
- 🔧 增强时间序列分析输出

---

**立即开始**: `uvx aigroup-econ-mcp` 🚀

让AI大模型成为你的专业计量经济学分析助手！
