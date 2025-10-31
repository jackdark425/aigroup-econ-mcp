# aigroup-econ-mcp - 专业计量经济学MCP工具

🎯 专为Roo-Code设计的计量经济学MCP服务 - 提供统计分析、回归建模、时间序列分析，无需复杂环境配置

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![MCP](https://img.shields.io/badge/MCP-1.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 功能特性

- 📊 **描述性统计分析** - 自动计算均值、方差、偏度、峰度等统计量
- 📈 **回归分析** - OLS回归、逐步回归、模型诊断
- 🧪 **假设检验** - t检验、F检验、卡方检验、ADF检验
- ⏰ **时间序列分析** - 平稳性检验、ARIMA模型、预测
- 🔄 **结构化输出** - 完整的Pydantic模型支持
- 🎯 **上下文管理** - 进度报告、日志记录、错误处理
- 📁 **文件输入支持** - 支持CSV/JSON文件自动解析
- 📊 **面板数据分析** - 固定效应、随机效应模型等
- 🤖 **机器学习集成** - 随机森林、梯度提升等算法

## 🚀 快速开始（Roo-Code用户）

### 一键启动MCP服务

```bash
# 使用uvx快速启动（推荐，无需安装）
uvx aigroup-econ-mcp
```

就这么简单！ MCP服务会自动：

✅ 下载最新版本
✅ 配置轻量级依赖（仅~50MB）
✅ 启动并连接到Roo-Code
✅ 提供21个专业计量经济学工具

### 配置Roo-Code

如果需要手动配置RooCode的MCP服务，请在RooCode的设置中添加以下配置：

```json
{
  "mcpServers": {
    "aigroup-econ-mcp": {
      "command": "uvx",
      "args": [
        "aigroup-econ-mcp"
      ],
      "env": {},
      "alwaysAllow": [
        "descriptive_statistics",
        "ols_regression",
        "hypothesis_testing",
        "time_series_analysis",
        "correlation_analysis",
        "panel_fixed_effects",
        "panel_random_effects",
        "panel_hausman_test",
        "panel_unit_root_test",
        "var_model_analysis",
        "vecm_model_analysis",
        "garch_model_analysis",
        "state_space_model_analysis",
        "variance_decomposition_analysis",
        "random_forest_regression_analysis",
        "gradient_boosting_regression_analysis",
        "lasso_regression_analysis",
        "ridge_regression_analysis",
        "cross_validation_analysis",
        "feature_importance_analysis_tool"
      ]
    }
  }
}
```

配置说明：

- `command`: 使用uvx运行，无需本地安装
- `args`: 启动参数
- `alwaysAllow`: 允许访问的工具列表
- `env`: 环境变量（可留空）

配置完成后，RooCode将自动连接到aigroup-econ-mcp服务，您可以直接使用以下工具：

| 工具类别 | 工具 | 功能 |
|---------|------|------|
| **基础统计** | descriptive_statistics | 描述性统计分析 |
| | ols_regression | OLS回归分析 |
| | hypothesis_testing | 假设检验 |
| | time_series_analysis | 时间序列分析 |
| | correlation_analysis | 相关性分析 |
| **面板数据** | panel_fixed_effects | 固定效应模型 |
| | panel_random_effects | 随机效应模型 |
| | panel_hausman_test | Hausman检验 |
| | panel_unit_root_test | 面板单位根检验 |
| **时间序列** | var_model_analysis | VAR模型分析 |
| | vecm_model_analysis | VECM模型分析 |
| | garch_model_analysis | GARCH模型分析 |
| | state_space_model_analysis | 状态空间模型分析 |
| | variance_decomposition_analysis | 方差分解分析 |
| **机器学习** | random_forest_regression_analysis | 随机森林回归 |
| | gradient_boosting_regression_analysis | 梯度提升树回归 |
| | lasso_regression_analysis | Lasso回归 |
| | ridge_regression_analysis | Ridge回归 |
| | cross_validation_analysis | 交叉验证 |
| | feature_importance_analysis_tool | 特征重要性分析 |

## 📦 安装方式

### 方式1：uvx（推荐，无需安装）

```bash
# 直接运行最新版本
uvx aigroup-econ-mcp

# 或指定版本
uvx aigroup-econ-mcp@1.0.0
```

优点：

⚡ 快速启动（几秒钟）
🔄 自动获取最新版本
💾 无需本地安装
🎯 轻量级依赖（~50MB，包含统计分析库）

### 方式2：pip安装

```bash
# 基础安装（包含所有计量经济学功能）
pip install aigroup-econ-mcp

# 运行
aigroup-econ-mcp
```

依赖说明：

- **核心依赖**（默认）：pandas, numpy, scipy, mcp, statsmodels, matplotlib
- **扩展依赖**：linearmodels（面板数据）, scikit-learn（机器学习）, arch（GARCH模型）
- **轻量级**：无需torch或其他重型依赖
- **推荐**：直接使用基础安装，包含所有计量经济学功能！

## ✨ 核心特性

1️⃣ 智能数据分析
✅ 自动清洗：自动处理缺失值和异常值
✅ 统计计算：完整的描述性统计量
✅ 可视化：自动生成图表和报告

2️⃣ 专业回归分析
📊 OLS回归：完整的回归诊断和残差分析
🔧 逐步回归：特征选择和模型优化
📈 模型评估：R²、调整R²、F检验等指标

3️⃣ 假设检验套件
🧪 多样化检验：t检验、F检验、卡方检验、ADF检验
📊 详细报告：统计量、p值、置信区间
💡 结果解读：自动生成检验结论和建议

4️⃣ 时间序列专业工具
⏰ 平稳性检验：ADF、KPSS等完整检验套件
📈 ARIMA建模：自动定阶和参数估计
🔮 预测功能：点预测和区间预测

5️⃣ 面板数据分析
🏢 固定效应模型：控制个体/时间固定效应
📊 随机效应模型：处理随机效应
🔍 Hausman检验：模型选择
📉 面板单位根检验：面板数据平稳性分析

6️⃣ 机器学习集成
🌳 随机森林：非线性关系建模
🚀 梯度提升：高精度预测
🔗 正则化回归：Lasso/Ridge防止过拟合
🔍 交叉验证：模型性能评估
🎯 特征重要性：变量选择

7️⃣ 文件输入支持
📁 自动解析：支持CSV/JSON文件自动解析
🔄 向后兼容：保持原有直接数据输入方式
⚙️ 灵活输入：可混合使用文件和直接数据

8️⃣ 结构化输出
📋 Pydantic模型：类型安全的数据结构
📊 丰富格式：表格、JSON、Markdown报告
🎯 错误处理：详细的错误信息和建议

## 🔧 故障排除

### uvx安装卡住
**问题**：`uvx aigroup-econ-mcp` 卡住不动

**解决**：
- 确保使用最新版本
- 检查网络连接
- 尝试清除缓存：`uvx --no-cache aigroup-econ-mcp`

### 工具返回错误
**问题**：统计分析返回NoneType或错误

**解决**：
- 确保数据格式正确（列表或字典）
- 检查数据中是否有缺失值
- 查看详细错误信息和参数要求

### RooCode中无法使用MCP工具
**问题**：在RooCode中看不到aigroup-econ-mcp工具

**解决**：
- 确保配置了正确的MCP服务配置
- 检查uvx是否正常工作：`uvx --version`
- 重启RooCode
- 查看RooCode的MCP服务日志

### MCP服务连接失败
**问题**：MCP服务启动失败或连接超时

**解决**：
- 检查网络连接
- 尝试使用 `uvx --no-cache aigroup-econ-mcp` 清除缓存
- 确保Python版本>=3.8
- 查看详细错误日志

## 📂 项目结构

### 使用uvx安装运行（推荐）

```bash
# 一键安装和运行
uvx aigroup-econ-mcp

# 指定端口运行
uvx aigroup-econ-mcp --port 8080 --debug

# 使用不同的传输协议
uvx aigroup-econ-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

### 本地开发

```bash
# 克隆项目
git clone https://github.com/jackdark425/aigroup-econ-mcp
cd aigroup-econ-mcp

# 开发模式运行
uv run aigroup-econ-mcp --port 8000 --debug

# 或使用uvx
uvx -p . aigroup-econ-mcp
```

## 与RooCode集成

在RooCode的MCP配置文件中添加：

```json
"aigroup-econ-mcp": {
  "command": "uvx",
  "args": [
    "aigroup-econ-mcp"
  ],
  "alwaysAllow": [
    "descriptive_statistics",
    "ols_regression",
    "hypothesis_testing",
    "time_series_analysis",
    "correlation_analysis",
    "panel_fixed_effects",
    "panel_random_effects",
    "panel_hausman_test",
    "panel_unit_root_test",
    "var_model_analysis",
    "vecm_model_analysis",
    "garch_model_analysis",
    "state_space_model_analysis",
    "variance_decomposition_analysis",
    "random_forest_regression_analysis",
    "gradient_boosting_regression_analysis",
    "lasso_regression_analysis",
    "ridge_regression_analysis",
    "cross_validation_analysis",
    "feature_importance_analysis_tool"
  ],
  "disabled": true
}
```

## 📋 工具详细说明

### 基础统计工具

#### descriptive_statistics
描述性统计分析工具

**参数：**
- `data`: 数值数据列表或字典
- `variables`: 变量名列表（可选）
- `output_format`: 输出格式（table/json）
- `file_path`: CSV/JSON文件路径（可选）
- `file_content`: CSV/JSON文件内容（可选）

**返回：**
- 基础统计量（均值、方差、偏度、峰度）
- 数据质量评估
- 可视化图表

#### ols_regression
OLS回归分析工具

**参数：**
- `y_data`: 因变量数据
- `x_data`: 自变量数据（列表或矩阵）
- `feature_names`: 变量名称（可选）
- `add_constant`: 是否添加常数项（默认true）
- `output_detail`: 输出详细程度（可选）
- `file_path`: CSV/JSON文件路径（可选）
- `file_content`: CSV/JSON文件内容（可选）

**返回：**
- 回归系数和统计显著性
- 模型拟合优度（R²、调整R²）
- 模型诊断（残差分析、异方差检验）
- 预测结果（如果提供预测数据）

#### hypothesis_testing
假设检验工具

**参数：**
- `data1`: 第一组数据
- `data2`: 第二组数据（可选）
- `test_type`: 检验类型（t_test/f_test/chi2_test/adf_test）
- `alpha`: 显著性水平（默认0.05）
- `file_path`: 文件路径（可选）
- `file_content`: 文件内容（可选）

**返回：**
- 检验统计量和p值
- 检验结果和置信区间
- 效应大小和统计功效

#### time_series_analysis
时间序列分析工具

**参数：**
- `data`: 时间序列数据
- `analysis_type`: 分析类型（stationarity/arima/forecast）
- `lags`: 滞后期数（默认12）
- `forecast_steps`: 预测步数（可选）
- `file_path`: 文件路径（可选）
- `file_content`: 文件内容（可选）

**返回：**
- 平稳性检验结果
- ARIMA模型参数
- 预测值和置信区间
- 模型诊断图表

#### correlation_analysis
相关性分析工具

**参数：**
- `data`: 变量数据字典
- `method`: 相关系数类型（pearson/spearman/kendall）
- `plot`: 是否生成可视化图表（默认true）
- `file_path`: 文件路径（可选）
- `file_content`: 文件内容（可选）

**返回：**
- 相关系数矩阵
- 显著性检验结果
- 相关性热力图

### 面板数据分析工具

#### panel_fixed_effects
固定效应模型分析工具

**参数：**
- `y_data`: 因变量数据
- `x_data`: 自变量数据
- `entity_ids`: 实体标识符
- `time_periods`: 时间标识符
- `feature_names`: 特征名称（可选）
- `entity_effects`: 是否包含实体效应（默认true）
- `time_effects`: 是否包含时间效应（默认false）
- `file_path`: CSV文件路径（可选）
- `file_content`: CSV文件内容（可选）

#### panel_random_effects
随机效应模型分析工具

**参数：**
- `y_data`: 因变量数据
- `x_data`: 自变量数据
- `entity_ids`: 实体标识符
- `time_periods`: 时间标识符
- `feature_names`: 特征名称（可选）
- `entity_effects`: 是否包含实体效应（默认true）
- `time_effects`: 是否包含时间效应（默认false）
- `file_path`: CSV文件路径（可选）
- `file_content`: CSV文件内容（可选）

#### panel_hausman_test
Hausman检验工具

**参数：**
- `y_data`: 因变量数据
- `x_data`: 自变量数据
- `entity_ids`: 实体标识符
- `time_periods`: 时间标识符
- `feature_names`: 特征名称（可选）
- `file_path`: CSV文件路径（可选）
- `file_content`: CSV文件内容（可选）

#### panel_unit_root_test
面板单位根检验工具

**参数：**
- `data`: 时间序列数据
- `y_data`: 因变量数据（可选）
- `entity_ids`: 实体标识符
- `time_periods`: 时间标识符
- `feature_names`: 特征名称（可选）
- `test_type`: 检验类型（默认levinlin）
- `file_path`: CSV文件路径（可选）
- `file_content`: CSV文件内容（可选）

### 高级时间序列工具

#### var_model_analysis
VAR模型分析工具

**参数：**
- `data`: 多变量时间序列数据
- `max_lags`: 最大滞后阶数（默认5）
- `ic`: 信息准则（默认aic）
- `file_path`: 文件路径（可选）
- `file_content`: 文件内容（可选）

#### vecm_model_analysis
VECM模型分析工具

**参数：**
- `data`: 多变量时间序列数据
- `coint_rank`: 协整秩（默认1）
- `deterministic`: 确定性项（默认co）
- `max_lags`: 最大滞后阶数（默认5）
- `file_path`: 文件路径（可选）
- `file_content`: 文件内容（可选）

#### garch_model_analysis
GARCH模型分析工具

**参数：**
- `data`: 时间序列数据
- `order`: GARCH模型阶数（默认(1, 1)）
- `dist`: 分布类型（默认normal）
- `file_path`: 文件路径（可选）
- `file_content`: 文件内容（可选）

#### state_space_model_analysis
状态空间模型分析工具

**参数：**
- `data`: 时间序列数据
- `state_dim`: 状态维度（默认1）
- `observation_dim`: 观测维度（默认1）
- `trend`: 是否包含趋势（默认true）
- `seasonal`: 是否包含季节性（默认false）
- `period`: 季节周期（默认12）
- `file_path`: 文件路径（可选）
- `file_content`: 文件内容（可选）

#### variance_decomposition_analysis
方差分解分析工具

**参数：**
- `data`: 多变量时间序列数据
- `periods`: 分解期数（默认10）
- `max_lags`: 最大滞后阶数（默认5）
- `file_path`: 文件路径（可选）
- `file_content`: 文件内容（可选）

### 机器学习工具

#### random_forest_regression_analysis
随机森林回归分析工具

**参数：**
- `y_data`: 因变量数据
- `x_data`: 自变量数据
- `feature_names`: 特征名称（可选）
- `n_estimators`: 树的数量（默认100）
- `max_depth`: 最大深度（可选）
- `file_path`: 文件路径（可选）
- `file_content`: 文件内容（可选）

#### gradient_boosting_regression_analysis
梯度提升回归分析工具

**参数：**
- `y_data`: 因变量数据
- `x_data`: 自变量数据
- `feature_names`: 特征名称（可选）
- `n_estimators`: 树的数量（默认100）
- `learning_rate`: 学习率（默认0.1）
- `max_depth`: 最大深度（默认3）
- `file_path`: 文件路径（可选）
- `file_content`: 文件内容（可选）

#### lasso_regression_analysis
Lasso回归分析工具

**参数：**
- `y_data`: 因变量数据
- `x_data`: 自变量数据
- `feature_names`: 特征名称（可选）
- `alpha`: 正则化强度（默认1.0）
- `file_path`: 文件路径（可选）
- `file_content`: 文件内容（可选）

#### ridge_regression_analysis
Ridge回归分析工具

**参数：**
- `y_data`: 因变量数据
- `x_data`: 自变量数据
- `feature_names`: 特征名称（可选）
- `alpha`: 正则化强度（默认1.0）
- `file_path`: 文件路径（可选）
- `file_content`: 文件内容（可选）

#### cross_validation_analysis
交叉验证分析工具

**参数：**
- `y_data`: 因变量数据
- `x_data`: 自变量数据
- `feature_names`: 特征名称（可选）
- `model_type`: 模型类型（默认random_forest）
- `cv_folds`: 交叉验证折数（默认5）
- `scoring`: 评分标准（默认r2）
- `file_path`: 文件路径（可选）
- `file_content`: 文件内容（可选）

#### feature_importance_analysis_tool
特征重要性分析工具

**参数：**
- `y_data`: 因变量数据
- `x_data`: 自变量数据
- `feature_names`: 特征名称（可选）
- `method`: 分析方法（默认random_forest）
- `top_k`: 返回前k个重要特征（默认5）
- `file_path`: 文件路径（可选）
- `file_content`: 文件内容（可选）

## 可用资源

### 示例数据集

```
resource://dataset/sample/economic_growth
resource://dataset/sample/stock_returns
resource://dataset/sample/time_series
```

### 提示模板

```
prompt://economic_analysis?data_description=...&analysis_type=descriptive
```

## 项目结构

```
aigroup-econ-mcp/
├── src/aigroup_econ_mcp/
│   ├── __init__.py              # 包初始化
│   ├── server.py                # MCP服务器核心
│   ├── cli.py                   # 命令行入口
│   └── tools/
│       ├── __init__.py
│       ├── statistics.py        # 统计分析工具
│       ├── regression.py         # 回归分析工具
│       ├── time_series.py        # 时间序列工具
│       ├── panel_data.py         # 面板数据工具
│       ├── machine_learning.py   # 机器学习工具
│       └── file_parser.py        # 文件解析工具
├── pyproject.toml               # 项目配置
├── README.md
└── examples/
```

## 依赖要求

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
- statsmodels >= 0.13.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0
- mcp >= 1.0.0
- pydantic >= 2.0.0
- linearmodels >= 7.0
- scikit-learn >= 1.0.0
- arch >= 6.0.0

## 开发

### 环境设置

```bash
# 安装开发依赖
uv add --dev pytest pytest-asyncio black isort mypy ruff

# 运行测试
uv run pytest

# 代码格式化
uv run black src/
uv run isort src/

# 类型检查
uv run mypy src/

# 代码检查
uv run ruff check src/
```

### 构建和发布

```bash
# 构建包
uv build

# 发布到PyPI
uv publish
```

## 许可证

MIT License

## 贡献

欢迎贡献代码！请查看[贡献指南](CONTRIBUTING.md)了解详情。

## 🤝 贡献

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 开启Pull Request

## 📄 许可证

MIT License - 查看 LICENSE 了解详情

## 🙏 致谢

- Model Context Protocol (MCP) - 模型上下文协议
- Roo-Code - AI编程助手
- statsmodels - 统计分析库
- pandas - 数据处理库
- scikit-learn - 机器学习库
- linearmodels - 面板数据分析库

## 📞 支持

💬 提交 [GitHub Issues](https://github.com/jackdark425/aigroup-econ-mcp/issues)
📧 邮件：jackdark425@gmail.com
📚 文档：查看项目文档和示例

**立即开始**: `uvx aigroup-econ-mcp` 🚀