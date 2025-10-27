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
✅ 提供5个专业计量经济学工具

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
        "correlation_analysis"
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

| 工具 | 功能 | 用途 |
|------|------|------|
| descriptive_statistics | 描述性统计分析 | 加载数据并自动计算统计量 |
| ols_regression | OLS回归分析 | 回归建模和模型诊断 |
| hypothesis_testing | 假设检验 | t检验、F检验、卡方检验、ADF检验 |
| time_series_analysis | 时间序列分析 | 平稳性检验、ARIMA模型、预测 |
| correlation_analysis | 相关性分析 | 变量间相关性分析和可视化 |

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

5️⃣ 结构化输出
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
    "correlation_analysis"
  ],
  "disabled": true
}
```



## 📋 工具详细说明

#### descriptive_statistics
描述性统计分析工具

**参数：**
- `data`: 数值数据列表或字典
- `variables`: 变量名列表（可选）
- `output_format`: 输出格式（table/json）

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

**返回：**
- 相关系数矩阵
- 显著性检验结果
- 相关性热力图


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
│       ├── regression.py        # 回归分析工具
│       └── time_series.py       # 时间序列工具
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

## 📞 支持

💬 提交 [GitHub Issues](https://github.com/jackdark425/aigroup-econ-mcp/issues)
📧 邮件：jackdark425@gmail.com
📚 文档：查看项目文档和示例

**立即开始**: `uvx aigroup-econ-mcp` 🚀

---
