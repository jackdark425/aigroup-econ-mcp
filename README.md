# aigroup-econ-mcp - 专业计量经济学MCP工具

🎯 专为Roo-Code设计的计量经济学MCP服务 - 提供统计分析、回归建模、时间序列分析，无需复杂环境配置

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![MCP](https://img.shields.io/badge/MCP-1.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/Version-0.4.0-orange.svg)

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
- **描述性统计** - 均值、方差、偏度、峰度等
- **假设检验** - t检验、F检验、卡方检验、ADF检验
- **相关性分析** - Pearson、Spearman、Kendall相关系数

### 📈 回归建模
- **OLS回归** - 普通最小二乘法回归分析
- **正则化回归** - Lasso、Ridge回归
- **模型诊断** - 残差分析、异方差检验

### ⏰ 时间序列分析
- **平稳性检验** - ADF、KPSS检验
- **ARIMA建模** - 自动定阶和参数估计
- **VAR/VECM模型** - 向量自回归/误差修正模型
- **GARCH模型** - 波动率建模

### 🏢 面板数据分析
- **固定效应模型** - 控制个体/时间固定效应
- **随机效应模型** - 处理随机效应
- **Hausman检验** - 模型选择检验
- **面板单位根检验** - 面板数据平稳性分析

### 🤖 机器学习集成
- **随机森林** - 非线性关系建模
- **梯度提升** - 高精度预测
- **特征重要性** - 变量选择分析
- **交叉验证** - 模型性能评估

## 🔧 工具列表

### 基础统计工具
| 工具 | 功能 | 输入方式 |
|------|------|----------|
| `descriptive_statistics` | 描述性统计分析 | 数据字典/CSV/JSON |
| `ols_regression` | OLS回归分析 | y_data, x_data/CSV/JSON |
| `hypothesis_testing` | 假设检验 | data1, data2/CSV/JSON |
| `correlation_analysis` | 相关性分析 | 数据字典/CSV/JSON |

### 时间序列工具
| 工具 | 功能 | 输入方式 |
|------|------|----------|
| `time_series_analysis` | 时间序列分析 | 时间序列数据/CSV/JSON |
| `var_model_analysis` | VAR模型分析 | 多变量时间序列/CSV/JSON |
| `vecm_model_analysis` | VECM模型分析 | 多变量时间序列/CSV/JSON |
| `garch_model_analysis` | GARCH模型分析 | 时间序列数据/CSV/JSON |
| `state_space_model_analysis` | 状态空间模型 | 时间序列数据/CSV/JSON |
| `variance_decomposition_analysis` | 方差分解 | 多变量时间序列/CSV/JSON |

### 面板数据工具
| 工具 | 功能 | 输入方式 |
|------|------|----------|
| `panel_fixed_effects` | 固定效应模型 | y_data, x_data, entity_ids, time_periods/CSV |
| `panel_random_effects` | 随机效应模型 | y_data, x_data, entity_ids, time_periods/CSV |
| `panel_hausman_test` | Hausman检验 | y_data, x_data, entity_ids, time_periods/CSV |
| `panel_unit_root_test` | 面板单位根检验 | 面板数据/CSV |

### 机器学习工具
| 工具 | 功能 | 输入方式 |
|------|------|----------|
| `random_forest_regression_analysis` | 随机森林回归 | y_data, x_data/CSV/JSON |
| `gradient_boosting_regression_analysis` | 梯度提升回归 | y_data, x_data/CSV/JSON |
| `lasso_regression_analysis` | Lasso回归 | y_data, x_data/CSV/JSON |
| `ridge_regression_analysis` | Ridge回归 | y_data, x_data/CSV/JSON |
| `cross_validation_analysis` | 交叉验证 | y_data, x_data/CSV/JSON |
| `feature_importance_analysis_tool` | 特征重要性 | y_data, x_data/CSV/JSON |

## 📁 文件输入支持

### 支持的文件格式
- **CSV文件** - 自动解析表头和数值数据
- **JSON文件** - 支持标准JSON数据格式
- **自动检测** - 智能识别文件格式和数据类型

### 使用方式

#### 方式1：直接数据输入（传统方式）
```json
{
  "data": {
    "GDP增长率": [3.2, 2.8, 3.5, 2.9],
    "通货膨胀率": [2.1, 2.3, 1.9, 2.4]
  }
}
```

#### 方式2：CSV文件输入（推荐）
```json
{
  "file_content": "GDP增长率,通货膨胀率\n3.2,2.1\n2.8,2.3\n3.5,1.9\n2.9,2.4",
  "file_format": "csv"
}
```

#### 方式3：文件路径输入
```json
{
  "file_path": "./test_data.csv",
  "file_format": "auto"
}
```

## ⚙️ 安装配置

### 方式1：uvx安装（推荐）
```bash
# 直接运行最新版本
uvx aigroup-econ-mcp

# 指定版本
uvx aigroup-econ-mcp@0.4.0
```

### 方式2：pip安装
```bash
# 安装包
pip install aigroup-econ-mcp

# 运行服务
aigroup-econ-mcp
```

### 依赖说明
- **核心依赖**: pandas, numpy, scipy, statsmodels, matplotlib
- **扩展依赖**: linearmodels, scikit-learn, arch
- **轻量级**: 无需torch或其他重型依赖

## 📚 使用示例

### 基础统计分析
```python
# 描述性统计
result = await descriptive_statistics(
    data={
        "GDP增长率": [3.2, 2.8, 3.5, 2.9],
        "通货膨胀率": [2.1, 2.3, 1.9, 2.4]
    }
)
```

### 回归分析
```python
# OLS回归
result = await ols_regression(
    y_data=[10, 12, 15, 18, 20],
    x_data=[[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
    feature_names=["广告支出", "价格"]
)
```

### 文件输入分析
```python
# 使用CSV文件
result = await descriptive_statistics(
    file_content="变量1,变量2\n1.2,3.4\n2.3,4.5\n3.4,5.6",
    file_format="csv"
)
```

## 🔍 故障排除

### 常见问题

#### uvx安装卡住
```bash
# 清除缓存重试
uvx --no-cache aigroup-econ-mcp
```

#### 工具返回错误
- 检查数据格式是否正确
- 确保没有缺失值
- 查看详细错误信息

#### MCP服务连接失败
- 检查网络连接
- 确保Python版本>=3.8
- 查看详细错误日志

### 数据要求
- **样本量**: 建议至少20个观测点
- **数据类型**: 所有变量必须为数值型
- **缺失值**: 自动处理或报错提示

## 🏗️ 项目架构

### 模块结构
```
src/aigroup_econ_mcp/
├── server.py                    # MCP服务器核心
├── cli.py                       # 命令行入口
├── config.py                    # 配置管理
└── tools/                       # 工具模块
    ├── base.py                  # 基础工具类
    ├── statistics.py            # 统计分析
    ├── regression.py            # 回归分析
    ├── time_series.py           # 时间序列
    ├── panel_data.py            # 面板数据
    ├── machine_learning.py      # 机器学习
    ├── file_parser.py           # 文件解析
    ├── data_loader.py           # 数据加载
    ├── decorators.py            # 装饰器
    ├── tool_registry.py         # 工具注册
    └── tool_handlers.py         # 业务处理器
```

### 设计特点
- **组件化架构** - 模块化设计，易于维护
- **统一接口** - 所有工具支持文件输入
- **错误处理** - 统一的错误处理和日志记录
- **性能优化** - 异步处理和缓存机制

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
2. 创建功能分支
3. 提交更改
4. 开启Pull Request

### 代码规范
- 遵循PEP 8编码规范
- 使用类型注解
- 添加单元测试
- 更新相关文档

## 📄 许可证

MIT License - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- **Model Context Protocol (MCP)** - 模型上下文协议
- **Roo-Code** - AI编程助手
- **statsmodels** - 统计分析库
- **pandas** - 数据处理库
- **scikit-learn** - 机器学习库
- **linearmodels** - 面板数据分析库

## 📞 支持

- 💬 [GitHub Issues](https://github.com/jackdark425/aigroup-econ-mcp/issues)
- 📧 邮箱：jackdark425@gmail.com
- 📚 文档：查看项目文档和示例

---

**立即开始**: `uvx aigroup-econ-mcp` 🚀