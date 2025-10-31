# aigroup-econ-mcp 项目概览

## 🎯 项目简介

**aigroup-econ-mcp** 是一个专业的计量经济学MCP（Model Context Protocol）工具，专为Roo-Code设计。它提供了完整的统计分析、回归建模、时间序列分析和机器学习功能，无需复杂的环境配置即可使用。

### 核心价值

- **🚀 一键启动**: 使用 `uvx aigroup-econ-mcp` 即可快速启动
- **📊 专业分析**: 提供21个专业计量经济学工具
- **📁 文件支持**: 支持CSV/JSON文件自动解析
- **🎯 易用性**: 无需复杂配置，开箱即用
- **⚡ 高性能**: 优化的算法和缓存机制

## ✨ 核心功能

### 统计分析
- **描述性统计**: 均值、方差、偏度、峰度等完整统计量
- **假设检验**: t检验、F检验、卡方检验、ADF检验
- **相关性分析**: Pearson、Spearman、Kendall相关系数

### 回归建模
- **OLS回归**: 普通最小二乘法回归分析
- **模型诊断**: 残差分析、异方差检验、多重共线性检测
- **正则化回归**: Lasso、Ridge回归防止过拟合

### 时间序列分析
- **平稳性检验**: ADF、KPSS检验
- **ARIMA建模**: 自动定阶和参数估计
- **VAR/VECM模型**: 向量自回归/误差修正模型
- **GARCH模型**: 波动率建模和预测

### 面板数据分析
- **固定效应模型**: 控制个体/时间固定效应
- **随机效应模型**: 处理随机效应
- **Hausman检验**: 模型选择检验
- **面板单位根检验**: 面板数据平稳性分析

### 机器学习集成
- **随机森林**: 非线性关系建模
- **梯度提升**: 高精度预测
- **特征重要性**: 变量选择分析
- **交叉验证**: 模型性能评估

### 文件输入支持
- **CSV文件**: 自动解析表头和数值数据
- **JSON文件**: 支持标准JSON数据格式
- **智能识别**: 自动检测数据类型和变量角色

## 🏗️ 技术架构

### 模块化设计
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
- **组件化架构**: 模块化设计，易于维护和扩展
- **统一接口**: 所有工具支持文件输入和直接数据输入
- **错误处理**: 统一的错误处理和用户友好的错误消息
- **性能优化**: 异步处理、缓存机制、智能算法

## 🚀 快速开始

### 一键启动
```bash
# 使用uvx快速启动（推荐）
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

### 基础使用示例

#### 描述性统计
```python
# 直接数据输入
result = await descriptive_statistics(
    data={
        "GDP增长率": [3.2, 2.8, 3.5, 2.9],
        "通货膨胀率": [2.1, 2.3, 1.9, 2.4]
    }
)

# 文件输入
result = await descriptive_statistics(
    file_content="GDP增长率,通货膨胀率\n3.2,2.1\n2.8,2.3\n3.5,1.9\n2.9,2.4",
    file_format="csv"
)
```

#### OLS回归分析
```python
result = await ols_regression(
    y_data=[10, 12, 15, 18, 20],
    x_data=[[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
    feature_names=["广告支出", "价格"]
)
```

## 📊 工具列表

### 基础统计工具 (5个)
| 工具 | 功能 | 输入方式 |
|------|------|----------|
| `descriptive_statistics` | 描述性统计分析 | 数据字典/CSV/JSON |
| `ols_regression` | OLS回归分析 | y_data, x_data/CSV/JSON |
| `hypothesis_testing` | 假设检验 | data1, data2/CSV/JSON |
| `time_series_analysis` | 时间序列分析 | 时间序列数据/CSV/JSON |
| `correlation_analysis` | 相关性分析 | 数据字典/CSV/JSON |

### 时间序列工具 (5个)
| 工具 | 功能 | 输入方式 |
|------|------|----------|
| `var_model_analysis` | VAR模型分析 | 多变量时间序列/CSV/JSON |
| `vecm_model_analysis` | VECM模型分析 | 多变量时间序列/CSV/JSON |
| `garch_model_analysis` | GARCH模型分析 | 时间序列数据/CSV/JSON |
| `state_space_model_analysis` | 状态空间模型 | 时间序列数据/CSV/JSON |
| `variance_decomposition_analysis` | 方差分解 | 多变量时间序列/CSV/JSON |

### 面板数据工具 (4个)
| 工具 | 功能 | 输入方式 |
|------|------|----------|
| `panel_fixed_effects` | 固定效应模型 | y_data, x_data, entity_ids, time_periods/CSV |
| `panel_random_effects` | 随机效应模型 | y_data, x_data, entity_ids, time_periods/CSV |
| `panel_hausman_test` | Hausman检验 | y_data, x_data, entity_ids, time_periods/CSV |
| `panel_unit_root_test` | 面板单位根检验 | 面板数据/CSV |

### 机器学习工具 (6个)
| 工具 | 功能 | 输入方式 |
|------|------|----------|
| `random_forest_regression_analysis` | 随机森林回归 | y_data, x_data/CSV/JSON |
| `gradient_boosting_regression_analysis` | 梯度提升回归 | y_data, x_data/CSV/JSON |
| `lasso_regression_analysis` | Lasso回归 | y_data, x_data/CSV/JSON |
| `ridge_regression_analysis` | Ridge回归 | y_data, x_data/CSV/JSON |
| `cross_validation_analysis` | 交叉验证 | y_data, x_data/CSV/JSON |
| `feature_importance_analysis_tool` | 特征重要性 | y_data, x_data/CSV/JSON |

## 🔧 系统要求

### 环境要求
- **Python**: 3.8+
- **操作系统**: Windows, macOS, Linux
- **内存**: 建议4GB+ RAM
- **存储**: 约50MB可用空间

### 核心依赖
- **pandas**: 数据处理和分析
- **numpy**: 数值计算
- **scipy**: 科学计算
- **statsmodels**: 统计分析
- **matplotlib**: 数据可视化
- **scikit-learn**: 机器学习
- **linearmodels**: 面板数据分析
- **arch**: GARCH模型分析

## 🎯 适用场景

### 学术研究
- 经济学实证研究
- 金融时间序列分析
- 社会科学数据分析
- 计量经济学教学

### 商业分析
- 市场趋势分析
- 销售预测建模
- 客户行为分析
- 风险评估

### 金融应用
- 股票收益率分析
- 波动率建模
- 投资组合优化
- 风险管理

### 政府机构
- 宏观经济监测
- 政策效果评估
- 社会经济调查
- 发展规划分析

## 📈 性能特点

### 计算性能
- **快速启动**: 几秒钟内即可开始分析
- **高效算法**: 优化的数值计算算法
- **内存管理**: 智能内存使用和缓存机制
- **并行处理**: 支持异步计算

### 用户体验
- **直观接口**: 统一的参数命名和结构
- **详细输出**: 完整的分析结果和诊断信息
- **错误处理**: 清晰的错误消息和建议
- **进度反馈**: 长时间计算的进度显示

## 🔄 版本演进

### 当前版本: v0.4.0
- ✅ 所有21个工具支持文件输入
- ✅ 优化的服务器架构
- ✅ 统一的错误处理
- ✅ 完整的测试覆盖

### 未来规划
- 🔄 更多文件格式支持（Excel, Parquet）
- 🔄 可视化输出增强
- 🔄 分布式计算支持
- 🔄 实时数据分析

## 🤝 社区与支持

### 获取帮助
- **文档**: 查看详细的[使用指南](../docs/user-guide/tools.md)
- **示例**: 参考[代码示例](../examples/)
- **问题**: 提交[GitHub Issue](https://github.com/jackdark425/aigroup-econ-mcp/issues)
- **讨论**: 参与社区讨论

### 贡献项目
我们欢迎社区贡献！您可以：
- 报告bug和改进建议
- 提交代码改进
- 编写文档和示例
- 分享使用案例

### 许可证
本项目采用 **MIT License**，允许自由使用、修改和分发。

---

**aigroup-econ-mcp 让专业的计量经济学分析变得简单易用！** 🎉

[快速开始 →](quickstart.md) | [工具指南 →](user-guide/tools.md) | [安装配置 →](installation.md)