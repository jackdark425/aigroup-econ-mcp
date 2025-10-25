# AIGroup 计量经济学 MCP 工具

专业计量经济学MCP工具 - 让大模型直接进行数据分析

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

## 快速开始

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
git clone https://github.com/aigroup/aigroup-econ-mcp
cd aigroup-econ-mcp

# 开发模式运行
uv run aigroup-econ-mcp --port 8000 --debug

# 或使用uvx
uvx -p . aigroup-econ-mcp
```

## 与Claude Desktop集成

在Claude Desktop的配置文件中添加：

```json
{
  "mcpServers": {
    "aigroup-econ-mcp": {
      "command": "uvx",
      "args": ["aigroup-econ-mcp", "--transport", "stdio"]
    }
  }
}
```

## 使用示例

### 描述性统计分析

```python
# 计算基础统计量
result = await descriptive_statistics({
    "GDP": [100, 110, 120, 115, 125],
    "Inflation": [2.1, 2.3, 1.9, 2.4, 2.0]
})
```

### OLS回归分析

```python
# 回归分析
result = await ols_regression(
    y_data=[100, 110, 120, 115, 125],
    x_data=[[2.1, 4.5], [2.3, 4.2], [1.9, 4.0], [2.4, 4.3], [2.0, 4.1]],
    feature_names=["inflation", "unemployment"]
)
```

### 假设检验

```python
# 假设检验
result = await hypothesis_testing(
    data1=[100, 110, 120, 115, 125],
    data2=[95, 105, 115, 120, 130],
    test_type="t_test"
)
```

### 时间序列分析

```python
# 时间序列分析
result = await time_series_analysis([100, 110, 120, 115, 125, 130, 128, 135])
```

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

## 支持

如有问题或建议，请通过[GitHub Issues](https://github.com/aigroup/aigroup-econ-mcp/issues)联系我们。

## 致谢

感谢Model Context Protocol (MCP)社区提供的优秀工具和文档。