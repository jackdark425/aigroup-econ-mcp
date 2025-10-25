# AIGroup 计量经济学 MCP 服务使用指南

## 项目概述

AIGroup 计量经济学 MCP 服务是一个专业的Model Context Protocol (MCP)工具，专为大语言模型提供完整的计量经济学分析功能。该服务使用最新的MCP特性，包括结构化输出、上下文管理、进度报告等。

## 功能特性

- 📊 **描述性统计分析** - 自动计算均值、方差、偏度、峰度等统计量
- 📈 **回归分析** - OLS回归、逐步回归、模型诊断（VIF、异方差检验等）
- 🧪 **假设检验** - t检验、F检验、卡方检验、ADF单位根检验
- ⏰ **时间序列分析** - 平稳性检验、ARIMA模型、自相关分析
- 🔄 **结构化输出** - 完整的Pydantic模型支持
- 🎯 **上下文管理** - 进度报告、日志记录、错误处理

## 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy statsmodels scipy matplotlib pydantic mcp click uvicorn
```

### 2. 启动服务器

```bash
# 使用uvx（推荐）
uvx aigroup-econ-mcp --port 8000

# 或直接运行
cd src
python -c "from aigroup_econ_mcp.cli import main; main(['--port', '8000'])"
```

### 3. 验证安装

运行示例验证所有功能：

```bash
cd examples
python basic_usage.py
```

## 与Claude Desktop集成

在Claude Desktop的配置文件中添加：

```json
{
  "mcpServers": {
    "aigroup-econ-mcp": {
      "command": "uvx",
      "args": ["aigroup-econ-mcp", "--transport", "stdio"],
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

## MCP工具使用示例

### 描述性统计分析

```python
# 分析经济数据
result = await descriptive_statistics({
    "GDP": [100, 110, 120, 115, 125],
    "Inflation": [2.1, 2.3, 1.9, 2.4, 2.0]
})

# 结构化结果包含：
# - count: 样本数量
# - mean: 均值
# - std: 标准差
# - min/max: 极值
# - skewness: 偏度
# - kurtosis: 峰度
```

### OLS回归分析

```python
# 回归分析
result = await ols_regression(
    y_data=[120, 135, 150, 160, 175],
    x_data=[[8, 100], [9, 98], [10, 95], [11, 94], [12, 92]],
    feature_names=["advertising", "price"]
)

# 结果包含：
# - R²和调整R²
# - F统计量和p值
# - AIC/BIC信息准则
# - 各系数详情（系数、标准误、t值、p值、置信区间）
```

### 假设检验

```python
# 双样本t检验
result = await hypothesis_testing(
    data1=[100, 110, 120, 115, 125],
    data2=[95, 105, 115, 120, 130],
    test_type="t_test"
)

# ADF单位根检验
result = await hypothesis_testing(
    data1=[100, 102, 98, 105, 103, 108, 106, 112],
    test_type="adf"
)
```

### 时间序列分析

```python
# 时间序列分析
result = await time_series_analysis([100, 110, 120, 115, 125, 130, 128, 135])

# 结果包含：
# - ADF/KPSS平稳性检验
# - 自相关和偏自相关函数
# - 序列平稳性判断
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
├── tests/
│   ├── __init__.py
│   └── test_econometrics.py     # 测试套件
├── examples/
│   ├── basic_usage.py           # 基础使用示例
│   ├── mcp_client_example.py    # MCP客户端示例
│   └── README.md               # 详细使用指南
├── pyproject.toml               # 项目配置
└── README.md                    # 项目文档
```

## 开发和测试

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_econometrics.py::TestStatistics -v
```

### 代码质量

```bash
# 格式化代码
black src/ examples/
isort src/ examples/

# 类型检查
mypy src/

# 代码检查
ruff check src/
```

## 部署选项

### 1. 本地开发

```bash
# 开发模式
uv run aigroup-econ-mcp --port 8000 --debug

# 或使用uvx
uvx -p . aigroup-econ-mcp --port 8000
```

### 2. Docker部署

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install uv && uv sync

COPY src/ src/
COPY README.md .

CMD ["uv", "run", "aigroup-econ-mcp", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. 生产部署

```bash
# 使用uvx部署（推荐）
uvx aigroup-econ-mcp --host 0.0.0.0 --port 8000 --transport streamable-http

# 或使用gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 aigroup_econ_mcp.server:create_mcp_server
```

## 许可证

MIT License

## 贡献

欢迎贡献代码！请查看主项目的[贡献指南](CONTRIBUTING.md)了解详情。

## 支持

如有问题或建议，请通过[GitHub Issues](https://github.com/aigroup/aigroup-econ-mcp/issues)联系我们。