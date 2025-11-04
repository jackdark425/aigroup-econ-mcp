# AIGroup 计量经济学 MCP 服务器 v2.0

## 🎉 新功能概览

v2.0 版本引入了**组件化设计**，支持多种输入输出格式，让计量经济学分析更加灵活和便捷！

## ✨ 主要更新

### 1. 文件输入支持
现在支持直接从文件加载数据，无需手动输入数据数组！

**支持的文件格式：**
- ✅ **TXT** - 文本文件（空格或制表符分隔）
- ✅ **JSON** - JSON格式数据
- ✅ **CSV** - 逗号分隔值文件
- ✅ **Excel** - Excel文件（.xlsx, .xls）

### 2. 多种输出格式
分析结果可以选择不同的格式输出：

- 📝 **Markdown** - 易读的Markdown格式（默认）
- 📄 **TXT** - 纯文本格式

### 3. 结果保存功能
可以直接将分析结果保存到文件，方便后续使用！

## 🔧 组件化架构

### 核心组件

1. **数据加载器** ([`tools/data_loader.py`](tools/data_loader.py))
   - `DataLoader` - 用于OLS和GMM的数据加载
   - `MLEDataLoader` - 用于MLE的数据加载
   - 自动识别文件格式并解析

2. **输出格式化器** ([`tools/output_formatter.py`](tools/output_formatter.py))
   - `MarkdownFormatter` - Markdown格式输出
   - `TextFormatter` - 纯文本格式输出
   - 统一的格式化接口

3. **工具实现**
   - [`tools/ols_tool.py`](tools/ols_tool.py) - OLS回归（增强版）
   - [`tools/mle_tool.py`](tools/mle_tool.py) - MLE估计（增强版）
   - [`tools/gmm_tool.py`](tools/gmm_tool.py) - GMM估计（增强版）

## 📚 可用工具

### 直接数据输入工具（保持向后兼容）

1. **basic_parametric_estimation_ols** - OLS回归
2. **basic_parametric_estimation_mle** - 最大似然估计
3. **basic_parametric_estimation_gmm** - 广义矩估计

### 文件输入工具（新增）

4. **ols_from_file** - 从文件执行OLS回归
5. **mle_from_file** - 从文件执行MLE估计
6. **gmm_from_file** - 从文件执行GMM估计

## 💡 使用示例

### 示例1：从CSV文件执行OLS回归

```python
# 使用MCP客户端调用
result = await session.call_tool(
    "ols_from_file",
    {
        "file_path": "data/regression_data.csv",
        "constant": True,
        "confidence_level": 0.95,
        "output_format": "markdown",
        "save_path": "results/ols_result.md"  # 可选
    }
)
```

### 示例2：从TXT文件执行MLE估计

```python
result = await session.call_tool(
    "mle_from_file",
    {
        "file_path": "data/sample.txt",
        "distribution": "normal",
        "output_format": "txt"
    }
)
```

### 示例3：从Excel文件执行GMM估计

```python
result = await session.call_tool(
    "gmm_from_file",
    {
        "file_path": "data/panel_data.xlsx",
        "constant": True,
        "output_format": "markdown",
        "save_path": "results/gmm_analysis.md"
    }
)
```

## 📊 数据格式要求

### 回归分析数据（OLS/GMM）

**CSV/Excel格式：**
```csv
y,x1,x2,x3
1.5,1.0,2.0,3.0
2.3,1.5,2.5,3.5
3.1,2.0,3.0,4.0
...
```

**JSON格式：**
```json
{
  "y_data": [1.5, 2.3, 3.1, ...],
  "x_data": [[1.0, 2.0], [1.5, 2.5], ...],
  "feature_names": ["X1", "X2"]  // 可选
}
```

**TXT格式（空格分隔）：**
```
1.5 1.0 2.0
2.3 1.5 2.5
3.1 2.0 3.0
...
```

### MLE估计数据

**TXT格式（每行一个值）：**
```
1.2
2.3
1.8
...
```

**JSON格式：**
```json
{
  "data": [1.2, 2.3, 1.8, ...]
}
```

## 🚀 快速开始

### 1. 启动服务器

```bash
python fastmcp_server.py
# 或
uv run mcp dev fastmcp_server.py
```

### 2. 运行测试

```bash
# 测试所有功能
python test_file_input.py

# 测试原有功能
python test_fastmcp_tools.py
```

### 3. 准备数据文件

将你的数据保存为支持的格式（CSV、JSON、TXT或Excel），确保：
- 第一列为因变量（y）
- 其余列为自变量（x）
- 包含列名（CSV/Excel）

### 4. 调用工具

通过MCP客户端调用相应的工具，指定文件路径和参数。

## 📁 项目结构

```
aigroup-econ-mcp/
├── fastmcp_server.py          # MCP服务器主程序（v2.0）
├── tools/
│   ├── data_loader.py         # 数据加载组件（新增）
│   ├── output_formatter.py    # 输出格式化组件（新增）
│   ├── ols_tool.py            # OLS工具（增强）
│   ├── mle_tool.py            # MLE工具（增强）
│   └── gmm_tool.py            # GMM工具（增强）
├── test_data/
│   ├── sample_data.csv        # 测试数据（CSV）
│   └── sample_mle.txt         # 测试数据（TXT）
├── test_file_input.py         # 文件输入功能测试
└── test_fastmcp_tools.py      # 原有功能测试
```

## 🔄 版本历史

### v2.0.0 (2025-11-04)
- ✨ 新增文件输入支持（txt/json/csv/excel）
- 📝 新增多种输出格式（markdown/txt）
- 💾 新增结果保存功能
- 🏗️ 实现组件化架构设计
- 🧪 添加完整的测试用例

### v1.4.1
- 🐛 修复GMM工具的bug（j_p_value为None的问题）
- ✅ 完成基础三个工具的测试

## 🛠️ 技术特性

- **模块化设计** - 清晰的职责分离
- **可扩展性** - 易于添加新的文件格式或输出格式
- **向后兼容** - 保留原有的直接数据输入接口
- **错误处理** - 完善的异常处理和错误提示
- **类型安全** - 使用类型注解提高代码质量

## 📖 依赖项

```toml
[dependencies]
mcp = "^1.0.0"
numpy = "^1.24.0"
scipy = "^1.10.0"
pandas = "^2.0.0"
openpyxl = "^3.1.0"  # Excel支持
pydantic = "^2.0.0"
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

---

**AIGroup 计量经济学 MCP 服务器 v2.0**
让计量经济学分析更简单、更灵活！ 🎯