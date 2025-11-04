# Tools Directory

## 当前架构 (v2.0 - Adapter Pattern)

### 活跃文件

1. **econometrics_adapter.py** - 核心适配器
   - 将 econometrics/ 核心算法适配为 MCP 工具
   - 支持文件输入和多种输出格式
   - 减少 84% 代码重复

2. **data_loader.py** - 数据加载组件
   - 支持 txt/json/csv/excel 格式
   - DataLoader: OLS/GMM 数据
   - MLEDataLoader: MLE 数据

3. **output_formatter.py** - 输出格式化组件
   - MarkdownFormatter: Markdown 格式
   - TextFormatter: 纯文本格式
   - 结果保存功能

### 架构优势

```
MCP Server (fastmcp_server.py)
    ↓
Adapter Layer (econometrics_adapter.py) - 90 lines
    ↓
Core Algorithms (econometrics/basic_parametric_estimation/)
    ├── ols/ols_model.py - OLS 核心算法
    ├── mle/mle_model.py - MLE 核心算法
    └── gmm/gmm_model.py - GMM 核心算法 (已修复 j_p_value bug)
```

**优点**：
- ✅ 代码复用：单一真相源
- ✅ DRY 原则：不重复自己
- ✅ 易于维护：Bug 只需修复一次
- ✅ 清晰分层：职责明确
- ✅ 易于扩展：新算法只需写适配器

### 备份文件

`bak_old_implementation/` - 旧的独立实现（已废弃）
- ols_tool.py (155 lines)
- mle_tool.py (219 lines) 
- gmm_tool.py (190 lines)

这些文件已被 econometrics_adapter.py (164 lines) 替代
节省代码：474 lines (84%)

## 使用示例

```python
from tools.econometrics_adapter import (
    ols_adapter,
    mle_adapter,
    gmm_adapter
)

# 直接数据输入
result = ols_adapter(
    y_data=[1,2,3,4,5],
    x_data=[[1],[2],[3],[4],[5]],
    output_format="json"
)

# 文件输入
result = mle_adapter(
    file_path="data/sample.csv",
    distribution="normal",
    output_format="markdown",
    save_path="results/mle.md"
)
```

## 迁移记录

- 2025-11-04: 切换到适配器模式
- 修复核心 GMM bug (j_p_value)
- 移动旧实现到 bak_old_implementation/
- 减少 84% 重复代码

## 相关文档

- [架构分析](../ARCHITECTURE_ANALYSIS.md)
- [服务器代码](../fastmcp_server.py)
- [核心算法](../econometrics/basic_parametric_estimation/)