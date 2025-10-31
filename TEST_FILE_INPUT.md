# CSV文件输入功能测试指南

## 重要提示

**您需要重启MCP服务器以加载新代码！**

### 如何重启MCP服务器

1. 在VSCode中，完全关闭并重新打开VSCode，或
2. 重启Roo/Cline扩展

## 测试步骤

### 测试1: 描述性统计（CSV输入）

```json
{
  "file_content": "GDP增长率,通货膨胀率,失业率\n3.2,2.1,4.5\n2.8,2.3,4.2\n3.5,1.9,4.0\n2.9,2.4,4.3\n3.1,2.2,4.1\n2.7,2.5,4.4",
  "file_format": "csv"
}
```

**预期输出**：
- 成功解析CSV
- 自动识别3个变量
- 输出描述性统计结果

### 测试2: OLS回归（CSV输入）

```json
{
  "file_content": "广告支出,价格,销售额\n800,99,12000\n900,95,13500\n750,102,11800\n1000,98,14200",
  "file_format": "csv"
}
```

**预期输出**：
- 自动识别因变量: 销售额（最后一列）
- 自动识别自变量: 广告支出, 价格
- 输出R²、系数等

### 测试3: JSON输入

```json
{
  "file_content": "{\"GDP增长率\": [3.2, 2.8, 3.5], \"通货膨胀率\": [2.1, 2.3, 1.9]}",
  "file_format": "json"
}
```

### 测试4: 自动格式检测

```json
{
  "file_content": "变量1,变量2\n1.2,3.4\n2.3,4.5",
  "file_format": "auto"
}
```

## 如果遇到问题

### 问题1: "Field required" 错误

**原因**: MCP服务器未重新加载新代码

**解决**: 完全重启VSCode或Roo/Cline扩展

### 问题2: 装饰器错误

**原因**: decorators.py模块导入失败

**解决**: 检查以下文件是否存在：
- `src/aigroup_econ_mcp/tools/decorators.py`
- `src/aigroup_econ_mcp/tools/tool_handlers.py`
- `src/aigroup_econ_mcp/tools/file_parser.py`

### 问题3: "file_content" 未定义

**原因**: 使用了旧版server.py

**解决**: 确认`src/aigroup_econ_mcp/__init__.py`中导入的是`server_v2`：
```python
from .server_v2 import create_mcp_server
```

## 验证清单

在测试前，请确认：

- [ ] VSCode已完全重启（或Roo/Cline扩展已重启）
- [ ] `__init__.py`导入的是`server_v2`
- [ ] 所有新模块文件存在：
  - [ ] `decorators.py`
  - [ ] `tool_handlers.py`  
  - [ ] `tool_registry.py`
  - [ ] `server_v2.py`

## 成功标志

如果一切正常，您应该看到：

1. **文件解析日志**: "检测到文件输入，开始解析..."
2. **解析成功日志**: "文件解析成功：X个变量，Y个观测"
3. **格式转换日志**: "数据已转换为XXX格式"
4. **正常输出**: 统计结果或回归结果

## 快速测试命令

重启后，直接使用：

```
用CSV数据测试descriptive_statistics工具：

file_content: "GDP,通胀,失业率\n3.2,2.1,4.5\n2.8,2.3,4.2\n3.5,1.9,4.0"
file_format: "csv"
```

祝测试顺利！🎉