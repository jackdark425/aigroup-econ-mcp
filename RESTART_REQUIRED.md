# 🔄 需要重启MCP服务器

## 最新修复
刚刚完成了参数名统一化修复：

### 修改的文件
1. `src/aigroup_econ_mcp/tools/file_parser.py` - 移除了`data1`参数
2. `src/aigroup_econ_mcp/server.py` - `hypothesis_testing`参数统一为`data`

### 修复的问题
- ❌ 之前：`single_var`类型返回`data`和`data1`两个参数
- ✅ 现在：统一只返回`data`参数
- ❌ 之前：`hypothesis_testing`使用`data1`参数
- ✅ 现在：所有工具统一使用`data`参数

## 重启步骤
1. 按 `Ctrl+Shift+P`
2. 输入 `MCP: Restart Server`
3. 选择 `aigroup-econ-mcp`

## 重启后测试
将继续测试剩余的工具（从`time_series_analysis`开始）