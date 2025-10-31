# 🔄 MCP服务器重启指南

## 🎯 目的
重启aigroup-econ-mcp服务器以加载修复后的代码

## ✅ 已完成的修复

### 修复的文件
- [`src/aigroup_econ_mcp/tools/file_parser.py`](src/aigroup_econ_mcp/tools/file_parser.py:1)

### 修复的Bug
1. ✅ 移除`variable_name`参数（single_var类型）
2. ✅ 移除`y_variable`参数（regression类型）
3. ✅ 移除`y_variable`参数（panel类型）
4. ✅ 添加`data1`参数映射（hypothesis_testing兼容）

## 🔄 重启步骤

### 方法1: 通过VSCode重启（推荐）

1. **打开命令面板**
   - 按 `Ctrl+Shift+P` (Windows) 或 `Cmd+Shift+P` (Mac)

2. **重启MCP服务器**
   - 输入: `MCP: Restart All Servers`
   - 或者: `MCP: Restart Server` → 选择 `aigroup-econ-mcp`

3. **验证重启成功**
   - 查看输出面板中的MCP日志
   - 应该看到 "Server started successfully"

### 方法2: 完全重启VSCode

如果方法1不生效，完全重启VSCode：
```bash
# 1. 保存所有文件
# 2. 关闭VSCode
# 3. 重新打开VSCode
# 4. MCP服务器会自动启动
```

## 📋 重启后检查清单

- [ ] MCP服务器状态显示为"运行中"
- [ ] 没有错误日志
- [ ] 可以看到20个工具
- [ ] 准备好进行测试

## 🧪 重启后测试

重启成功后，我将测试所有20个工具的CSV支持功能：

### 测试清单（20个工具）

#### 基础统计 (5个)
1. [ ] descriptive_statistics
2. [ ] ols_regression  
3. [ ] hypothesis_testing
4. [ ] time_series_analysis
5. [ ] correlation_analysis

#### 面板数据 (4个)
6. [ ] panel_fixed_effects
7. [ ] panel_random_effects
8. [ ] panel_hausman_test
9. [ ] panel_unit_root_test

#### 高级时间序列 (5个)
10. [ ] var_model_analysis
11. [ ] vecm_model_analysis
12. [ ] garch_model_analysis
13. [ ] state_space_model_analysis
14. [ ] variance_decomposition_analysis

#### 机器学习 (6个)
15. [ ] random_forest_regression_analysis
16. [ ] gradient_boosting_regression_analysis
17. [ ] lasso_regression_analysis
18. [ ] ridge_regression_analysis
19. [ ] cross_validation_analysis
20. [ ] feature_importance_analysis_tool

## ⚡ 快速重启命令

如果您更倾向使用命令行：

```bash
# 重新安装MCP服务器（确保最新代码）
cd d:/aigroup-econ-mcp
pip install -e .

# 然后在VSCode中重启MCP服务器
```

---

## 📞 如遇问题

如果重启后仍有问题：

1. 检查MCP日志输出
2. 确认虚拟环境正确：`d:\aigroup-econ-mcp\.venv`
3. 验证server.py语法：`python -m py_compile src/aigroup_econ_mcp/server.py`
4. 检查依赖安装：`pip list | grep aigroup`

---

**准备好后，请告诉我，我将立即开始测试所有20个工具！** 🚀