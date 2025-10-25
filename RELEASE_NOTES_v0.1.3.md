# aigroup-econ-mcp v0.1.3 发布说明

## 🎯 本次更新重点

**关键Bug修复版本** - 修复了影响OLS回归和时间序列分析的严重问题

## 🐛 已修复的问题

### 1. NumPy类型序列化错误
**问题**: `ols_regression`和`time_series_analysis`工具返回错误
```
Error: Input should be a valid dictionary or instance of OLSRegressionResult
[type=model_type, input_value=None, input_type=NoneType]
```

**根本原因**: 
- statsmodels库返回`numpy.float64`等NumPy类型
- Pydantic无法正确序列化这些类型到JSON
- FastMCP的structuredContent验证失败

**解决方案**:
```python
# 修复前
result = OLSRegressionResult(
    rsquared=model.rsquared,  # numpy.float64 ❌
    ...
)

# 修复后  
result = OLSRegressionResult(
    rsquared=float(model.rsquared),  # Python float ✅
    ...
)
```

### 2. OLS数据形状转换错误
**问题**: "endog and exog matrices are different sizes"

**根本原因**: 
- 错误使用`np.column_stack()`改变了输入数据形状
- 输入数据已经是正确的行列格式

**解决方案**:
```python
# 修复前
X = np.column_stack(x_data)  # ❌ 错误地重塑数据

# 修复后
X = np.array(x_data)  # ✅ 保持原始格式
```

## ✅ 受影响的工具

| 工具 | 状态 | 修复内容 |
|------|------|----------|
| `ols_regression` | ✅ 已修复 | NumPy类型转换 + 数据形状修复 |
| `time_series_analysis` | ✅ 已修复 | NumPy类型转换 |
| `descriptive_statistics` | ✅ 正常 | 无需修复 |
| `hypothesis_testing` | ✅ 正常 | 无需修复 |
| `correlation_analysis` | ✅ 正常 | 无需修复 |

## 📊 测试验证

### 测试结果
```
✅ OLS回归分析
   - R² = 0.9907
   - 成功返回structuredContent
   - 所有系数正确计算

✅ 时间序列分析
   - ADF检验正常
   - ACF/PACF计算正确
   - 平稳性判断准确
```

### 验证文件
- `test_simple_ols.py` - 直接函数调用测试
- `BUG_FIX_REPORT.md` - 详细修复报告

## 🚀 如何升级

### 方法1: 使用uvx（推荐）
```bash
# uvx会自动获取最新版本
uvx aigroup-econ-mcp
```

### 方法2: 使用pip
```bash
pip install --upgrade aigroup-econ-mcp
```

### 方法3: 从源码安装
```bash
git clone https://github.com/yourusername/aigroup-econ-mcp.git
cd aigroup-econ-mcp
pip install -e .
```

## 📝 使用示例

### OLS回归分析（已修复）
```python
# 现在可以正常工作！
y_data = [12000, 13500, 11800, 14200, 15100]
x_data = [[800, 5.2], [900, 5.8], [750, 4.9], ...]

result = await session.call_tool(
    "ols_regression",
    arguments={
        "y_data": y_data,
        "x_data": x_data,
        "feature_names": ["广告支出", "价格指数"]
    }
)
# ✅ 返回完整的回归结果，包含R²、系数、p值等
```

### 时间序列分析（已修复）
```python
# 现在可以正常工作！
data = [12000, 13500, 11800, 14200, ...]

result = await session.call_tool(
    "time_series_analysis",
    arguments={"data": data}
)
# ✅ 返回ADF检验、ACF、PACF等完整分析结果
```

## 🔍 技术细节

### 修复的关键代码片段

**ols_regression**:
```python
# 类型转换
coefficients[var_name] = {
    "coef": float(coef),           # ✅
    "std_err": float(model.bse[i]), # ✅
    "t_value": float(model.tvalues[i]), # ✅
    "p_value": float(model.pvalues[i]), # ✅
    ...
}

# 数据处理
X = np.array(x_data)  # ✅ 正确的转换方式
X = sm.add_constant(X)
```

**time_series_analysis**:
```python
result = TimeSeriesStatsResult(
    adf_statistic=float(adf_result[0]),      # ✅
    adf_pvalue=float(adf_result[1]),         # ✅
    stationary=bool(adf_result[1] < 0.05),   # ✅
    acf=[float(x) for x in acf_values],      # ✅
    pacf=[float(x) for x in pacf_values]     # ✅
)
```

## 📚 相关文档

- [BUG_FIX_REPORT.md](BUG_FIX_REPORT.md) - 详细的问题诊断和修复报告
- [CHANGELOG.md](CHANGELOG.md) - 完整更新日志
- [README.md](README.md) - 使用文档

## 🙏 致谢

感谢社区用户报告这些问题并提供详细的测试反馈！

---

**发布日期**: 2025-10-25  
**版本**: v0.1.3  
**重要性**: 🔴 **高优先级** - 强烈建议升级