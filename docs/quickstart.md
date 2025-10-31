# 快速入门指南

## 🚀 5分钟快速上手

本指南将帮助您在5分钟内完成aigroup-econ-mcp的安装和第一个分析。

### 步骤1: 安装和启动

#### 方式1: uvx一键启动（推荐）
```bash
# 一键启动MCP服务
uvx aigroup-econ-mcp
```

#### 方式2: pip安装
```bash
# 安装包
pip install aigroup-econ-mcp

# 启动服务
aigroup-econ-mcp
```

### 步骤2: 配置Roo-Code

在RooCode的MCP设置中添加以下配置：

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

### 步骤3: 运行第一个分析

#### 示例1: 描述性统计

**直接数据输入**:
```python
# 使用描述性统计工具
result = await descriptive_statistics(
    data={
        "GDP增长率": [3.2, 2.8, 3.5, 2.9, 3.1, 2.7, 3.3, 3.0],
        "通货膨胀率": [2.1, 2.3, 1.9, 2.4, 2.2, 2.5, 2.0, 2.3],
        "失业率": [4.5, 4.2, 4.0, 4.3, 4.1, 4.4, 3.9, 4.2]
    }
)

# 查看结果
print(result)
```

**CSV文件输入**:
```python
# 准备CSV数据
csv_content = """GDP增长率,通货膨胀率,失业率
3.2,2.1,4.5
2.8,2.3,4.2
3.5,1.9,4.0
2.9,2.4,4.3
3.1,2.2,4.1
2.7,2.5,4.4
3.3,2.0,3.9
3.0,2.3,4.2"""

# 使用文件输入
result = await descriptive_statistics(
    file_content=csv_content,
    file_format="csv"
)
```

#### 示例2: OLS回归分析

```python
# 销售数据回归分析
result = await ols_regression(
    y_data=[12000, 13500, 11800, 14200, 13800, 15100, 12500, 14800],  # 销售额
    x_data=[
        [800, 99],   # [广告支出, 价格]
        [900, 95],
        [750, 102],
        [1000, 98],
        [850, 96],
        [950, 94],
        [820, 97],
        [880, 93]
    ],
    feature_names=["广告支出", "价格"]
)

# 查看回归结果
print(f"R²: {result.rsquared:.3f}")
print(f"系数: {result.coefficients}")
```

#### 示例3: 相关性分析

```python
# 分析变量间相关性
result = await correlation_analysis(
    data={
        "销售额": [12000, 13500, 11800, 14200, 13800],
        "广告支出": [800, 900, 750, 1000, 850],
        "价格": [99, 95, 102, 98, 96],
        "竞争对手数量": [3, 3, 4, 3, 4]
    },
    method="pearson"
)

# 查看相关性矩阵
print("相关性矩阵:")
for var1, correlations in result.correlation_matrix.items():
    for var2, corr in correlations.items():
        print(f"{var1} vs {var2}: {corr:.3f}")
```

## 📊 典型工作流

### 宏观经济分析工作流

```python
# 1. 描述性统计
stats_result = await descriptive_statistics(
    data={
        "GDP增长率": [3.2, 2.8, 3.5, 2.9, 3.1, 2.7],
        "通货膨胀率": [2.1, 2.3, 1.9, 2.4, 2.2, 2.5],
        "失业率": [4.5, 4.2, 4.0, 4.3, 4.1, 4.4]
    }
)

# 2. 相关性分析
corr_result = await correlation_analysis(
    data={
        "GDP增长率": [3.2, 2.8, 3.5, 2.9, 3.1, 2.7],
        "通货膨胀率": [2.1, 2.3, 1.9, 2.4, 2.2, 2.5],
        "失业率": [4.5, 4.2, 4.0, 4.3, 4.1, 4.4]
    }
)

# 3. 假设检验
test_result = await hypothesis_testing(
    data1=[3.2, 2.8, 3.5, 2.9, 3.1],  # 前5个季度
    data2=[2.7, 3.0, 2.8, 3.2, 2.9],  # 后5个季度
    test_type="t_test"
)

print("宏观经济分析完成!")
```

### 金融时间序列分析工作流

```python
# 1. 平稳性检验
stationarity_result = await time_series_analysis(
    data=[100.5, 102.3, 101.8, 103.5, 104.2, 103.8, 105.1, 104.7, 106.2, 105.8],
    analysis_type="stationarity"
)

# 2. 自相关分析
acf_result = await time_series_analysis(
    data=[100.5, 102.3, 101.8, 103.5, 104.2, 103.8, 105.1, 104.7, 106.2, 105.8],
    analysis_type="acf_pacf"
)

print(f"数据是否平稳: {stationarity_result.is_stationary}")
```

## 📁 文件输入使用

### CSV文件格式

**经济数据CSV**:
```csv
年份,GDP增长率,通货膨胀率,失业率
2020,3.2,2.1,4.5
2021,2.8,2.3,4.2
2022,3.5,1.9,4.0
2023,2.9,2.4,4.3
```

**销售数据CSV**:
```csv
月份,广告支出,价格,销售额,竞争对手数量
1月,800,99,12000,3
2月,900,95,13500,3
3月,750,102,11800,4
4月,1000,98,14200,3
```

### JSON文件格式

**多变量数据JSON**:
```json
{
  "GDP增长率": [3.2, 2.8, 3.5, 2.9, 3.1],
  "通货膨胀率": [2.1, 2.3, 1.9, 2.4, 2.2],
  "失业率": [4.5, 4.2, 4.0, 4.3, 4.1]
}
```

**时间序列数据JSON**:
```json
{
  "日期": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
  "股价": [100.5, 102.3, 101.8, 103.5]
}
```

## 🔧 常用工具速查

### 基础统计工具

| 工具 | 用途 | 示例 |
|------|------|------|
| `descriptive_statistics` | 描述性统计 | `data={"变量": [值]}` |
| `ols_regression` | OLS回归 | `y_data=[], x_data=[]` |
| `hypothesis_testing` | 假设检验 | `data1=[], data2=[]` |
| `correlation_analysis` | 相关性分析 | `data={}, method="pearson"` |

### 时间序列工具

| 工具 | 用途 | 示例 |
|------|------|------|
| `time_series_analysis` | 时间序列分析 | `data=[], analysis_type="stationarity"` |
| `var_model_analysis` | VAR模型 | `data={}, max_lags=3` |
| `garch_model_analysis` | GARCH模型 | `data=[], order=(1,1)` |

### 机器学习工具

| 工具 | 用途 | 示例 |
|------|------|------|
| `random_forest_regression_analysis` | 随机森林 | `y_data=[], x_data=[], n_estimators=100` |
| `feature_importance_analysis_tool` | 特征重要性 | `y_data=[], x_data=[], method="random_forest"` |

## 🐛 常见问题

### Q: uvx启动失败怎么办？
**A**: 尝试以下解决方案：
```bash
# 清除缓存重试
uvx --no-cache aigroup-econ-mcp

# 检查网络连接
ping pypi.org

# 使用pip安装
pip install aigroup-econ-mcp
```

### Q: 工具返回错误怎么办？
**A**: 检查数据格式：
- 确保所有数据都是数值类型
- 检查数据长度是否匹配
- 验证没有缺失值或无效值

### Q: 如何查看详细错误信息？
**A**: 启用调试模式：
```bash
uvx aigroup-econ-mcp --debug
```

### Q: 文件输入不工作？
**A**: 检查文件格式：
- CSV文件必须有表头
- JSON文件必须是有效格式
- 所有列必须是数值类型

## 🎯 下一步学习

完成快速入门后，建议：

1. **深入学习工具**: 查看[工具使用指南](user-guide/tools.md)
2. **掌握文件输入**: 学习[文件输入功能](user-guide/file-input.md)
3. **探索高级功能**: 参考[高级示例](examples/advanced.md)
4. **解决实际问题**: 查看[案例研究](examples/case-studies.md)

## 📞 获取帮助

如果遇到问题：

1. **查看文档**: 首先查阅相关文档
2. **搜索Issue**: 查看是否已有解决方案
3. **提交Issue**: 提供详细的问题描述
4. **社区讨论**: 参与社区交流

---

**恭喜！您已经完成了快速入门。现在可以开始使用aigroup-econ-mcp进行专业的数据分析了！** 🎉

[查看完整工具列表 →](user-guide/tools.md) | [学习文件输入 →](user-guide/file-input.md) | [探索高级功能 →](examples/advanced.md)