
# 工具使用指南

## 📋 工具概览

aigroup-econ-mcp 提供21个专业计量经济学工具，涵盖统计分析、回归建模、时间序列分析、面板数据分析和机器学习。

### 工具分类

| 类别 | 工具数量 | 主要功能 |
|------|----------|----------|
| 基础统计 | 5个 | 描述性统计、假设检验、相关性分析 |
| 时间序列 | 5个 | VAR、VECM、GARCH、状态空间模型 |
| 面板数据 | 4个 | 固定效应、随机效应、Hausman检验 |
| 机器学习 | 6个 | 随机森林、梯度提升、正则化回归 |
| 其他 | 1个 | 方差分解分析 |

## 🔧 基础统计工具

### descriptive_statistics - 描述性统计分析

**功能**: 计算变量的描述性统计量，包括均值、方差、偏度、峰度等。

**参数**:
- `data`: 数据字典或文件路径/内容
- `variables`: 变量名列表（可选）
- `file_content`: CSV/JSON文件内容（可选）
- `file_format`: 文件格式（auto/csv/json）

**使用示例**:
```python
# 直接数据输入
result = await descriptive_statistics(
    data={
        "GDP增长率": [3.2, 2.8, 3.5, 2.9, 3.1],
        "通货膨胀率": [2.1, 2.3, 1.9, 2.4, 2.2]
    }
)

# 文件输入
result = await descriptive_statistics(
    file_content="变量1,变量2\n1.0,2.0\n3.0,4.0\n5.0,6.0",
    file_format="csv"
)
```

**返回结果**:
```json
{
  "variables": ["GDP增长率", "通货膨胀率"],
  "statistics": {
    "GDP增长率": {
      "count": 5,
      "mean": 3.1,
      "std": 0.25,
      "min": 2.8,
      "max": 3.5,
      "skewness": 0.12,
      "kurtosis": -0.85
    }
  }
}
```

### ols_regression - OLS回归分析

**功能**: 执行普通最小二乘法回归分析，提供完整的回归诊断。

**参数**:
- `y_data`: 因变量数据
- `x_data`: 自变量数据（列表或矩阵）
- `feature_names`: 变量名称（可选）
- `add_constant`: 是否添加常数项（默认true）
- `file_content`: CSV/JSON文件内容（可选）
- `file_format`: 文件格式（auto/csv/json）

**使用示例**:
```python
# 直接数据输入
result = await ols_regression(
    y_data=[10, 12, 15, 18, 20],  # 销售额
    x_data=[[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],  # [广告支出, 价格]
    feature_names=["广告支出", "价格"]
)

# 文件输入（CSV最后一列为因变量）
result = await ols_regression(
    file_content="广告支出,价格,销售额\n800,99,12000\n900,95,13500",
    file_format="csv"
)
```

**返回结果**:
```json
{
  "rsquared": 0.95,
  "adj_rsquared": 0.93,
  "f_statistic": 45.2,
  "f_pvalue": 0.001,
  "coefficients": {
    "const": 2.5,
    "广告支出": 1.8,
    "价格": -0.5
  },
  "n_obs": 5
}
```

### hypothesis_testing - 假设检验

**功能**: 执行各种假设检验，包括t检验、F检验、卡方检验、ADF检验。

**参数**:
- `data1`: 第一组数据
- `data2`: 第二组数据（可选）
- `test_type`: 检验类型（t_test/f_test/chi2_test/adf_test）
- `alpha`: 显著性水平（默认0.05）
- `file_content`: 文件内容（可选）
- `file_format`: 文件格式（auto/csv/json）

**使用示例**:
```python
# 双样本t检验
result = await hypothesis_testing(
    data1=[3.2, 2.8, 3.5, 2.9],  # 组A
    data2=[2.5, 2.9, 2.3, 2.6],  # 组B
    test_type="t_test"
)

# ADF平稳性检验
result = await hypothesis_testing(
    data1=[100, 102, 101, 103, 105],  # 时间序列数据
    test_type="adf_test"
)
```

**返回结果**:
```json
{
  "test_type": "双样本t检验",
  "statistic": 2.34,
  "p_value": 0.032,
  "significant": true,
  "confidence_interval": [0.12, 0.89]
}
```

### correlation_analysis - 相关性分析

**功能**: 计算变量间的相关系数矩阵，支持Pearson、Spearman、Kendall方法。

**参数**:
- `data`: 变量数据字典
- `method`: 相关系数类型（pearson/spearman/kendall）
- `plot`: 是否生成可视化图表（默认true）
- `file_content`: 文件内容（可选）
- `file_format`: 文件格式（auto/csv/json）

**使用示例**:
```python
# 直接数据输入
result = await correlation_analysis(
    data={
        "GDP增长率": [3.2, 2.8, 3.5, 2.9],
        "通货膨胀率": [2.1, 2.3, 1.9, 2.4],
        "失业率": [4.5, 4.2, 4.0, 4.3]
    },
    method="pearson"
)
```

**返回结果**:
```json
{
  "method": "pearson",
  "correlation_matrix": {
    "GDP增长率": {
      "GDP增长率": 1.0,
      "通货膨胀率": 0.65,
      "失业率": -0.72
    },
    "通货膨胀率": {
      "GDP增长率": 0.65,
      "通货膨胀率": 1.0,
      "失业率": -0.58
    }
  },
  "n_obs": 4
}
```

## ⏰ 时间序列工具

### time_series_analysis - 时间序列分析

**功能**: 执行时间序列分析，包括平稳性检验、自相关分析、ARIMA建模。

**参数**:
- `data`: 时间序列数据
- `analysis_type`: 分析类型（stationarity/arima/forecast）
- `lags`: 滞后期数（默认12）
- `forecast_steps`: 预测步数（可选）
- `file_content`: 文件内容（可选）
- `file_format`: 文件格式（auto/csv/json）

**使用示例**:
```python
# 平稳性检验
result = await time_series_analysis(
    data=[100, 102, 101, 103, 105, 104, 106, 107],
    analysis_type="stationarity"
)

# ARIMA建模
result = await time_series_analysis(
    data=[100, 102, 101, 103, 105, 104, 106, 107],
    analysis_type="arima"
)
```

### var_model_analysis - VAR模型分析

**功能**: 向量自回归模型分析，用于多变量时间序列建模。

**参数**:
- `data`: 多变量时间序列数据
- `max_lags`: 最大滞后阶数（默认5）
- `ic`: 信息准则（默认aic）
- `file_content`: 文件内容（可选）
- `file_format`: 文件格式（auto/csv/json）

**使用示例**:
```python
result = await var_model_analysis(
    data={
        "GDP增长率": [3.2, 2.8, 3.5, 2.9, 3.1],
        "通货膨胀率": [2.1, 2.3, 1.9, 2.4, 2.2],
        "失业率": [4.5, 4.2, 4.0, 4.3, 4.1]
    },
    max_lags=3
)
```

### garch_model_analysis - GARCH模型分析

**功能**: GARCH模型分析，用于波动率建模和金融时间序列分析。

**参数**:
- `data`: 时间序列数据
- `order`: GARCH模型阶数（默认(1, 1)）
- `dist`: 分布类型（默认normal）
- `file_content`: 文件内容（可选）
- `file_format`: 文件格式（auto/csv/json）

**使用示例**:
```python
# 股票收益率波动率建模
result = await garch_model_analysis(
    data=[0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.02, 0.01],
    order=(1, 1)
)
```

## 🏢 面板数据工具

### panel_fixed_effects - 固定效应模型

**功能**: 面板数据固定效应模型分析，控制个体固定效应。

**参数**:
- `y_data`: 因变量数据
- `x_data`: 自变量数据
- `entity_ids`: 实体标识符
- `time_periods`: 时间标识符
- `feature_names`: 特征名称（可选）
- `entity_effects`: 是否包含实体效应（默认true）
- `time_effects`: 是否包含时间效应（默认false）
- `file_content`: CSV文件内容（可选）
- `file_format`: 文件格式（auto/csv）

**使用示例**:
```python
result = await panel_fixed_effects(
    y_data=[100, 120, 150, 130, 180, 200],  # 利润
    x_data=[[5, 100], [6, 98], [7.5, 95], [6.5, 97], [9, 92], [10, 90]],  # [广告支出, 价格]
    entity_ids=["公司A", "公司A", "公司A", "公司B", "公司B", "公司B"],
    time_periods=["2020Q1", "2020Q2", "2020Q3", "2020Q1", "2020Q2", "2020Q3"],
    feature_names=["广告支出", "价格"]
)
```

### panel_random_effects - 随机效应模型

**功能**: 面板数据随机效应模型分析。

**参数**: 与固定效应模型相同

**使用示例**:
```python
result = await panel_random_effects(
    y_data=[100, 120, 150, 130, 180, 200],
    x_data=[[5, 100], [6, 98], [7.5, 95], [6.5, 97], [9, 92], [10, 90]],
    entity_ids=["公司A", "公司A", "公司A", "公司B", "公司B", "公司B"],
    time_periods=["2020Q1", "2020Q2", "2020Q3", "2020Q1", "2020Q2", "2020Q3"],
    feature_names=["广告支出", "价格"]
)
```

### panel_hausman_test - Hausman检验

**功能**: Hausman检验，用于选择固定效应模型还是随机效应模型。

**参数**: 与固定效应模型相同

**使用示例**:
```python
result = await panel_hausman_test(
    y_data=[100, 120, 150, 130, 180, 200],
    x_data=[[5, 100], [6, 98], [7.5, 95], [6.5, 97], [9, 92], [10, 90]],
    entity_ids=["公司A", "公司A", "公司A", "公司B", "公司B", "公司B"],
    time_periods=["2020Q1", "2020Q2", "2020Q3", "2020Q1", "2020Q2", "2020Q3"],
    feature_names=["广告支出", "价格"]
)
```

## 🤖 机器学习工具

### random_forest_regression_analysis - 随机森林回归

**功能**: 随机森林回归分析，用于非线性关系建模。

**参数**:
- `y_data`: 因变量数据
- `x_data`: 自变量数据
- `feature_names`: 特征名称（可选）
- `n_estimators`: 树的数量（默认100）
- `max_depth`: 最大深度（可选）
- `file_content`: 文件内容（可选）
- `file_format`: 文件格式（auto/csv/json）

**使用示例**:
```python
result = await random_forest_regression_analysis(
    y_data=[10, 12, 15, 18, 20, 22, 25, 28],
    x_data=[[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]],
    feature_names=["特征1", "特征2"],
    n_estimators=100
)
```

### gradient_boosting_regression_analysis - 梯度提升回归

**功能**: 梯度提升树回归分析，提供高精度预测。

**参数**:
- `y_data`: 因变量数据
- `x_data`: 自变量数据
- `feature_names`: 特征名称（可选）
- `n_estimators`: 树的数量（默认100）
- `learning_rate`: 学习率（默认0.1）
- `max_depth`: 最大深度（默认3）
- `file_content`: 文件内容（可选）
- `file_format`: 文件格式（auto/csv/json）

**使用示例**:
```python
result = await gradient_boosting_regression_analysis(
    y_data=[10, 12, 15, 18, 20, 22, 25, 28],
    x_data=[[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]],
    feature_names=["特征1", "特征2"],
    n_estimators=100,
    learning_rate=0.1
)
```

### feature_importance_analysis_tool - 特征重要性分析

**功能**: 分析特征重要性，用于变量选择。

**参数**:
- `y_data`: 因变量数据
- `x_data`: 自变量数据
- `feature_names`: 特征名称（可选）
- `method`: 分析方法（默认random_forest）
- `top_k`: 返回前k个重要特征（默认5）
- `file_content`: 文件内容（可选）
- `file_format`: 文件格式（auto/csv/json）

**使用示例**:
```python
result = await feature_importance_analysis_tool(
    y_data=[10, 12, 15, 18, 20, 22, 25, 28],
    x_data=[[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10]],
    feature_names=["特征1", "特征2", "特征3"],
    top_k=3
)
```

## 📁 文件输入最佳实践

### CSV文件格式要求

**多变量数据CSV**:
```csv
变量1,变量2,变量3
1.0,2.0,3.0
4.0,5.0,6.0
7.0,8.0,9.0
```

**时间序列数据CSV**:
```csv
日期,数值
2024-01-01,100
2024-01-02,102
2024-01-03,101
```

**面板数据CSV**:
```csv
实体ID,时间,变量1,变量2,因变量
A,2020,10,20,100
A,2021,12,22,120
B,2020,8,18,80
B,2021,9,19,90
```

### JSON文件格式要求

**多变量数据JSON**:
```json
{
  "变量1": [1.0, 4.0, 7.0],
  "变量2": [2.0, 5.0, 8.0],
  "变量3": [3.0, 6.0, 9.0]
}
```

**时间序列数据JSON**:
```json
{
  "时间戳": ["2024-01-01", "2024-01-02", "2024-01-03"],
  "数值": [100, 102, 101]
}
```

## 🎯 使用技巧

### 1. 数据预处理
- 确保所有数据为数值类型
- 处理缺失值和异常值
- 标准化或归一化数据（如需要）

### 2. 参数调优
- 从简单模型开始，逐步增加复杂度
- 使用交叉验证选择最优参数
- 关注模型诊断指标

### 3. 结果解释
- 结合业务背景解释统计结果
- 关注统计显著性和经济显著性
- 使用可视化辅助理解

### 4. 性能优化
- 对于大数据集，使用文件输入
- 合理设置模型参数避免过拟合
- 利用缓存机制提高重复计算