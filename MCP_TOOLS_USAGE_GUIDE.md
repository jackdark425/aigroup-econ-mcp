# AIGroup-Econ-MCP 工具使用指南

## 概述

AIGroup-Econ-MCP 服务器提供20个计量经济学分析工具，支持多种数据输入方式。所有工具都支持CSV和JSON格式的文件输入。

## 数据输入方式

每个工具支持三种输入方式（按优先级）：

1. **file_path**: 文件路径（相对于工作目录）
2. **file_content**: 文件内容字符串
3. **直接数据参数**: 如 `data`, `y_data`, `x_data` 等

## 工具分类

### 1. 基础统计工具（5个）

#### 1.1 descriptive_statistics
**功能**: 计算描述性统计量

**输入参数**:
- `file_path`: CSV/JSON文件路径（可选）
- `file_content`: 文件内容（可选）
- `data`: 数据字典 `{"变量名": [数据列表]}` （可选）

**CSV格式要求**:
```csv
var1,var2,var3
1.2,2.3,3.4
2.1,3.2,4.3
```

**输出**: 均值、标准差、最小值、最大值、中位数、偏度、峰度、相关系数矩阵

---

#### 1.2 ols_regression
**功能**: OLS回归分析

**输入参数**:
- `file_path`: CSV/JSON文件路径
- `y_data`: 因变量列表
- `x_data`: 自变量矩阵（二维列表）
- `feature_names`: 特征名称列表

**CSV格式**: 最后一列为因变量，其余列为自变量
```csv
x1,x2,x3,y
1.0,2.0,3.0,10.0
2.0,3.0,4.0,15.0
```

**输出**: R²、调整R²、F统计量、回归系数、置信区间

---

#### 1.3 hypothesis_testing
**功能**: 假设检验

**输入参数**:
- `data`: 第一组数据
- `data2`: 第二组数据（双样本检验）
- `test_type`: 检验类型（"t_test" 或 "adf"）

**输出**: 检验统计量、p值、显著性判断、置信区间

---

#### 1.4 time_series_analysis
**功能**: 时间序列分析

**输入参数**:
- `data`: 时间序列数据（单变量）

**输出**: ADF检验、ACF、PACF、平稳性判断

---

#### 1.5 correlation_analysis
**功能**: 相关性分析

**输入参数**:
- `data`: 多变量数据字典
- `method`: 相关系数类型（"pearson", "spearman", "kendall"）

**输出**: 相关系数矩阵

---

### 2. 面板数据工具（4个）

**重要提示**: 面板数据工具需要CSV文件包含特定的列名关键词：

#### 列名要求

1. **实体ID列**: 列名必须包含以下关键词之一
   - `id`, `entity`, `firm`, `company`, `country`, `region`
   - 示例: `entity_id`, `firm_id`, `country`

2. **时间列**: 列名必须包含以下关键词之一
   - `time`, `date`, `year`, `month`, `day`, `period`, `quarter`
   - 示例: `time_period`, `year`, `date`

3. **数据列**: 其他列将被识别为数据变量
   - 最后一列默认为因变量
   - 其余列为自变量

#### 2.1 panel_fixed_effects
**功能**: 固定效应模型

**CSV格式示例**:
```csv
entity_id,time_period,x1,x2,y
A,2010,100,3.5,150
A,2011,105,4.2,155
B,2010,95,3.2,145
B,2011,100,3.8,150
```

**输入参数**:
- `file_path`: CSV文件路径
- `entity_effects`: 是否包含个体效应（默认True）
- `time_effects`: 是否包含时间效应（默认False）

**输出**: R²、回归系数、F统计量

---

#### 2.2 panel_random_effects
**功能**: 随机效应模型

**格式要求**: 与fixed_effects相同

**输出**: R²、回归系数、组间R²

---

#### 2.3 panel_hausman_test
**功能**: Hausman检验（选择固定效应还是随机效应）

**格式要求**: 与fixed_effects相同

**输出**: 检验统计量、p值、模型推荐

---

#### 2.4 panel_unit_root_test
**功能**: 面板单位根检验

**格式要求**: 需要entity_id和time列，加上一个数据变量

**输入参数**:
- `test_type`: 检验类型（默认"levinlin"）

**输出**: 检验统计量、p值、平稳性判断

---

### 3. 高级时间序列工具（5个）

#### 3.1 var_model_analysis
**功能**: VAR模型分析

**输入参数**:
- `data`: 多变量时间序列字典
- `max_lags`: 最大滞后阶数（默认5）
- `ic`: 信息准则（"aic", "bic", "hqic"）

**最小数据量**: 至少20个观测值

**输出**: 滞后阶数、AIC值

---

#### 3.2 vecm_model_analysis
**功能**: VECM模型（协整分析）

**输入参数**:
- `data`: 多变量时间序列
- `coint_rank`: 协整秩（默认1）
- `deterministic`: 确定性项（默认"co"）

**输出**: 协整秩、AIC值

---

#### 3.3 garch_model_analysis
**功能**: GARCH模型（波动率建模）

**输入参数**:
- `data`: 单变量时间序列
- `order`: GARCH阶数（默认(1,1)）
- `dist`: 分布类型（默认"normal"）

**输出**: 持久性参数

---

#### 3.4 state_space_model_analysis
**功能**: 状态空间模型

**输入参数**:
- `data`: 单变量时间序列
- `state_dim`: 状态维度
- `trend`: 是否包含趋势
- `seasonal`: 是否包含季节性

**输出**: AIC值

---

#### 3.5 variance_decomposition_analysis
**功能**: 方差分解

**输入参数**:
- `data`: 多变量时间序列
- `periods`: 预测期数（默认10）

**输出**: 方差分解结果

---

### 4. 机器学习工具（6个）

#### 4.1 random_forest_regression_analysis
**功能**: 随机森林回归

**输入参数**:
- `y_data`, `x_data`: 数据或文件路径
- `n_estimators`: 树的数量（默认100）
- `max_depth`: 最大深度（可选）

**输出**: R²、特征重要性

---

#### 4.2 gradient_boosting_regression_analysis
**功能**: 梯度提升树回归

**输入参数**:
- `n_estimators`: 迭代次数（默认100）
- `learning_rate`: 学习率（默认0.1）
- `max_depth`: 最大深度（默认3）

**输出**: R²得分

---

#### 4.3 lasso_regression_analysis
**功能**: Lasso回归（L1正则化）

**输入参数**:
- `alpha`: 正则化强度（默认1.0）

**输出**: R²、非零系数

---

#### 4.4 ridge_regression_analysis
**功能**: Ridge回归（L2正则化）

**输入参数**:
- `alpha`: 正则化强度（默认1.0）

**输出**: R²得分

---

#### 4.5 cross_validation_analysis
**功能**: 交叉验证

**输入参数**:
- `model_type`: 模型类型（"random_forest", "lasso", "ridge"）
- `cv_folds`: 折数（默认5）
- `scoring`: 评分标准（默认"r2"）

**输出**: 平均得分、标准差

---

#### 4.6 feature_importance_analysis_tool
**功能**: 特征重要性分析

**输入参数**:
- `method`: 方法（"random_forest", "gradient_boosting"）
- `top_k`: 返回前K个重要特征（默认5）

**输出**: 特征重要性排序

---

## CSV文件格式规范

### 基本要求
- 第一行为列名（表头）
- 数据从第二行开始
- 使用逗号分隔（支持自动检测制表符、分号等）

### 示例格式

**1. 多变量数据**:
```csv
var1,var2,var3,var4
1.2,2.3,3.4,4.5
2.1,3.2,4.3,5.4
3.0,4.1,5.2,6.3
```

**2. 回归数据（最后一列为因变量）**:
```csv
feature1,feature2,feature3,target
1.0,2.0,3.0,10.0
2.0,3.0,4.0,15.0
3.0,4.0,5.0,20.0
```

**3. 面板数据**:
```csv
entity_id,time_period,x1,x2,y
A,2010,100,3.5,150
A,2011,105,4.2,155
B,2010,95,3.2,145
B,2011,100,3.8,150
```

## JSON文件格式规范

支持两种JSON格式：

**格式1: 变量-数据字典**
```json
{
  "var1": [1.2, 2.1, 3.0],
  "var2": [2.3, 3.2, 4.1],
  "var3": [3.4, 4.3, 5.2]
}
```

**格式2: 记录数组**
```json
[
  {"var1": 1.2, "var2": 2.3, "var3": 3.4},
  {"var1": 2.1, "var2": 3.2, "var3": 4.3},
  {"var1": 3.0, "var2": 4.1, "var3": 5.2}
]
```

## 常见问题

### 1. 面板数据识别失败
**问题**: "面板数据需要包含实体ID和时间标识变量"

**解决方案**:
- 确保列名包含关键词
  - 实体列: `entity_id`, `firm_id`, `company`, `country`, 等
  - 时间列: `time_period`, `year`, `date`, `quarter`, 等
- 检查列名大小写（系统会自动转换为小写匹配）

### 2. 数据量不足
**问题**: "Data length insufficient, need at least 20 observations"

**解决方案**: 
- VAR/VECM模型需要至少20个观测值
- 增加数据量或使用其他模型

### 3. 变量数量不足
**问题**: "回归分析至少需要2个变量"

**解决方案**:
- 检查CSV文件是否有足够的列
- 确保至少有1个自变量和1个因变量

## 测试用例

### 测试1: 基础统计
```python
# 使用文件路径
result = await descriptive_statistics(file_path="data.csv")

# 或使用直接数据
result = await descriptive_statistics(
    data={"x": [1, 2, 3], "y": [4, 5, 6]}
)
```

### 测试2: 回归分析
```python
result = await ols_regression(
    file_path="regression_data.csv"
)
```

### 测试3: 面板数据
```python
result = await panel_fixed_effects(
    file_path="panel_test_data.csv",
    entity_effects=True,
    time_effects=False
)
```

## 工具性能

| 工具类型 | 最小数据量 | 推荐数据量 | 执行时间 |
|---------|----------|----------|---------|
| 基础统计 | 5 | 30+ | < 1s |
| OLS回归 | 10 | 30+ | < 1s |
| 面板数据 | 15 | 50+ | 1-2s |
| VAR模型 | 20 | 50+ | 2-5s |
| 随机森林 | 20 | 100+ | 2-10s |

## 版本信息

- **服务器版本**: 0.2.0
- **支持的工具数量**: 20
- **文件格式**: CSV, JSON
- **编码**: UTF-8

## 技术支持

如有问题，请参考：
- CSV_FILE_PATH_USAGE.md - CSV文件输入使用指南
- CSV_PATH_SUPPORT_FINAL_REPORT.md - 文件支持功能报告
- MCP_TOOLS_TEST_REPORT.md - 工具测试报告