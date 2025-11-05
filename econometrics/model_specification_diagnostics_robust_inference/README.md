# 模型规范、诊断和稳健推断工具

本模块提供了完整的模型规范检验、诊断测试和稳健推断方法工具集。

## 工具列表

### 1. 模型诊断检验 (Model Diagnostic Tests)
**工具名称**: `model_diagnostic_tests`

**功能**: 执行综合的模型诊断测试，包括：
- 异方差检验（Breusch-Pagan、White检验）
- 自相关检验（Durbin-Watson检验）
- 正态性检验（Jarque-Bera检验）
- 多重共线性诊断（方差膨胀因子VIF）

**使用场景**: 
- OLS回归后的模型验证
- 检测模型假设是否满足
- 识别数据质量问题

### 2. 广义最小二乘法 (Generalized Least Squares - GLS)
**工具名称**: `generalized_least_squares`

**功能**: 处理异方差性和自相关的回归方法

**主要特点**:
- 可指定误差项协方差矩阵
- 在满足GLS假设时比OLS更有效
- 适用于存在异方差或自相关的数据

**使用场景**:
- 时间序列数据回归
- 存在已知异方差模式的数据

### 3. 加权最小二乘法 (Weighted Least Squares - WLS)
**工具名称**: `weighted_least_squares`

**功能**: 使用权重处理已知异方差性的回归方法

**主要特点**:
- 需要提供观测值权重
- 权重通常为方差的倒数
- 适用于分组数据或调查数据

**使用场景**:
- 调查数据分析
- 分组数据回归
- 已知误差方差的数据

### 4. 稳健标准误回归 (Robust Standard Errors)
**工具名称**: `robust_errors_regression`

**功能**: 计算异方差稳健的标准误

**主要特点**:
- 支持多种协方差矩阵类型（HC0、HC1、HC2、HC3）
- 不改变系数估计，只调整标准误
- 在存在异方差时提供有效推断

**使用场景**:
- 横截面数据分析
- 异方差问题明显但形式未知
- 需要稳健推断的场景

### 5. 模型选择准则 (Model Selection Criteria)
**工具名称**: `model_selection_criteria`

**功能**: 计算多种模型选择信息准则

**提供指标**:
- AIC（赤池信息准则）
- BIC（贝叶斯信息准则）
- HQIC（汉南-奎因信息准则）
- 交叉验证得分（可选）

**使用场景**:
- 比较不同模型规格
- 变量选择
- 确定最优模型

### 6. 正则化回归 (Regularized Regression)
**工具名称**: `regularized_regression`

**功能**: 处理多重共线性和高维数据的正则化方法

**支持方法**:
- 岭回归（Ridge）：L2惩罚
- LASSO：L1惩罚，可进行变量选择
- 弹性网络（Elastic Net）：L1和L2的组合

**使用场景**:
- 高维数据回归
- 变量选择
- 处理多重共线性

### 7. 联立方程模型 (Simultaneous Equations Model)
**工具名称**: `simultaneous_equations_model`

**功能**: 两阶段最小二乘法（2SLS）处理联立方程系统

**主要特点**:
- 处理内生性问题
- 需要有效的工具变量
- 支持多方程系统

**使用场景**:
- 供需模型
- 宏观经济模型
- 存在双向因果关系的模型

## 使用示例

### 诊断检验示例
```python
# 使用MCP工具
{
  "y_data": [1.0, 2.0, 3.0, 4.0, 5.0],
  "x_data": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]],
  "feature_names": ["x1", "x2"],
  "constant": true
}
```

### 稳健标准误回归示例
```python
{
  "y_data": [1.0, 2.0, 3.0, 4.0, 5.0],
  "x_data": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]],
  "cov_type": "HC1",
  "confidence_level": 0.95
}
```

### 正则化回归示例
```python
{
  "y_data": [1.0, 2.0, 3.0, 4.0, 5.0],
  "x_data": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]],
  "method": "ridge",
  "alpha": 1.0
}
```

## 技术细节

### 实现架构
- **核心算法**: 位于各子模块的 `*_model.py` 文件
- **MCP适配器**: `tools/model_specification_adapter.py`
- **工具注册**: `tools/mcp_tool_groups/model_specification_tools.py`

### 依赖库
- `statsmodels`: 用于统计模型和诊断检验
- `scikit-learn`: 用于正则化方法
- `linearmodels`: 用于联立方程模型
- `numpy`, `pandas`: 基础数据处理

### 数据格式支持
- **输入**: JSON、CSV、Excel、TXT
- **输出**: JSON、Markdown、HTML

## 注意事项

1. **诊断检验**: 应在OLS回归后使用，检验模型假设
2. **GLS/WLS**: 需要正确指定协方差矩阵或权重
3. **稳健标准误**: 不改变系数估计，仅影响推断
4. **正则化**: alpha参数需要通过交叉验证选择
5. **联立方程**: 需要有效且足够数量的工具变量

## 贡献者
AIGroup Economics Team

## 许可证
MIT License