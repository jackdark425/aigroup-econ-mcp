# AIGroup计量经济学MCP工具完整指南

## 概述

本MCP服务器提供14个计量经济学分析工具，采用组件化架构设计，支持多种数据格式输入和输出。

## 服务器配置

```json
{
  "server_name": "aigroup-econ-mcp",
  "version": "2.1.0-component",
  "architecture": "Component-Based",
  "tools": [
    "basic_parametric_estimation_ols",
    "basic_parametric_estimation_mle",
    "basic_parametric_estimation_gmm",
    "time_series_arima_model",
    "time_series_exponential_smoothing",
    "time_series_garch_model",
    "time_series_unit_root_tests",
    "time_series_var_svar_model",
    "time_series_cointegration_analysis",
    "panel_data_dynamic_model",
    "panel_data_diagnostics",
    "panel_var_model",
    "structural_break_tests",
    "time_varying_parameter_models"
  ],
  "description": "Econometrics MCP Tools with component-based architecture"
}
```

## 工具概览

### 基础参数估计工具 (3个)

1. **OLS回归分析 (basic_parametric_estimation_ols)**
   - 核心算法: econometrics/basic_parametric_estimation/ols/ols_model.py
   - 输入方式: 直接数据(y_data + x_data) 或 文件(file_path)
   - 支持格式: txt/json/csv/excel

2. **最大似然估计 (basic_parametric_estimation_mle)**
   - 核心算法: econometrics/basic_parametric_estimation/mle/mle_model.py
   - 输入方式: 直接数据(data) 或 文件(file_path)
   - 分布类型: normal, poisson, exponential
   - 支持格式: txt/json/csv/excel

3. **广义矩估计 (basic_parametric_estimation_gmm)**
   - 核心算法: econometrics/basic_parametric_estimation/gmm/gmm_model.py
   - 输入方式: 直接数据(y_data + x_data) 或 文件(file_path)
   - 已修复: j_p_value bug
   - 支持格式: txt/json/csv/excel

### 时间序列工具 (6个)

4. **ARIMA模型 (time_series_arima_model)**
   - 参数: (p,d,q) 阶数
   - 功能: 多步预测

5. **指数平滑模型 (time_series_exponential_smoothing)**
   - 组件: 趋势项, 季节项
   - 功能: 多步预测

6. **GARCH模型 (time_series_garch_model)**
   - 功能: 条件方差建模
   - 参数: (p,q) 阶数

7. **单位根检验 (time_series_unit_root_tests)**
   - 检验方法: ADF, PP, KPSS
   - 功能: 平稳性检验

8. **VAR/SVAR模型 (time_series_var_svar_model)**
   - 模型类型: VAR, SVAR
   - 功能: 多变量时间序列分析

9. **协整分析 (time_series_cointegration_analysis)**
   - 检验方法: Engle-Granger, Johansen
   - 模型: VECM
   - 功能: 长期均衡关系分析

### 面板数据工具 (3个)

10. **动态面板模型 (panel_data_dynamic_model)**
    - 模型类型: 差分GMM, 系统GMM
    - 数据: 横截面和时间序列数据

11. **面板数据诊断测试 (panel_data_diagnostics)**
    - 检验方法: Hausman, Pooling F, LM, 组内相关性
    - 功能: 模型选择 (FE vs RE vs Pooled)

12. **面板VAR模型 (panel_var_model)**
    - 功能: 面板向量自回归
    - 效应: 个体效应和时间效应

### 高级计量工具 (2个)

13. **结构断点检验 (structural_break_tests)**
    - 检验方法: Chow, Quandt-Andrews, Bai-Perron
    - 功能: 检测时间序列结构变化

14. **时变参数模型 (time_varying_parameter_models)**
    - 模型类型: TAR, STAR, Markov Switching
    - 功能: 基于阈值的机制转换

## 详细参数说明

### 通用参数格式

#### 输入数据格式
- **直接数据输入**: 使用 `y_data`, `x_data`, `data` 等参数
- **文件输入**: 使用 `file_path` 参数
- **支持的文件格式**: txt, json, csv, excel (.xlsx, .xls)

#### 输出格式选项
- `output_format`: json, markdown, txt
- `save_path`: 可指定输出文件路径保存结果

#### 通用配置参数
- `confidence_level`: 置信水平（默认0.95）
- `constant`: 是否包含常数项（默认true）
- `feature_names`: 特征名称列表

### 工具特定参数示例

#### 1. OLS回归分析
```json
{
  "y_data": [1, 2, 3, 4, 5],
  "x_data": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
  "feature_names": ["X1", "X2"],
  "constant": true,
  "confidence_level": 0.95
}
```

#### 2. ARIMA模型
```json
{
  "data": [1.2, 2.1, 3.4, 4.2, 5.1, 6.3, 7.2, 8.1, 9.4, 10.2],
  "order": [1, 1, 1],
  "forecast_steps": 3
}
```

#### 3. 动态面板模型
```json
{
  "y_data": [1.2, 2.1, 3.4, 4.2, 5.1],
  "x_data": [[1, 0.5], [2, 1.2], [3, 1.8], [4, 2.5], [5, 3.1]],
  "entity_ids": [1, 1, 1, 2, 2],
  "time_periods": [1, 2, 3, 1, 2],
  "model_type": "diff_gmm"
}
```

## 参数选项说明

### 分布类型 (MLE)
- `normal`: 正态分布
- `poisson`: 泊松分布  
- `exponential`: 指数分布

### 单位根检验类型
- `adf`: Augmented Dickey-Fuller检验
- `pp`: Phillips-Perron检验
- `kpss`: KPSS检验

### VAR/SVAR模型类型
- `var`: 向量自回归模型
- `svar`: 结构向量自回归模型

### 协整分析方法
- `johansen`: Johansen协整检验
- `engle-granger`: Engle-Granger协整检验

### 动态面板模型类型
- `diff_gmm`: 差分GMM模型
- `sys_gmm`: 系统GMM模型

### 面板诊断测试类型
- `hausman`: Hausman检验 (FE vs RE)
- `pooling_f`: Pooling F检验
- `lm`: LM检验
- `within_correlation`: 组内相关性检验

### 结构断点检验类型
- `chow`: Chow检验
- `quandt-andrews`: Quandt-Andrews检验
- `bai-perron`: Bai-Perron多重断点检验

### 时变参数模型类型
- `tar`: 门限自回归模型
- `star`: 平滑转换自回归模型
- `markov_switching`: 马尔科夫转换模型

### STAR类型
- `logistic`: Logistic转换函数
- `exponential`: 指数转换函数

## 架构信息

**架构**: Component-Based  
**Python版本**: 3.8+  
**MCP协议**: FastMCP  
**文件格式**: txt, json, csv, excel (.xlsx, .xls)  
**输出格式**: json, markdown, txt

## 优势特点

- **组件化设计**: 工具按功能分组，便于维护和扩展
- **模块化**: 每个工具组独立管理
- **DRY原则**: 复用核心算法，无重复代码
- **易于扩展**: 轻松添加新工具类别
- **性能优化**: 高效的数据处理和计算

## 使用建议

1. **数据准备**: 确保数据格式正确，特别是多维数组的嵌套结构
2. **参数选择**: 根据具体分析需求选择合适的模型参数
3. **输出格式**: 根据后续处理需求选择合适的输出格式
4. **错误处理**: 注意工具可能返回的错误信息，如矩阵奇异等

## 示例调用

```python
# OLS回归分析示例
result = await mcp.basic_parametric_estimation_ols(
    y_data=[1, 2, 3, 4, 5],
    x_data=[[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
    feature_names=["X1", "X2"],
    constant=True,
    output_format="json"
)

# ARIMA模型示例
result = await mcp.time_series_arima_model(
    data=[1.2, 2.1, 3.4, 4.2, 5.1, 6.3, 7.2, 8.1, 9.4, 10.2],
    order=[1, 1, 1],
    forecast_steps=3,
    output_format="json"
)
```

这个完整指南包含了所有必要信息，帮助大模型正确理解和使用所有14个计量经济学工具。