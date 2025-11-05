# AIGroup计量经济学MCP工具完整指南

## 概述

本MCP服务器提供**21个**计量经济学分析工具，采用组件化架构设计，支持多种数据格式输入和输出。

## 服务器配置

```json
{
  "server_name": "aigroup-econ-mcp",
  "version": "2.2.0-component",
  "architecture": "Component-Based",
  "tool_groups": 3,
  "total_tools": 21,
  "tools": [
    "basic_parametric_estimation_ols",
    "basic_parametric_estimation_mle",
    "basic_parametric_estimation_gmm",
    "model_diagnostic_tests",
    "generalized_least_squares",
    "weighted_least_squares",
    "robust_errors_regression",
    "model_selection_criteria",
    "regularized_regression",
    "simultaneous_equations_model",
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

### 模型规范、诊断和稳健推断工具 (7个) 🆕

4. **模型诊断检验 (model_diagnostic_tests)**
   - 核心算法: econometrics/model_specification_diagnostics_robust_inference/diagnostic_tests/
   - 功能: 异方差检验(Breusch-Pagan, White)、自相关检验(Durbin-Watson)、正态性检验(Jarque-Bera)、多重共线性诊断(VIF)
   - 输入方式: 直接数据(y_data + x_data) 或 文件(file_path)
   - 支持格式: txt/json/csv/excel

5. **广义最小二乘法 (generalized_least_squares)**
   - 核心算法: econometrics/model_specification_diagnostics_robust_inference/generalized_least_squares/
   - 功能: 处理异方差性和自相关的GLS回归
   - 特点: 可指定误差项协方差矩阵
   - 输入方式: 直接数据 或 文件

6. **加权最小二乘法 (weighted_least_squares)**
   - 核心算法: econometrics/model_specification_diagnostics_robust_inference/weighted_least_squares/
   - 功能: 使用权重处理已知异方差性
   - 特点: 需要提供观测值权重（通常为方差的倒数）
   - 输入方式: 直接数据 或 文件

7. **稳健标准误回归 (robust_errors_regression)**
   - 核心算法: econometrics/model_specification_diagnostics_robust_inference/robust_errors/
   - 功能: 计算异方差稳健的标准误
   - 支持类型: HC0, HC1, HC2, HC3
   - 特点: 不改变系数估计，只调整标准误

8. **模型选择准则 (model_selection_criteria)**
   - 核心算法: econometrics/m odel_specification_diagnostics_robust_inference/model_selection/
   - 功能: 计算AIC、BIC、HQIC信息准则
   - 附加功能: K折交叉验证、留一法交叉验证
   - 用途: 模型比较和变量选择

9. **正则化回归 (regularized_regression)**
   - 核心算法: econometrics/model_specification_diagnostics_robust_inference/regularization/
   - 方法: Ridge回归(L2)、LASSO(L1)、Elastic Net(L1+L2)
   - 功能: 处理多重共线性和高维数据
   - 特点: 可进行变量选择（LASSO）

10. **联立方程模型 (simultaneous_equations_model)**
    - 核心算法: econometrics/model_specification_diagnostics_robust_inference/simultaneous_equations/
    - 方法: 两阶段最小二乘法(2SLS)
    - 功能: 处理联立方程系统和内生性问题
    - 要求: 需要有效的工具变量

### 时间序列工具 (6个)

11. **ARIMA模型 (time_series_arima_model)**
    - 参数: (p,d,q) 阶数
    - 功能: 多步预测

12. **指数平滑模型 (time_series_exponential_smoothing)**
    - 组件: 趋势项, 季节项
    - 功能: 多步预测

13. **GARCH模型 (time_series_garch_model)**
    - 功能: 条件方差建模
    - 参数: (p,q) 阶数

14. **单位根检验 (time_series_unit_root_tests)**
    - 检验方法: ADF, PP, KPSS
    - 功能: 平稳性检验

15. **VAR/SVAR模型 (time_series_var_svar_model)**
    - 模型类型: VAR, SVAR
    - 功能: 多变量时间序列分析

16. **协整分析 (time_series_cointegration_analysis)**
    - 检验方法: Engle-Granger, Johansen
    - 模型: VECM
    - 功能: 长期均衡关系分析

### 面板数据工具 (3个)

17. **动态面板模型 (panel_data_dynamic_model)**
    - 模型类型: 差分GMM, 系统GMM
    - 数据: 横截面和时间序列数据

18. **面板数据诊断测试 (panel_data_diagnostics)**
    - 检验方法: Hausman, Pooling F, LM, 组内相关性
    - 功能: 模型选择 (FE vs RE vs Pooled)

19. **面板VAR模型 (panel_var_model)**
    - 功能: 面板向量自回归
    - 效应: 个体效应和时间效应

### 高级计量工具 (2个)

20. **结构断点检验 (structural_break_tests)**
    - 检验方法: Chow, Quandt-Andrews, Bai-Perron
    - 功能: 检测时间序列结构变化

21. **时变参数模型 (time_varying_parameter_models)**
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

#### 2. 模型诊断检验 🆕
```json
{
  "y_data": [1, 2, 3, 4, 5],
  "x_data": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
  "feature_names": ["X1", "X2"],
  "constant": true
}
```

#### 3. 稳健标准误回归 🆕
```json
{
  "y_data": [1, 2, 3, 4, 5],
  "x_data": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
  "cov_type": "HC1",
  "confidence_level": 0.95
}
```

#### 4. 正则化回归 🆕
```json
{
  "y_data": [1, 2, 3, 4, 5],
  "x_data": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
  "method": "ridge",
  "alpha": 1.0,
  "l1_ratio": 0.5
}
```

#### 5. 加权最小二乘法 🆕
```json
{
  "y_data": [1, 2, 3, 4, 5],
  "x_data": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
  "weights": [1.0, 0.8, 1.2, 0.9, 1.1],
  "confidence_level": 0.95
}
```

#### 6. ARIMA模型
```json
{
  "data": [1.2, 2.1, 3.4, 4.2, 5.1, 6.3, 7.2, 8.1, 9.4, 10.2],
  "order": [1, 1, 1],
  "forecast_steps": 3
}
```

#### 7. 动态面板模型
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

### 稳健标准误类型 🆕
- `HC0`: 白异方差一致性标准误
- `HC1`: 修正的HC0（小样本调整）
- `HC2`: 杠杆调整的标准误
- `HC3`: 杠杆调整的标准误（更稳健）

### 正则化方法 🆕
- `ridge`: 岭回归（L2惩罚）
- `lasso`: LASSO回归（L1惩罚，可变量选择）
- `elastic_net`: 弹性网络（L1+L2惩罚）

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

## 工具组分类

### 第一组：基础参数估计 (3个工具)
专注于基本的统计估计方法，适用于大多数标准回归分析场景。

### 第二组：模型规范、诊断和稳健推断 (7个工具) 🆕
提供全面的模型诊断、规范检验和稳健估计方法，确保模型的可靠性和有效性。

### 第三组：时间序列和面板数据 (11个工具)
涵盖时间序列分析、面板数据建模和高级计量方法。

## 架构信息

**架构**: Component-Based  
**版本**: 2.2.0  
**Python版本**: 3.8+  
**MCP协议**: FastMCP  
**工具组数量**: 3  
**总工具数**: 21  
**文件格式**: txt, json, csv, excel (.xlsx, .xls)  
**输出格式**: json, markdown, txt

## 优势特点

- **组件化设计**: 工具按功能分组，便于维护和扩展
- **模块化**: 每个工具组独立管理
- **DRY原则**: 复用核心算法，无重复代码
- **易于扩展**: 轻松添加新工具类别
- **性能优化**: 高效的数据处理和计算
- **全面诊断**: 新增完整的模型诊断和稳健推断工具 🆕
- **稳健性**: 支持多种稳健估计方法处理数据问题 🆕

## 使用建议

1. **数据准备**: 确保数据格式正确，特别是多维数组的嵌套结构
2. **参数选择**: 根据具体分析需求选择合适的模型参数
3. **模型诊断**: 在进行推断前使用诊断工具检验模型假设 🆕
4. **稳健性检查**: 对于可能存在异方差的数据使用稳健标准误 🆕
5. **输出格式**: 根据后续处理需求选择合适的输出格式
6. **错误处理**: 注意工具可能返回的错误信息，如矩阵奇异等

## 典型工作流程 🆕

### 标准回归分析流程
1. 使用 `basic_parametric_estimation_ols` 进行OLS回归
2. 使用 `model_diagnostic_tests` 检验模型假设
3. 如发现异方差：
   - 使用 `robust_errors_regression` 获取稳健标准误，或
   - 使用 `weighted_least_squares` 或 `generalized_least_squares`
4. 使用 `model_selection_criteria` 进行模型比较

### 高维数据分析流程
1. 使用 `regularized_regression` 处理多重共线性
2. 通过LASSO进行变量选择
3. 使用交叉验证选择最优alpha参数

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

# 模型诊断检验示例 🆕
diagnostic_result = await mcp.model_diagnostic_tests(
    y_data=[1, 2, 3, 4, 5],
    x_data=[[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
    feature_names=["X1", "X2"],
    constant=True
)

# 稳健标准误回归示例 🆕
robust_result = await mcp.robust_errors_regression(
    y_data=[1, 2, 3, 4, 5],
    x_data=[[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
    cov_type="HC1",
    confidence_level=0.95
)

# 正则化回归示例 🆕
ridge_result = await mcp.regularized_regression(
    y_data=[1, 2, 3, 4, 5],
    x_data=[[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
    method="ridge",
    alpha=1.0
)

# ARIMA模型示例
arima_result = await mcp.time_series_arima_model(
    data=[1.2, 2.1, 3.4, 4.2, 5.1, 6.3, 7.2, 8.1, 9.4, 10.2],
    order=[1, 1, 1],
    forecast_steps=3,
    output_format="json"
)
```

## 更新历史

### v2.2.0 (当前版本) 🆕
- 新增7个模型规范、诊断和稳健推断工具
- 总工具数从14个增加到21个
- 增强了模型诊断和稳健推断能力
- 添加了正则化方法支持

### v2.1.0
- 提供14个基础工具
- 实现组件化架构
- 支持多种数据格式

---

这个完整指南包含了所有必要信息，帮助大模型正确理解和使用所有**21个**计量经济学工具。