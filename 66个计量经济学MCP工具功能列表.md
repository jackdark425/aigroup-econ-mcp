# aigroup-econ-mcp工具功能列表

| 工具名称 | 功能描述 |
|:-:|:-:|
| **基础参数估计** | |
| `basic_parametric_estimation_ols` | OLS回归分析，支持直接数据输入或文件路径，包含系数估计、标准误、置信区间等 |
| `basic_parametric_estimation_mle` | 最大似然估计，支持正态、泊松、指数分布，包含参数估计和置信区间 |
| `basic_parametric_estimation_gmm` | 广义矩估计方法，支持工具变量，包含J检验和参数估计 |
| **时间序列与面板数据** | |
| `time_series_arima_model` | ARIMA时间序列模型，支持(p,d,q)参数设定和多步预测 |
| `time_series_exponential_smoothing` | 指数平滑模型，支持趋势和季节性成分，包含多步预测 |
| `time_series_garch_model` | GARCH波动率模型，支持(p,q)参数设定，用于条件方差建模 |
| `time_series_unit_root_tests` | 单位根检验，支持ADF、PP、KPSS检验，检验时间序列平稳性 |
| `time_series_var_svar_model` | VAR/SVAR模型，支持多元时间序列分析和结构向量自回归 |
| `time_series_cointegration_analysis` | 协整分析，支持Johansen和Engle-Granger方法，检验长期均衡关系 |
| `panel_data_dynamic_model` | 动态面板数据模型，支持差分GMM和系统GMM方法 |
| `panel_data_diagnostics` | 面板数据诊断检验，包含Hausman检验、F检验、LM检验等 |
| `panel_var_model` | 面板VAR模型，支持个体和时间效应的向量自回归分析 |
| `structural_break_tests` | 结构断点检验，支持Chow检验、Quandt-Andrews检验等 |
| `time_varying_parameter_models` | 时变参数模型，支持TAR、STAR、马尔可夫转换模型 |
| **因果推断方法** | |
| `causal_difference_in_differences` | 双重差分法(DID)，用于政策干预效果评估，验证平行趋势假设 |
| `causal_instrumental_variables` | 工具变量法(IV/2SLS)，解决内生性问题，需要工具变量相关性和外生性 |
| `causal_propensity_score_matching` | 倾向得分匹配(PSM)，控制混杂因素，支持最近邻匹配等方法 |
| `causal_fixed_effects` | 固定效应模型，控制不随时间变化的个体异质性，适用于面板数据 |
| `causal_random_effects` | 随机效应模型，假设个体效应随机，适用于面板数据分析 |
| `causal_regression_discontinuity` | 回归断点设计(RDD)，利用连续变量的断点进行因果识别 |
| `causal_synthetic_control` | 合成控制法，构造反事实对照组，用于政策评估和比较案例研究 |
| `causal_event_study` | 事件研究法，分析处理前后的动态效应，验证平行趋势假设 |
| `causal_triple_difference` | 三重差分法(DDD)，进一步控制混杂因素，用于复杂政策评估 |
| `causal_mediation_analysis` | 中介效应分析，识别因果机制，基于Baron-Kenny方法 |
| `causal_moderation_analysis` | 调节效应分析，检验条件效应，通过交互项回归实现 |
| `causal_control_function` | 控制函数法，解决内生性问题，适用于非线性模型的内生性处理 |
| `causal_first_difference` | 一阶差分模型，消除固定效应，适用于短面板数据分析 |
| **机器学习方法** | |
| `ml_random_forest` | 随机森林分析，支持回归和分类问题，包含特征重要性分析 |
| `ml_gradient_boosting` | 梯度提升机分析，支持sklearn和XGBoost算法，高性能集成学习 |
| `ml_support_vector_machine` | 支持向量机分析，支持回归和分类，多种核函数选择 |
| `ml_neural_network` | 神经网络(MLP)分析，可配置网络结构，支持回归和分类 |
| `ml_kmeans_clustering` | K均值聚类分析，无监督学习，包含聚类质量评估 |
| `ml_hierarchical_clustering` | 层次聚类分析，树状图可视化，支持多种链接方法 |
| `ml_double_machine_learning` | 双重/去偏机器学习，用于因果推断中的处理效应估计 |
| `ml_causal_forest` | 因果森林分析，用于异质性治疗效应估计，支持诚实估计 |
| **微观计量模型** | |
| `micro_logit` | Logistic回归模型，适用于二元因变量，基于最大似然估计 |
| `micro_probit` | Probit回归模型，基于正态分布假设，适用于二元因变量 |
| `micro_multinomial_logit` | 多项Logit模型，适用于多分类问题，基于最大似然估计 |
| `micro_poisson` | 泊松回归模型，适用于计数数据，处理离散因变量 |
| `micro_negative_binomial` | 负二项回归模型，处理过度离散问题，适用于计数数据 |
| `micro_tobit` | Tobit模型(截断回归)，适用于受限因变量，处理截断数据 |
| `micro_heckman` | Heckman样本选择模型，处理样本选择偏差，两阶段估计 |
| **分布分析与分解** | |
| `decomposition_oaxaca_blinder` | Oaxaca-Blinder分解，分解两组之间的平均差异为禀赋效应和系数效应 |
| `decomposition_variance_anova` | 方差分解(ANOVA)，单因素方差分析，组间方差vs组内方差 |
| `decomposition_time_series` | 时间序列分解，趋势-季节-随机分解，支持加法/乘法模型 |
| **缺失数据处理** | |
| `missing_data_simple_imputation` | 简单插补方法，支持均值/中位数/众数/常数填充 |
| `missing_data_multiple_imputation` | 多重插补(MICE)，迭代插补算法，基于链式方程 |
| **模型规范与诊断** | |
| `model_diagnostic_tests` | 模型诊断检验，包含异方差性、自相关、正态性、多重共线性检验 |
| `generalized_least_squares` | 广义最小二乘法(GLS)，处理异方差性和自相关问题 |
| `weighted_least_squares` | 加权最小二乘法(WLS)，处理异方差性，支持自定义权重 |
| `robust_errors_regression` | 稳健标准误回归，异方差稳健标准误(HC0, HC1, HC2, HC3) |
| `model_selection_criteria` | 模型选择准则，包含AIC、BIC、HQIC、交叉验证等 |
| `regularized_regression` | 正则化回归，支持Ridge、LASSO、Elastic Net，用于特征选择 |
| `simultaneous_equations_model` | 联立方程模型(2SLS)，处理内生性问题，需要有效工具变量 |
| **非参数与半参数方法** | |
| `nonparametric_kernel_regression` | 核回归(非参数)，支持多种核函数和带宽选择方法 |
| `nonparametric_quantile_regression` | 分位数回归，分析条件分位数，稳健于异常值 |
| `nonparametric_spline_regression` | 样条回归，灵活的非线性拟合，支持样条基函数 |
| `nonparametric_gam_model` | 广义可加模型(GAM)，多个平滑函数的加和，灵活的函数形式 |
| **空间计量经济学** | |
| `spatial_weights_matrix` | 空间权重矩阵构建，支持Queen、Rook、KNN、距离带、核函数等 |
| `spatial_morans_i_test` | Moran's I空间自相关检验，全局空间聚集性分析 |
| `spatial_gearys_c_test` | Geary's C空间自相关检验，另一种全局空间相关性度量 |
| `spatial_local_moran_lisa` | 局部Moran's I(LISA)分析，识别HH、LL、HL、LH聚类 |
| `spatial_regression_model` | 空间回归模型，支持SAR、SEM、SDM模型，基于最大似然估计 |
| `spatial_gwr_model` | 地理加权回归(GWR)，局部回归系数估计，捕捉空间异质性 |
| **统计推断技术** | |
| `inference_bootstrap` | Bootstrap重采样推断，置信区间估计，支持多种统计量 |
| `inference_permutation_test` | 置换检验(非参数)，两样本比较，支持均值/中位数/方差比检验 |

## 工具特点总结

1. **数据输入灵活性**：所有工具都支持直接数据输入或文件路径输入
2. **输出格式统一**：支持JSON、文本等多种输出格式，可保存结果到文件
3. **参数可配置**：大多数工具都提供丰富的参数配置选项
4. **错误处理完善**：包含完整的错误处理和日志记录机制
5. **理论基础扎实**：基于成熟的计量经济学理论和统计方法
6. **应用场景广泛**：覆盖经济学研究的各个领域和数据类型

## 使用说明

- 所有工具都支持异步调用，可通过MCP协议访问
- 数据格式要求详见各工具的文档说明
- 参数默认值经过优化，适用于大多数应用场景
- 结果包含详细的统计指标和诊断信息