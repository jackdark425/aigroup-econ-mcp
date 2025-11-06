# 微观离散与受限数据模型

该模块实现了微观计量经济学中常用的离散选择模型和受限因变量模型，适用于因变量为分类、计数、截断等非连续情况的数据分析。本模块基于statsmodels等现有库构建，避免重复造轮子。

## 依赖库

- statsmodels >= 0.13.0
- numpy
- pandas
- scipy

安装依赖：
```bash
pip install statsmodels numpy pandas scipy
```

## 模型列表

### 离散选择模型

1. **LogitModel** - Logistic回归模型
   - 基于statsmodels的Logit模型实现
   - 适用于二元选择问题

2. **ProbitModel** - Probit回归模型
   - 基于statsmodels的Probit模型实现
   - 基于正态分布假设

3. **MultinomialLogit** - 多项Logit模型
   - 基于statsmodels的MNLogit模型实现
   - 适用于无序多分类选择问题

4. **OrderedLogit** - 有序Logit模型
   - 基于statsmodels的OrderedModel实现
   - 适用于有序多分类选择问题

5. **ConditionalLogit** - 条件Logit模型
   - 基于statsmodels的ConditionalLogit实现
   - 适用于配对选择等条件选择模型

### 受限因变量模型

1. **TobitModel** - Tobit模型（截断回归模型）
   - 基于statsmodels的Tobit模型实现
   - 适用于因变量在某个阈值处被截断的情况
   - 支持上下界截断

2. **HeckmanModel** - Heckman两阶段选择模型
   - 基于statsmodels构建的两阶段选择模型
   - 用于处理样本选择偏差问题

### 计数数据模型

1. **PoissonModel** - 泊松回归模型
   - 基于statsmodels的Poisson模型实现
   - 适用于计数数据建模
   - 假设均值等于方差

2. **NegativeBinomialModel** - 负二项回归模型
   - 基于statsmodels的NegativeBinomial模型实现
   - 适用于过度离散的计数数据
   - 允许方差大于均值

3. **ZeroInflatedPoissonModel** - 零膨胀泊松模型
   - 基于statsmodels的ZeroInflatedPoisson实现
   - 适用于零值过多的计数数据

4. **ZeroInflatedNegativeBinomialModel** - 零膨胀负二项模型
   - 基于statsmodels的ZeroInflatedNegativeBinomialP实现
   - 适用于零值过多且过度离散的计数数据

## 使用示例

### Logit模型示例

```python
from econometrics.specific_data_modeling.micro_discrete_limited_data import LogitModel
import numpy as np

# 生成示例数据
X = np.random.normal(0, 1, (1000, 2))
y = np.random.binomial(1, 1 / (1 + np.exp(-(0.5*X[:, 0] - 0.3*X[:, 1]))))

# 拟合模型
model = LogitModel()
model.fit(X, y)

# 预测
probabilities = model.predict_proba(X)
predictions = model.predict(X)

# 查看模型摘要
print(model.summary())
```

### Tobit模型示例

```python
from econometrics.specific_data_modeling.micro_discrete_limited_data import TobitModel
import numpy as np

# 生成示例数据
X = np.random.normal(0, 1, (1000, 2))
y_latent = 1.0*X[:, 0] - 0.5*X[:, 1] + np.random.normal(0, 0.5, 1000)
y = np.maximum(y_latent, 0)  # 左截断于0

# 拟合模型
model = TobitModel(lower_bound=0)
model.fit(X, y)

# 预测
predictions = model.predict(X)

# 查看模型摘要
print(model.summary())
```

### 泊松模型示例

```python
from econometrics.specific_data_modeling.micro_discrete_limited_data import PoissonModel
import numpy as np

# 生成示例数据
X = np.random.normal(0, 1, (1000, 2))
mu = np.exp(0.5*X[:, 0] - 0.3*X[:, 1])
y = np.random.poisson(mu)

# 拟合模型
model = PoissonModel()
model.fit(X, y)

# 预测
predictions = model.predict(X)

# 查看模型摘要
print(model.summary())
```

## 模型输出

所有模型都提供以下输出：
- 参数估计值 (通过.results_.params访问)
- 标准误差 (通过.results_.bse访问)
- t统计量 (通过.results_.tvalues访问)
- p值 (通过.results_.pvalues访问)
- 模型拟合统计量 (AIC, BIC, 对数似然等)
- 预测方法 (predict, predict_proba等)
- 模型摘要 (summary方法)

## 注意事项

1. 数据预处理：确保数据符合模型假设
2. 模型诊断：检查模型拟合优度和残差
3. 过度离散：对于计数数据，如果存在过度离散，应使用负二项模型而非泊松模型
4. 边界值处理：模型中已对数值边界情况进行处理，但用户仍需注意数据质量
5. 依赖库：确保安装了statsmodels等依赖库

## 参考文献

1. Cameron, A. C., & Trivedi, P. K. (2013). Regression analysis of count data. Cambridge university press.
2. Greene, W. H. (2003). Econometric analysis. Pearson Education India.
3. Wooldridge, J. M. (2010). Econometric analysis of cross section and panel data. MIT press.
4. Statsmodels Documentation: https://www.statsmodels.org/stable/index.html