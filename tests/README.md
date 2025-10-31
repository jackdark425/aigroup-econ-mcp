# aigroup-econ-mcp 测试文档

## 📋 测试结构概述

本项目采用分层测试架构，确保代码质量和功能稳定性。

### 测试目录结构

```
tests/
├── README.md                    # 测试文档
├── conftest.py                  # pytest配置和fixtures
├── run_tests.py                 # 测试运行脚本
├── test_unit/                   # 单元测试
│   ├── test_statistics.py       # 统计分析测试
│   ├── test_regression.py       # 回归分析测试
│   ├── test_time_series.py      # 时间序列测试
│   ├── test_panel_data.py       # 面板数据测试
│   └── test_machine_learning.py # 机器学习测试
├── test_integration/            # 集成测试
│   ├── test_econometrics.py     # 计量经济学集成测试
│   └── test_file_input.py       # 文件输入集成测试
└── test_data/                   # 测试数据
    ├── sample_economic_data.csv
    └── sample_financial_data.json
```

## 🧪 测试类型说明

### 1. 单元测试 (Unit Tests)
- **位置**: `tests/test_unit/`
- **目的**: 测试单个函数或类的功能
- **特点**: 快速、隔离、无外部依赖
- **覆盖范围**: 所有核心工具函数

### 2. 集成测试 (Integration Tests)
- **位置**: `tests/test_integration/`
- **目的**: 测试多个模块的协同工作
- **特点**: 验证数据流和接口兼容性
- **覆盖范围**: 工作流、跨模块交互

### 3. 文件输入测试 (File Input Tests)
- **位置**: `tests/test_integration/test_file_input.py`
- **目的**: 测试CSV/JSON文件输入功能
- **特点**: 验证文件解析和数据转换
- **覆盖范围**: 所有支持的文件格式

## 🚀 运行测试

### 方式1: 使用测试运行脚本（推荐）

```bash
# 运行所有测试
python tests/run_tests.py

# 只运行单元测试
python tests/run_tests.py --unit

# 只运行集成测试
python tests/run_tests.py --integration

# 只运行文件输入测试
python tests/run_tests.py --file-input

# 详细输出
python tests/run_tests.py --verbose
```

### 方式2: 使用pytest直接运行

```bash
# 运行所有测试
pytest

# 运行特定目录的测试
pytest tests/test_unit/
pytest tests/test_integration/

# 运行特定文件
pytest tests/test_unit/test_statistics.py

# 运行标记的测试
pytest -m "unit"
pytest -m "integration"
pytest -m "file_input"

# 生成测试报告
pytest --html=report.html --self-contained-html
```

### 方式3: 使用uv运行

```bash
# 运行所有测试
uv run pytest

# 运行特定测试
uv run pytest tests/test_unit/
```

## 📊 测试标记说明

pytest标记用于分类和组织测试：

| 标记 | 说明 | 示例 |
|------|------|------|
| `@pytest.mark.unit` | 单元测试 | `test_descriptive_stats` |
| `@pytest.mark.integration` | 集成测试 | `test_econometric_workflow` |
| `@pytest.mark.file_input` | 文件输入测试 | `test_csv_parsing` |
| `@pytest.mark.slow` | 慢速测试 | `test_large_dataset` |
| `@pytest.mark.performance` | 性能测试 | `test_benchmark_regression` |

## 🔧 测试配置

### pytest配置 (conftest.py)

提供以下测试fixtures：

- `sample_economic_data`: 样本经济数据
- `sample_regression_data`: 样本回归数据
- `sample_time_series_data`: 样本时间序列数据
- `sample_panel_data`: 样本面板数据
- `sample_csv_content`: 样本CSV内容
- `sample_json_content`: 样本JSON内容
- `sample_ml_data`: 样本机器学习数据

### 使用fixtures示例

```python
def test_regression_with_fixture(sample_regression_data):
    """使用fixture的测试示例"""
    y_data = sample_regression_data["y_data"]
    X_data = sample_regression_data["X_data"]
    feature_names = sample_regression_data["feature_names"]
    
    result = perform_ols_regression(y_data, X_data, feature_names)
    assert result.rsquared >= 0
```

## 📈 测试覆盖率

### 覆盖率报告生成

```bash
# 安装覆盖率工具
pip install pytest-cov

# 生成覆盖率报告
pytest --cov=src/aigroup_econ_mcp --cov-report=html
pytest --cov=src/aigroup_econ_mcp --cov-report=term-missing
```

### 覆盖率目标

- **单元测试覆盖率**: > 90%
- **集成测试覆盖率**: > 80%
- **整体覆盖率**: > 85%

## 🐛 故障排除

### 常见问题

#### 1. 导入错误
```bash
# 确保在项目根目录运行测试
cd d:/aigroup-econ-mcp
python tests/run_tests.py
```

#### 2. 依赖缺失
```bash
# 安装测试依赖
pip install pytest pytest-cov pytest-html
```

#### 3. 测试超时
```bash
# 增加超时时间
pytest --timeout=300
```

#### 4. 内存不足
```bash
# 减少测试数据规模
pytest -k "not slow"
```

### 调试测试

```bash
# 进入调试模式
pytest --pdb

# 详细输出
pytest -v

# 显示所有输出
pytest -s
```

## 📝 编写新测试

### 单元测试模板

```python
import pytest
from src.aigroup_econ_mcp.tools.statistics import calculate_descriptive_stats

class TestNewFeature:
    """新功能测试类"""
    
    def test_feature_basic_functionality(self):
        """测试基础功能"""
        # 准备测试数据
        data = [1, 2, 3, 4, 5]
        
        # 执行测试
        result = calculate_descriptive_stats(data)
        
        # 验证结果
        assert result.count == 5
        assert result.mean == 3.0
    
    def test_feature_edge_cases(self):
        """测试边界情况"""
        # 测试空数据
        with pytest.raises(ValueError):
            calculate_descriptive_stats([])
    
    def test_feature_with_fixture(self, sample_economic_data):
        """使用fixture的测试"""
        data = sample_economic_data["gdp_growth"]
        result = calculate_descriptive_stats(data)
        assert result.mean > 0
```

### 集成测试模板

```python
import pytest

@pytest.mark.integration
class TestIntegrationWorkflow:
    """集成工作流测试"""
    
    def test_complete_workflow(self, sample_economic_data):
        """测试完整工作流"""
        # 1. 描述性统计
        stats = calculate_descriptive_stats(sample_economic_data["gdp_growth"])
        
        # 2. 相关性分析
        corr = calculate_correlation_matrix({
            "GDP": sample_economic_data["gdp_growth"],
            "Inflation": sample_economic_data["inflation"]
        })
        
        # 3. 回归分析
        y_data = sample_economic_data["gdp_growth"]
        X_data = [[inf] for inf in sample_economic_data["inflation"]]
        reg_result = perform_ols_regression(y_data, X_data, ["Inflation"])
        
        # 验证工作流完整性
        assert stats.mean > 0
        assert len(corr.correlation_matrix) == 2
        assert reg_result.rsquared >= 0
```

## 🔍 测试最佳实践

### 1. 测试命名
- 使用描述性名称
- 遵循 `test_<功能>_<场景>` 格式
- 包含边界情况和错误处理

### 2. 测试组织
- 每个测试类专注于一个模块
- 使用fixtures避免重复代码
- 标记测试类型便于分类运行

### 3. 测试数据
- 使用可重现的随机数据
- 包含各种数据类型和规模
- 提供边界值测试数据

### 4. 断言设计
- 验证关键业务逻辑
- 检查错误处理和异常
- 确保性能要求

## 📞 支持

如有测试相关问题，请：

1. 查看测试输出和错误信息
2. 检查测试数据格式
3. 验证依赖包版本
4. 提交GitHub Issue

---

**测试是质量保证的关键环节，请确保所有新功能都包含相应的测试！** 🎯