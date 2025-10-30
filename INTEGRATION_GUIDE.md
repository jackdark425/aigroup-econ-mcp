# 文件输入功能集成指南

## 📊 集成进度总览

### ✅ 已完成 (5/20)
1. ✅ descriptive_statistics - 描述性统计
2. ✅ ols_regression - OLS回归  
3. ✅ hypothesis_testing - 假设检验
4. ✅ time_series_analysis - 时间序列分析
5. ✅ correlation_analysis - 相关性分析（待添加处理逻辑）

### 🔄 待集成 (15/20)

**时间序列类** (5个):
- [ ] var_model_analysis
- [ ] vecm_model_analysis
- [ ] garch_model_analysis
- [ ] state_space_model_analysis
- [ ] variance_decomposition_analysis

**面板数据类** (4个):
- [ ] panel_fixed_effects
- [ ] panel_random_effects
- [ ] panel_hausman_test
- [ ] panel_unit_root_test

**机器学习类** (6个):
- [ ] random_forest_regression_analysis
- [ ] gradient_boosting_regression_analysis
- [ ] lasso_regression_analysis
- [ ] ridge_regression_analysis
- [ ] cross_validation_analysis
- [ ] feature_importance_analysis_tool

## 🛠️ 标准集成流程（每个工具约3-5分钟）

### 步骤1: 添加文件输入参数

在工具函数定义中添加两个新参数：

```python
@mcp.tool()
async def tool_name(
    ctx: Context[ServerSession, AppContext],
    # 原有参数改为Optional
    existing_param: Optional[...] = None,
    # ... 其他参数
    
    # 👇 添加这两个参数（复制粘贴即可）
    file_content: Annotated[
        Optional[str],
        Field(
            default=None,
            description="""CSV或JSON文件内容

📁 支持格式：
- CSV: [根据工具调整描述]
- JSON: [根据工具调整描述]

💡 使用方式：
- 提供文件内容字符串
- 系统会自动解析并识别变量"""
        )
    ] = None,
    file_format: Annotated[
        str,
        Field(
            default="auto",
            description="""文件格式：auto/csv/json"""
        )
    ] = "auto"
) -> CallToolResult:
```

### 步骤2: 添加文件处理逻辑

在函数体开始处添加文件输入处理：

```python
    """工具文档字符串..."""
    try:
        # 👇 添加这段代码（根据tool_type调整）
        params = await process_file_for_tool(
            ctx=ctx,
            file_content=file_content,
            file_format=file_format,
            tool_type='...', # 见下方tool_type对照表
            现有参数1=现有参数1,
            现有参数2=现有参数2
            # ... 列出所有数据相关的参数
        )
        
        # 👇 提取处理后的参数
        现有参数1 = params.get('现有参数1')
        现有参数2 = params.get('现有参数2')
        
        # 👇 验证必需参数
        if 现有参数1 is None:
            raise ValueError("必须提供数据或file_content参数")
        
        # 原有的业务逻辑保持不变...
```

### 步骤3: tool_type对照表

| 工具类型 | tool_type值 | 输出格式 |
|---------|------------|---------|
| 单变量工具 | `single_var` | `data: List[float]` |
| 多变量字典 | `multi_var_dict` | `data: Dict[str, List[float]]` |
| 回归分析 | `regression` | `y_data, x_data, feature_names` |
| 面板数据 | `panel` | `y_data, x_data, entity_ids, time_periods` |

## 📋 具体工具集成模板

### 模板A: 单变量工具 (如garch_model_analysis)

```python
@mcp.tool()
async def garch_model_analysis(
    ctx: Context[ServerSession, AppContext],
    data: Optional[List[float]] = None,  # 改为Optional
    order: Tuple[int, int] = (1, 1),
    dist: str = "normal",
    # 👇 新增
    file_content: Optional[str] = None,
    file_format: str = "auto"
) -> CallToolResult:
    """..."""
    try:
        # 👇 新增
        params = await process_file_for_tool(
            ctx=ctx,
            file_content=file_content,
            file_format=file_format,
            tool_type='single_var',
            data=data
        )
        data = params.get('data')
        
        if data is None:
            raise ValueError("必须提供data或file_content")
        
        # 原有逻辑保持不变
        await ctx.info(f"开始GARCH模型分析...")
        result = garch_model(data, order=order, dist=dist)
        ...
```

### 模板B: 多变量字典工具 (如var_model_analysis)

```python
@mcp.tool()
async def var_model_analysis(
    ctx: Context[ServerSession, AppContext],
    data: Optional[Dict[str, List[float]]] = None,  # 改为Optional
    max_lags: int = 5,
    ic: str = "aic",
    # 👇 新增
    file_content: Optional[str] = None,
    file_format: str = "auto"
) -> CallToolResult:
    """..."""
    try:
        # 👇 新增
        params = await process_file_for_tool(
            ctx=ctx,
            file_content=file_content,
            file_format=file_format,
            tool_type='multi_var_dict',
            data=data
        )
        data = params.get('data')
        
        if data is None:
            raise ValueError("必须提供data或file_content")
        
        # 原有逻辑保持不变
        await ctx.info(f"开始VAR模型分析...")
        result = var_model(data, max_lags=max_lags, ic=ic)
        ...
```

### 模板C: 回归类工具 (如random_forest_regression)

```python
@mcp.tool()
async def random_forest_regression_analysis(
    ctx: Context[ServerSession, AppContext],
    y_data: Optional[List[float]] = None,  # 改为Optional
    x_data: Optional[List[List[float]]] = None,  # 改为Optional
    feature_names: Optional[List[str]] = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    # 👇 新增
    file_content: Optional[str] = None,
    file_format: str = "auto"
) -> CallToolResult:
    """..."""
    try:
        # 👇 新增
        params = await process_file_for_tool(
            ctx=ctx,
            file_content=file_content,
            file_format=file_format,
            tool_type='regression',
            y_data=y_data,
            x_data=x_data,
            feature_names=feature_names
        )
        y_data = params.get('y_data')
        x_data = params.get('x_data')
        feature_names = params.get('feature_names')
        
        if y_data is None or x_data is None:
            raise ValueError("必须提供数据或file_content")
        
        # 原有逻辑保持不变
        await ctx.info(f"开始随机森林回归...")
        result = random_forest_regression(
            y_data, x_data, feature_names, n_estimators, max_depth
        )
        ...
```

### 模板D: 面板数据工具 (如panel_fixed_effects)

```python
@mcp.tool()
async def panel_fixed_effects(
    ctx: Context[ServerSession, AppContext],
    y_data: Optional[List[float]] = None,  # 改为Optional
    x_data: Optional[List[List[float]]] = None,  # 改为Optional
    entity_ids: Optional[List[str]] = None,  # 改为Optional
    time_periods: Optional[List[str]] = None,  # 改为Optional
    feature_names: Optional[List[str]] = None,
    entity_effects: bool = True,
    time_effects: bool = False,
    # 👇 新增
    file_content: Optional[str] = None,
    file_format: str = "auto"
) -> CallToolResult:
    """..."""
    try:
        # 👇 新增
        params = await process_file_for_tool(
            ctx=ctx,
            file_content=file_content,
            file_format=file_format,
            tool_type='panel',
            y_data=y_data,
            x_data=x_data,
            entity_ids=entity_ids,
            time_periods=time_periods,
            feature_names=feature_names
        )
        y_data = params.get('y_data')
        x_data = params.get('x_data')
        entity_ids = params.get('entity_ids')
        time_periods = params.get('time_periods')
        feature_names = params.get('feature_names')
        
        if y_data is None or x_data is None:
            raise ValueError("必须提供数据或file_content")
        
        # 原有逻辑保持不变
        await ctx.info(f"开始固定效应模型分析...")
        result = fixed_effects_model(
            y_data, x_data, entity_ids, time_periods, 
            feature_names, entity_effects, time_effects
        )
        ...
```

## ✅ 集成检查清单

完成每个工具后，检查以下项目：

- [ ] 所有数据参数已改为Optional
- [ ] 添加了file_content和file_format参数
- [ ] 参数描述中注明"如果提供file_content，此参数可为空"
- [ ] 在try块开始处添加了process_file_for_tool调用
- [ ] tool_type选择正确
- [ ] 所有数据参数都传入了process_file_for_tool
- [ ] 提取了所有需要的参数
- [ ] 添加了数据验证
- [ ] 原有逻辑保持不变

## 🧪 测试每个工具

集成完成后，可以使用以下代码测试：

```python
# 测试CSV输入
csv_content = """
var1,var2,var3
1.0,2.0,3.0
4.0,5.0,6.0
"""

result = await tool_name(
    ctx=ctx,
    file_content=csv_content,
    file_format="csv"
)

# 测试JSON输入
json_content = {
    "var1": [1.0, 4.0],
    "var2": [2.0, 5.0],
    "var3": [3.0, 6.0]
}

result = await tool_name(
    ctx=ctx,
    file_content=json.dumps(json_content),
    file_format="json"
)

# 测试向后兼容（原有方式）
result = await tool_name(
    ctx=ctx,
    data=[1.0, 2.0, 3.0]  # 原有参数
)
```

## 💡 常见问题

### Q1: 如何确定tool_type?
**A**: 查看工具的数据输入结构：
- 单个数值列表 → `single_var`
- 字典{变量名: [数据]} → `multi_var_dict`  
- 有y_data和x_data → `regression`
- 有entity_ids和time_periods → `panel`

### Q2: 参数顺序重要吗?
**A**: file_content和file_format应该放在最后，其他参数保持原有顺序。

### Q3: 需要修改原有业务逻辑吗?
**A**: 不需要！只需在开头添加文件处理，其余代码保持不变。

### Q4: 如何处理特殊的参数结构?
**A**: 参考已完成的类似工具，或查看file_parser.py中的convert_to_tool_format方法。

## 📝 批量集成建议

1. **按类型分组**: 先完成所有单变量工具，再做多变量，最后是回归和面板
2. **使用代码片段**: 创建VSCode snippets加快输入
3. **一次提交一类**: 完成一类工具后提交代码，便于回滚
4. **及时测试**: 每完成3-5个工具就测试一次

## 🎯 优先级建议

**高优先级** (常用工具):
1. correlation_analysis - 相关性分析
2. var_model_analysis - VAR模型
3. random_forest_regression_analysis - 随机森林
4. panel_fixed_effects - 固定效应

**中优先级**:
5-10. 其他时间序列和机器学习工具

**低优先级**:
11-15. 剩余工具

## 📞 需要帮助?

- 查看已完成工具的代码作为参考
- 运行 `examples/file_input_usage.py` 了解使用方法
- 查看 `COMPONENT_DESIGN.md` 了解架构设计

---

**预计总时间**: 15个工具 × 5分钟 = 75分钟
**建议分批完成**: 每次5个工具，分3批完成