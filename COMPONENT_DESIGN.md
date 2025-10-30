# 文件输入功能 - 组件化设计文档

## 🎯 设计理念

采用**组件化架构**，将文件输入功能设计为可复用的组件，而不是在每个工具中重复代码。这样可以：

1. ✅ **统一接口**: 所有工具使用相同的文件输入处理方式
2. ✅ **易于维护**: 修改一处，所有工具自动更新
3. ✅ **可扩展**: 轻松添加新的文件格式支持
4. ✅ **解耦合**: 文件处理逻辑与业务逻辑分离

## 📦 组件架构

```
文件输入功能
├── FileParser (文件解析器)
│   ├── parse_file_content()  # 解析文件
│   ├── convert_to_tool_format()  # 格式转换
│   └── auto_detect_tool_params()  # 智能推荐
│
├── FileInputHandler (输入处理器)
│   ├── process_input()  # 处理输入
│   └── with_file_support()  # 装饰器
│
└── UnifiedFileInput (统一接口)
    └── handle()  # 统一处理入口
```

## 🔧 核心组件

### 1. FileParser - 文件解析器

**职责**: 解析CSV/JSON文件并转换为标准格式

**位置**: `src/aigroup_econ_mcp/tools/file_parser.py`

**主要方法**:
```python
class FileParser:
    @staticmethod
    def parse_file_content(content: str, file_format: str) -> Dict[str, Any]:
        """解析文件内容，返回标准化数据"""
        
    @staticmethod
    def convert_to_tool_format(parsed_data: Dict, tool_type: str) -> Dict:
        """转换为特定工具所需格式"""
        
    @staticmethod
    def auto_detect_tool_params(parsed_data: Dict) -> Dict:
        """智能推荐适合的工具"""
```

**支持的工具类型**:
- `single_var`: 单变量数据 → `List[float]`
- `multi_var_dict`: 多变量字典 → `Dict[str, List[float]]`
- `multi_var_matrix`: 多变量矩阵 → `List[List[float]]`
- `regression`: 回归分析 → `y_data, x_data, feature_names`
- `panel`: 面板数据 → `y_data, x_data, entity_ids, time_periods`

### 2. UnifiedFileInput - 统一接口组件

**职责**: 为所有工具提供统一的文件输入处理

**位置**: `src/aigroup_econ_mcp/tools/file_input_handler.py`

**核心方法**:
```python
class UnifiedFileInput:
    @staticmethod
    async def handle(
        ctx: Any,
        file_content: Optional[str],
        file_format: str,
        tool_type: str,
        original_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        统一处理文件输入
        - 自动记录日志
        - 解析文件
        - 转换格式
        - 合并参数
        """
```

### 3. process_file_for_tool - 便捷函数

**职责**: 提供简洁的API供工具使用

```python
async def process_file_for_tool(
    ctx: Any,
    file_content: Optional[str],
    file_format: str,
    tool_type: str,
    **kwargs
) -> Dict[str, Any]:
    """
    一行代码处理文件输入
    
    使用示例:
        params = await process_file_for_tool(
            ctx=ctx,
            file_content=file_content,
            file_format=file_format,
            tool_type='regression',
            y_data=y_data,
            x_data=x_data
        )
    """
```

## 🔨 使用方式

### 方式1: 使用便捷函数 (推荐)

在工具函数中添加文件输入支持只需3步：

**步骤1**: 添加参数
```python
async def my_tool(
    ctx: Context,
    data: Optional[List[float]] = None,  # 改为可选
    file_content: Optional[str] = None,  # 新增
    file_format: str = "auto"  # 新增
) -> CallToolResult:
```

**步骤2**: 处理文件输入
```python
    params = await process_file_for_tool(
        ctx=ctx,
        file_content=file_content,
        file_format=file_format,
        tool_type='single_var',  # 根据工具选择类型
        data=data
    )
    data = params.get('data')
```

**步骤3**: 验证并执行
```python
    if data is None:
        raise ValueError("必须提供data或file_content")
    
    # 执行原有逻辑
    result = compute_something(data)
```

### 方式2: 使用装饰器

```python
from .tools.file_input_handler import FileInputHandler

@FileInputHandler.with_file_support('regression')
async def my_regression_tool(
    ctx: Context,
    y_data: Optional[List[float]] = None,
    x_data: Optional[List[List[float]]] = None,
    file_content: Optional[str] = None,
    file_format: str = "auto"
):
    # 装饰器会自动处理file_content并填充y_data和x_data
    # 直接使用即可
    result = compute_regression(y_data, x_data)
```

### 方式3: 使用混入类

```python
from .tools.file_input_handler import FileInputMixin

class MyAnalysisTool(FileInputMixin):
    async def analyze(self, file_content: str):
        # 解析文件
        parsed = self.parse_file_input(file_content)
        
        # 转换格式
        data = self.convert_for_tool(parsed, 'regression')
        
        # 获取推荐
        recommendations = self.get_recommendations(parsed)
```

## 📋 实际示例

### 示例1: 描述性统计 (已完成)

```python
@mcp.tool()
async def descriptive_statistics(
    ctx: Context[ServerSession, AppContext],
    data: Optional[Dict[str, List[float]]] = None,
    file_content: Optional[str] = None,
    file_format: str = "auto"
) -> CallToolResult:
    try:
        # 使用组件处理文件输入
        params = await process_file_for_tool(
            ctx=ctx,
            file_content=file_content,
            file_format=file_format,
            tool_type='multi_var_dict',
            data=data
        )
        data = params.get('data')
        
        if data is None:
            raise ValueError("必须提供data或file_content参数")
        
        # 执行计算
        result = _compute_descriptive_stats(data)
        
        # 返回结果
        return CallToolResult(...)
```

### 示例2: OLS回归 (已完成)

```python
@mcp.tool()
async def ols_regression(
    ctx: Context[ServerSession, AppContext],
    y_data: Optional[List[float]] = None,
    x_data: Optional[List[List[float]]] = None,
    feature_names: Optional[List[str]] = None,
    file_content: Optional[str] = None,
    file_format: str = "auto"
) -> CallToolResult:
    try:
        # 使用组件处理文件输入
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
        
        # 执行回归
        result = _compute_ols_regression(y_data, x_data, feature_names)
        
        return CallToolResult(...)
```

## 🎨 组件优势

### 1. 代码复用
```python
# ❌ 不使用组件 - 每个工具都要重复
async def tool1(...):
    if file_content:
        parsed = FileParser.parse_file_content(...)
        # 10行代码...

async def tool2(...):
    if file_content:
        parsed = FileParser.parse_file_content(...)
        # 10行代码...（重复！）

# ✅ 使用组件 - 一行搞定
async def tool1(...):
    params = await process_file_for_tool(...)

async def tool2(...):
    params = await process_file_for_tool(...)
```

### 2. 统一日志
组件自动提供详细日志：
```
✓ 检测到文件输入，开始解析...
✓ 文件解析成功：3个变量，100个观测，数据类型=multivariate
✓ 数据已转换：因变量=sales，自变量=['advertising', 'price']
```

### 3. 错误处理
统一的错误处理和用户友好的错误消息：
```python
try:
    params = await process_file_for_tool(...)
except ValueError as e:
    # 自动记录错误日志
    # 返回友好的错误消息
```

### 4. 易于扩展
添加新文件格式只需修改FileParser：
```python
# 在FileParser中添加
@staticmethod
def _parse_excel(content: str) -> Dict:
    # 解析Excel
    pass

# 所有工具自动支持Excel！
```

## 📊 工具集成进度

### ✅ 已完成 (2/20)
1. ✅ **descriptive_statistics** - 使用组件化方法
2. ✅ **ols_regression** - 使用组件化方法

### 🔄 集成模板

对于剩余18个工具，使用以下模板快速集成：

```python
@mcp.tool()
async def tool_name(
    ctx: Context[ServerSession, AppContext],
    # 原有参数改为Optional
    data: Optional[...] = None,
    # 添加文件参数
    file_content: Optional[str] = None,
    file_format: str = "auto"
) -> CallToolResult:
    """工具说明..."""
    try:
        # 一行代码处理文件输入
        params = await process_file_for_tool(
            ctx=ctx,
            file_content=file_content,
            file_format=file_format,
            tool_type='...', # single_var/multi_var_dict/regression/panel
            data=data
            # ... 其他参数
        )
        
        # 提取参数
        data = params.get('data')
        
        # 验证
        if data is None:
            raise ValueError("必须提供数据或file_content")
        
        # 执行原有逻辑（无需修改）
        result = _compute_...(data)
        
        return CallToolResult(...)
```

## 🧪 测试

### 单元测试
```bash
pytest tests/test_file_input.py -v
```

### 集成测试
```python
# 测试文件输入
result = await descriptive_statistics(
    ctx=ctx,
    file_content=csv_content,
    file_format="csv"
)

# 测试传统输入（向后兼容）
result = await descriptive_statistics(
    ctx=ctx,
    data={"var1": [1, 2, 3]}
)
```

## 📚 相关文档

- 📖 [使用示例](examples/file_input_usage.py)
- 📖 [功能说明](FILE_INPUT_FEATURE.md)
- 📖 [测试用例](tests/test_file_input.py)

## 🎯 最佳实践

1. **始终使用process_file_for_tool**: 不要直接调用FileParser
2. **验证参数**: 确保data不为None再使用
3. **保持向后兼容**: 所有原有参数改为Optional
4. **添加有意义的日志**: 让用户知道发生了什么
5. **统一错误消息**: 使用清晰的错误提示

## 🚀 性能考虑

- ✅ 文件解析采用流式处理，内存占用小
- ✅ 使用缓存避免重复解析
- ✅ 异步处理，不阻塞其他操作
- ✅ 智能检测，减少不必要的转换

## 📈 后续规划

- [ ] 添加文件大小限制配置
- [ ] 支持流式文件上传
- [ ] 添加数据预览功能
- [ ] 支持更多文件格式（Excel, Parquet）
- [ ] 提供批量文件处理

---

**设计原则**: 简单、统一、可维护、可扩展