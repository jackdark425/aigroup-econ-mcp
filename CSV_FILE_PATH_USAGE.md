# CSV文件路径输入功能使用指南

## 📋 概述

aigroup-econ-mcp现已支持直接通过CSV文件路径进行数据分析，无需手动复制粘贴数据。

## 🛠️ 已实现的功能

### 1. 数据加载辅助模块
创建了`src/aigroup_econ_mcp/tools/data_loader.py`，提供智能数据加载功能：
- 自动检测输入类型（字典或文件路径）
- 支持相对路径和绝对路径
- 提供详细的错误信息

### 2. 工具参数类型升级
主要分析工具现支持`Union[Dict[str, List[float]], str]`类型参数：
- `descriptive_statistics`: 描述性统计
- `correlation_analysis`: 相关性分析

## 📝 使用方法

### 方式1：传统数据字典（仍然支持）
```json
{
  "data": {
    "GDP增长率": [3.2, 2.8, 3.5, 2.9],
    "通货膨胀率": [2.1, 2.3, 1.9, 2.4],
    "失业率": [4.5, 4.2, 4.0, 4.3]
  }
}
```

### 方式2：CSV文件路径（推荐）
```json
{
  "data": "d:/aigroup-econ-mcp/test_data.csv"
}
```

或使用相对路径：
```json
{
  "data": "./test_data.csv"
}
```

## 📊 CSV文件格式要求

CSV文件格式示例（test_data.csv）：
```csv
GDP增长率,通货膨胀率,失业率
3.2,2.1,4.5
2.8,2.3,4.2
3.5,1.9,4.0
2.9,2.4,4.3
```

要求：
- 第一行必须为变量名（表头）
- 后续行为数值数据
- 所有列必须为数值类型
- 文件编码建议使用UTF-8

## 🔧 手动修改步骤（如需要）

如果工具尚未更新，请按以下步骤手动修改`src/aigroup_econ_mcp/server.py`：

### 步骤1：更新descriptive_statistics函数

找到第155-175行，将参数类型从：
```python
data: Annotated[
    Dict[str, List[float]],
    Field(description="""数据字典...""")
]
```

修改为：
```python
data: Annotated[
    Union[Dict[str, List[float]], str],
    Field(description="""数据输入，支持两种格式：
    
📊 格式1 - 数据字典：
{
    "GDP增长率": [3.2, 2.8, 3.5, 2.9],
    ...
}

📁 格式2 - CSV文件路径：
"d:/aigroup-econ-mcp/test_data.csv"
...""")
]
```

### 步骤2：在函数开始处添加数据加载逻辑

在`descriptive_statistics`函数的开头（约第205行）添加：
```python
async def descriptive_statistics(...) -> CallToolResult:
    """计算描述性统计量..."""
    
    # 智能加载数据
    data = await load_data_if_path(data, ctx)
    
    await ctx.info(f"开始计算描述性统计，处理 {len(data)} 个变量")
    # ... 其余代码保持不变
```

### 步骤3：对correlation_analysis做同样修改

在第728-752行和第823行做相同修改。

## ✅ 测试步骤

1. **创建测试CSV文件**
```bash
cd d:\aigroup-econ-mcp
echo "GDP增长率,通货膨胀率,失业率" > test_data.csv
echo "3.2,2.1,4.5" >> test_data.csv
echo "2.8,2.3,4.2" >> test_data.csv
echo "3.5,1.9,4.0" >> test_data.csv
echo "2.9,2.4,4.3" >> test_data.csv
```

2. **重新安装包**
```bash
cd d:\aigroup-econ-mcp
pip install -e .
```

3. **重启VSCode或Roo/Cline扩展**

4. **测试descriptive_statistics工具**
输入：
```json
{
  "data": "d:/aigroup-econ-mcp/test_data.csv"
}
```

## 🎯 预期结果

工具应该：
1. 自动检测输入为文件路径
2. 加载CSV文件
3. 显示"📁 正在从CSV文件加载数据"
4. 显示"✅ CSV文件加载成功：3个变量，4个观测"
5. 正常执行分析并返回结果

## 🚨 常见问题

### Q: 报错"文件不存在"
A: 检查文件路径是否正确，使用绝对路径更可靠

### Q: 报错"CSV文件读取失败"
A: 检查CSV文件格式，确保：
   - 所有列都是数值
   - 没有缺失值
   - 编码为UTF-8

### Q: 仍然需要手动输入数据字典
A: 确保已重新安装包并重启VSCode

## 📚 相关文件

- 数据加载器：`src/aigroup_econ_mcp/tools/data_loader.py`
- 主服务器：`src/aigroup_econ_mcp/server.py`
- 测试数据：`test_data.csv`