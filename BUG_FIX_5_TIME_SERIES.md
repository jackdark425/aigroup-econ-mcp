# 🐛 Bug修复 #5: 缺少time_series类型支持

## 问题
测试`var_model_analysis`工具时发现错误：
```
VAR model fitting failed: Data cannot be empty
```

## 根本原因
`file_parser.py`的`convert_to_tool_format`函数中缺少对`time_series`工具类型的处理。

影响的工具（3个）：
- var_model_analysis
- vecm_model_analysis  
- variance_decomposition_analysis

## 修复方案
在`file_parser.py`第342行添加`time_series`类型处理：

```python
elif tool_type == 'time_series':
    # 时间序列类型，与multi_var_dict相同，返回字典格式
    return {"data": data}
```

## 文件修改
- ✅ `src/aigroup_econ_mcp/tools/file_parser.py` (第339-343行)

## 需要重启
**请重启MCP服务器以加载此修复！**

## Bug修复总结
1. ✅ regression类型 - 移除y_variable参数
2. ✅ panel类型 - 移除y_variable参数  
3. ✅ single_var类型 - 移除data1参数
4. ✅ hypothesis_testing - 参数统一为data
5. ✅ time_series类型 - 添加类型支持