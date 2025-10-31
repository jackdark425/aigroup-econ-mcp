"""
批量为所有工具添加CSV路径支持的脚本
"""
import re

def update_server_file():
    with open('src/aigroup_econ_mcp/server.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 更新导入语句
    content = content.replace(
        'from .tools.data_loader import load_data_if_path',
        'from .tools.data_loader import load_data_if_path, load_single_var_if_path'
    )
    
    # 2. 修改time_series_analysis的参数类型和数据加载
    content = re.sub(
        r'(async def time_series_analysis\(\s+ctx:.*?\n\s+data: Annotated\[\s+)(List\[float\])',
        r'\1Union[List[float], str]',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'(async def time_series_analysis.*?""")\s+(await ctx\.info\(f"开始时间序列分析)',
        r'\1\n    # 智能加载数据：如果是文件路径则加载CSV\n    data = await load_single_var_if_path(data, ctx)\n    \n    \2',
        content,
        flags=re.DOTALL
    )
    
    # 3. 修改hypothesis_testing的data1参数
    content = re.sub(
        r'(async def hypothesis_testing\(\s+ctx:.*?\n\s+data1: Annotated\[\s+)(List\[float\])',
        r'\1Union[List[float], str]',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'(async def hypothesis_testing.*?""")\s+(await ctx\.info\(f"开始假设检验)',
        r'\1\n    # 智能加载数据：如果是文件路径则加载CSV\n    data1 = await load_single_var_if_path(data1, ctx)\n    if data2 is not None:\n        data2 = await load_single_var_if_path(data2, ctx)\n    \n    \2',
        content,
        flags=re.DOTALL
    )
    
    # 4. 修改garch_model_analysis的参数类型
    content = re.sub(
        r'(async def garch_model_analysis\(\s+ctx:.*?\n\s+data: Annotated\[\s+)(List\[float\])',
        r'\1Union[List[float], str]',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'(async def garch_model_analysis.*?""")\s+(await ctx\.info\(f"开始GARCH)',
        r'\1\n    # 智能加载数据：如果是文件路径则加载CSV\n    data = await load_single_var_if_path(data, ctx)\n    \n    \2',
        content,
        flags=re.DOTALL
    )
    
    # 5. 修改var_model_analysis的参数类型
    content = re.sub(
        r'(async def var_model_analysis\(\s+ctx:.*?\n\s+data: Annotated\[\s+)(Dict\[str, List\[float\]\])',
        r'\1Union[Dict[str, List[float]], str]',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'(async def var_model_analysis.*?""")\s+(await ctx\.info\(f"开始VAR)',
        r'\1\n    # 智能加载数据：如果是文件路径则加载CSV\n    data = await load_data_if_path(data, ctx)\n    \n    \2',
        content,
        flags=re.DOTALL
    )
    
    # 6. 修改vecm_model_analysis的参数类型
    content = re.sub(
        r'(async def vecm_model_analysis\(\s+ctx:.*?\n\s+data: Annotated\[\s+)(Dict\[str, List\[float\]\])',
        r'\1Union[Dict[str, List[float]], str]',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'(async def vecm_model_analysis.*?""")\s+(await ctx\.info\(f"开始VECM)',
        r'\1\n    # 智能加载数据：如果是文件路径则加载CSV\n    data = await load_data_if_path(data, ctx)\n    \n    \2',
        content,
        flags=re.DOTALL
    )
    
    # 7. 修改variance_decomposition_analysis的参数类型
    content = re.sub(
        r'(async def variance_decomposition_analysis\(\s+ctx:.*?\n\s+data: Annotated\[\s+)(Dict\[str, List\[float\]\])',
        r'\1Union[Dict[str, List[float]], str]',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'(async def variance_decomposition_analysis.*?""")\s+(await ctx\.info\(f"开始方差分解)',
        r'\1\n    # 智能加载数据：如果是文件路径则加载CSV\n    data = await load_data_if_path(data, ctx)\n    \n    \2',
        content,
        flags=re.DOTALL
    )
    
    # 写回文件
    with open('src/aigroup_econ_mcp/server.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 所有工具的CSV路径支持已添加完成!")
    print("\n修改的工具:")
    print("  1. time_series_analysis - Union[List[float], str]")
    print("  2. hypothesis_testing - data1和data2都支持CSV")
    print("  3. garch_model_analysis - Union[List[float], str]")
    print("  4. var_model_analysis - Union[Dict[str, List[float]], str]")
    print("  5. vecm_model_analysis - Union[Dict[str, List[float]], str]")
    print("  6. variance_decomposition_analysis - Union[Dict[str, List[float]], str]")
    print("\n已添加数据加载逻辑到所有相关函数！")

if __name__ == "__main__":
    update_server_file()