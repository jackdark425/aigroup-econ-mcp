#!/usr/bin/env python3
"""修改server.py中的类型定义"""

import re

# 读取文件
with open('src/aigroup_econ_mcp/server.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 替换descriptive_statistics的参数类型
content = re.sub(
    r'(async def descriptive_statistics\([^)]+data: Annotated\[\s+)Dict\[str, List\[float\]\]',
    r'\1Union[Dict[str, List[float]], str]',
    content,
    flags=re.DOTALL
)

# 替换correlation_analysis的参数类型  
content = re.sub(
    r'(async def correlation_analysis\([^)]+data: Annotated\[\s+)Dict\[str, List\[float\]\]',
    r'\1Union[Dict[str, List[float]], str]',
    content,
    flags=re.DOTALL
)

# 写回文件
with open('src/aigroup_econ_mcp/server.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 类型定义修改完成！")