"""
MCP 工具注册中心
自动发现和注册所有工具组件
"""

from typing import Dict, List, Callable, Any
import importlib
import inspect
from pathlib import Path


class ToolGroup:
    """工具组基类"""
    
    # 工具组元数据
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    @classmethod
    def get_tools(cls) -> List[Dict[str, Any]]:
        """
        返回工具组中的所有工具
        
        返回格式:
        [
            {
                "name": "tool_name",
                "handler": async_function,
                "description": "Tool description"
            },
            ...
        ]
        """
        raise NotImplementedError
    
    @classmethod
    def get_help_text(cls) -> str:
        """返回工具组的帮助文档"""
        return f"{cls.name} - {cls.description}"


class ToolRegistry:
    """工具注册中心"""
    
    def __init__(self):
        self.tool_groups: List[ToolGroup] = []
        self.tools: Dict[str, Dict[str, Any]] = {}
    
    def register_group(self, group: ToolGroup):
        """注册工具组"""
        self.tool_groups.append(group)
        
        # 注册工具组中的所有工具
        for tool in group.get_tools():
            self.tools[tool["name"]] = {
                "handler": tool["handler"],
                "description": tool.get("description", ""),
                "group": group.name
            }
    
    def auto_discover_groups(self, base_path: str = None):
        """自动发现并注册所有工具组"""
        if base_path is None:
            base_path = Path(__file__).parent
        
        # 扫描 mcp_tool_groups 目录
        groups_dir = Path(base_path) / "mcp_tool_groups"
        if not groups_dir.exists():
            print(f"工具组目录不存在: {groups_dir}")
            return
        
        print(f"扫描工具组目录: {groups_dir}")
        
        # 导入所有工具组模块
        for module_file in groups_dir.glob("*_tools.py"):
            module_name = module_file.stem
            print(f"发现工具组模块: {module_name}")
            try:
                module = importlib.import_module(f"tools.mcp_tool_groups.{module_name}")
                
                # 查找工具组类
                found_groups = []
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                        issubclass(obj, ToolGroup) and
                        obj != ToolGroup and
                        hasattr(obj, 'get_tools')):
                        
                        found_groups.append(name)
                        # 实例化并注册
                        group_instance = obj()
                        self.register_group(group_instance)
                        print(f"  注册工具组: {name}")
                
                if not found_groups:
                    print(f"  在模块 {module_name} 中未找到工具组类")
                        
            except Exception as e:
                print(f"加载工具组 {module_name} 失败: {e}")
    
    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """获取所有已注册的工具"""
        return self.tools
    
    def get_tool_names(self) -> List[str]:
        """获取所有工具名称列表"""
        return list(self.tools.keys())
    
    def get_help_text(self) -> str:
        """生成完整的帮助文档"""
        help_lines = ["Econometrics Tools Guide (Component-Based Architecture):\n"]
        
        for group in self.tool_groups:
            help_lines.append(f"\n{group.name}")
            help_lines.append("=" * len(group.name))
            help_lines.append(group.get_help_text())
            help_lines.append("")
        
        return "\n".join(help_lines)


# 全局注册中心实例
registry = ToolRegistry()