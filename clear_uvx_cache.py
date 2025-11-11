#!/usr/bin/env python3
"""
清除 uvx 缓存脚本
解决 uvx 总是使用旧版本的问题
"""

import os
import sys
import shutil
import platform
from pathlib import Path


def get_cache_path():
    """获取 uv 缓存路径"""
    system = platform.system()
    
    if system == "Windows":
        # Windows: %LOCALAPPDATA%\uv\cache\wheels
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "uv" / "cache" / "wheels"
    elif system in ["Darwin", "Linux"]:
        # macOS/Linux: ~/.cache/uv/wheels
        home = Path.home()
        return home / ".cache" / "uv" / "wheels"
    
    return None


def clear_cache():
    """清除 uvx 缓存"""
    cache_path = get_cache_path()
    
    if not cache_path:
        print(f"❌ 无法确定缓存路径（系统: {platform.system()}）")
        return False
    
    print(f"🔍 缓存路径: {cache_path}")
    
    if not cache_path.exists():
        print("✅ 缓存目录不存在，无需清除")
        return True
    
    try:
        print(f"🗑️  正在删除缓存目录...")
        shutil.rmtree(cache_path)
        print("✅ 缓存清除成功！")
        print("\n💡 现在可以运行: uvx aigroup-econ-mcp")
        return True
    except PermissionError:
        print("❌ 权限不足，请使用管理员权限运行此脚本")
        return False
    except Exception as e:
        print(f"❌ 清除缓存失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("🧹 uvx 缓存清除工具")
    print("   解决 aigroup-econ-mcp 总是运行旧版本的问题")
    print("=" * 60)
    print()
    
    # 显示系统信息
    print(f"系统: {platform.system()}")
    print(f"Python: {sys.version.split()[0]}")
    print()
    
    # 清除缓存
    success = clear_cache()
    
    print()
    print("=" * 60)
    
    if success:
        print("📝 后续步骤:")
        print("   1. 运行: uvx aigroup-econ-mcp")
        print("   2. 或者: uvx --no-cache aigroup-econ-mcp")
        print("   3. 验证版本: uvx aigroup-econ-mcp --version")
    else:
        print("💡 替代方案:")
        print("   使用 --no-cache 参数: uvx --no-cache aigroup-econ-mcp")
    
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())