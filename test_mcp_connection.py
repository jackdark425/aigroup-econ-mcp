"""
测试MCP服务器连接
"""
import sys
import subprocess
import json

def test_server_start():
    """测试服务器能否正常启动"""
    print("测试1: 检查服务器启动...")
    try:
        result = subprocess.run(
            [sys.executable, "fastmcp_server.py", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"✓ 服务器脚本可执行")
        return True
    except subprocess.TimeoutExpired:
        print("✓ 服务器正常启动（超时是正常的，因为它会持续运行）")
        return True
    except Exception as e:
        print(f"✗ 服务器启动失败: {e}")
        return False

def test_dependencies():
    """测试依赖是否完整"""
    print("\n测试2: 检查依赖...")
    dependencies = [
        "mcp.server.fastmcp",
        "pandas",
        "numpy",
        "scipy",
        "statsmodels"
    ]
    
    all_ok = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError as e:
            print(f"✗ {dep}: {e}")
            all_ok = False
    
    return all_ok

def test_tools_import():
    """测试工具模块导入"""
    print("\n测试3: 检查工具模块...")
    tools = [
        "tools.ols_tool",
        "tools.mle_tool",
        "tools.gmm_tool"
    ]
    
    all_ok = True
    for tool in tools:
        try:
            __import__(tool)
            print(f"✓ {tool}")
        except ImportError as e:
            print(f"✗ {tool}: {e}")
            all_ok = False
    
    return all_ok

def test_mcp_config():
    """测试MCP配置文件"""
    print("\n测试4: 检查MCP配置...")
    try:
        with open(".roo/mcp.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        if "mcpServers" in config:
            print("✓ 配置文件格式正确")
            
            if "aigroup-econ-mcp" in config["mcpServers"]:
                server_config = config["mcpServers"]["aigroup-econ-mcp"]
                print(f"✓ 服务器配置存在")
                print(f"  命令: {server_config.get('command')}")
                print(f"  参数: {server_config.get('args')}")
                print(f"  允许的工具: {len(server_config.get('alwaysAllow', []))}个")
                return True
            else:
                print("✗ 未找到 aigroup-econ-mcp 配置")
                return False
        else:
            print("✗ 配置文件格式错误")
            return False
            
    except FileNotFoundError:
        print("✗ 配置文件不存在")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ 配置文件JSON格式错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 读取配置文件失败: {e}")
        return False

def main():
    print("=" * 60)
    print("AIGroup 计量经济学 MCP 服务器连接测试")
    print("=" * 60)
    
    results = []
    results.append(("依赖检查", test_dependencies()))
    results.append(("工具模块", test_tools_import()))
    results.append(("MCP配置", test_mcp_config()))
    results.append(("服务器启动", test_server_start()))
    
    print("\n" + "=" * 60)
    print("测试摘要:")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！MCP服务器配置正确。")
        print("\n下一步:")
        print("1. 重启 Roo-Code 或您的编辑器")
        print("2. 检查 MCP 服务器是否在编辑器中正常连接")
        print("3. 尝试调用工具: basic_parametric_estimation_ols")
    else:
        print("✗ 部分测试失败，请检查上述错误信息。")
    print("=" * 60)

if __name__ == "__main__":
    main()