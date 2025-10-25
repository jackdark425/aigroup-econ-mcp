#!/usr/bin/env python3
"""
测试aigroup-econ-mcp服务器功能
"""

import json
import subprocess
import sys
from typing import Dict, Any

def send_mcp_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """发送MCP请求并获取响应"""
    print(f"   发送请求: {request}")

    process = subprocess.Popen(
        ["uv", "run", "aigroup-econ-mcp", "main", "--transport", "stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # 发送请求
    request_str = json.dumps(request) + "\n"
    process.stdin.write(request_str)
    process.stdin.flush()

    # 读取所有输出
    import time
    time.sleep(1)  # 等待服务器启动

    response_str = process.stdout.read()
    error_str = process.stderr.read()

    process.terminate()

    print(f"   收到响应: {response_str}")
    if error_str:
        print(f"   错误输出: {error_str}")

    if not response_str.strip():
        raise ValueError("没有收到响应")

    return json.loads(response_str)

def test_server_capabilities():
    """测试服务器能力"""
    print("🔍 测试服务器能力...")

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }

    try:
        response = send_mcp_request(request)
        print("✅ 服务器初始化成功")
        print(f"   服务器名称: {response.get('result', {}).get('serverInfo', {}).get('name', 'Unknown')}")
        print(f"   协议版本: {response.get('result', {}).get('protocolVersion', 'Unknown')}")
        return True
    except Exception as e:
        print(f"❌ 服务器初始化失败: {e}")
        return False

def test_list_tools():
    """测试工具列表"""
    print("\n🔧 测试工具列表...")

    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }

    try:
        response = send_mcp_request(request)
        tools = response.get("result", {}).get("tools", [])
        print(f"✅ 找到 {len(tools)} 个工具:")
        for tool in tools:
            print(f"   - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")
        return True
    except Exception as e:
        print(f"❌ 获取工具列表失败: {e}")
        return False

def test_list_resources():
    """测试资源列表"""
    print("\n📁 测试资源列表...")

    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "resources/list",
        "params": {}
    }

    try:
        response = send_mcp_request(request)
        resources = response.get("result", {}).get("resources", [])
        print(f"✅ 找到 {len(resources)} 个资源:")
        for resource in resources:
            print(f"   - {resource.get('uri', 'Unknown')}: {resource.get('name', 'No name')}")
        return True
    except Exception as e:
        print(f"❌ 获取资源列表失败: {e}")
        return False

def test_descriptive_stats():
    """测试描述性统计功能"""
    print("\n📊 测试描述性统计...")

    request = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "descriptive_statistics",
            "arguments": {
                "data": {
                    "变量A": [1, 2, 3, 4, 5],
                    "变量B": [2, 4, 6, 8, 10]
                }
            }
        }
    }

    try:
        response = send_mcp_request(request)
        result = response.get("result", {})
        content = result.get("content", [])
        if content:
            print("✅ 描述性统计计算成功:")
            print(f"   结果: {content[0].get('text', 'No result')[:100]}...")
        return True
    except Exception as e:
        print(f"❌ 描述性统计测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 开始测试aigroup-econ-mcp服务器...")
    print("=" * 50)

    tests = [
        test_server_capabilities,
        test_list_tools,
        test_list_resources,
        test_descriptive_stats
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"🧪 测试完成: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！服务器配置正确。")
        return 0
    else:
        print("⚠️  部分测试失败，需要检查配置。")
        return 1

if __name__ == "__main__":
    sys.exit(main())