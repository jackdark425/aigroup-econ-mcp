#!/usr/bin/env python3
"""
全面的MCP服务器功能测试
"""

import subprocess
import json
import sys

def send_mcp_request(process, request: dict) -> dict:
    """发送MCP请求并获取响应"""
    request_str = json.dumps(request) + "\n"
    print(f"   发送请求: {request_str.strip()}")
    process.stdin.write(request_str)
    process.stdin.flush()

    # 读取响应
    response_str = process.stdout.readline().strip()
    print(f"   收到响应: {response_str}")

    if not response_str:
        raise ValueError("没有收到响应")

    return json.loads(response_str)

def test_comprehensive_functionality():
    """测试服务器的完整功能"""
    print("🧪 全面测试MCP服务器功能...")

    # 启动服务器进程
    process = subprocess.Popen(
        ["uv", "run", "aigroup-econ-mcp", "main", "--transport", "stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    try:
        import time
        time.sleep(2)  # 等待服务器启动

        tests_passed = 0
        total_tests = 0

        # 测试1: 初始化
        print("\n1️⃣ 测试初始化...")
        total_tests += 1
        init_request = {
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

        init_response = send_mcp_request(process, init_request)
        if init_response.get("result", {}).get("protocolVersion") == "2024-11-05":
            print("   ✅ 初始化成功")
            tests_passed += 1
        else:
            print("   ❌ 初始化失败")

        # 测试2: 工具列表
        print("\n2️⃣ 测试工具列表...")
        total_tests += 1
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }

        tools_response = send_mcp_request(process, tools_request)
        tools = tools_response.get("result", {}).get("tools", [])
        expected_tools = [
            "descriptive_statistics",
            "ols_regression",
            "hypothesis_testing",
            "time_series_analysis",
            "correlation_analysis"
        ]

        found_tools = [tool.get("name") for tool in tools]
        if all(tool in found_tools for tool in expected_tools):
            print(f"   ✅ 找到所有 {len(expected_tools)} 个工具")
            tests_passed += 1
        else:
            print(f"   ❌ 工具不完整，期望: {expected_tools}, 实际: {found_tools}")

        # 测试3: 资源列表
        print("\n3️⃣ 测试资源列表...")
        total_tests += 1
        resources_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
            "params": {}
        }

        resources_response = send_mcp_request(process, resources_request)
        resources = resources_response.get("result", {}).get("resources", [])
        if len(resources) >= 1:
            print(f"   ✅ 找到 {len(resources)} 个资源")
            tests_passed += 1
        else:
            print("   ❌ 没有找到资源")

        # 测试4: 描述性统计
        print("\n4️⃣ 测试描述性统计...")
        total_tests += 1
        stats_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "descriptive_statistics",
                "arguments": {
                    "data": {
                        "Sales": [100, 120, 150, 130, 180],
                        "Advertising": [20, 25, 30, 28, 35]
                    }
                }
            }
        }

        stats_response = send_mcp_request(process, stats_request)
        if stats_response.get("result", {}).get("content"):
            print("   ✅ 描述性统计计算成功")
            tests_passed += 1
        else:
            print("   ❌ 描述性统计计算失败")

        # 测试5: 回归分析
        print("\n5️⃣ 测试回归分析...")
        total_tests += 1
        regression_request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "ols_regression",
                "arguments": {
                    "y_data": [100, 120, 150, 130, 180],
                    "x_data": [[20], [25], [30], [28], [35]],
                    "feature_names": ["Advertising"]
                }
            }
        }

        regression_response = send_mcp_request(process, regression_request)
        if regression_response.get("result", {}).get("content"):
            print("   ✅ 回归分析计算成功")
            tests_passed += 1
        else:
            print("   ❌ 回归分析计算失败")

        print(f"\n🎯 测试结果: {tests_passed}/{total_tests} 通过")

        if tests_passed == total_tests:
            print("🎉 所有测试通过！MCP服务器配置正确且功能完整。")
            return True
        else:
            print("⚠️  部分测试失败，需要进一步检查。")
            return False

    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        return False
    finally:
        process.terminate()
        process.wait()

if __name__ == "__main__":
    success = test_comprehensive_functionality()
    sys.exit(0 if success else 1)