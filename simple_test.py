#!/usr/bin/env python3
"""
简单的MCP服务器测试
"""

import subprocess
import json
import sys

def test_basic_communication():
    """测试基本的MCP通信"""
    print("测试MCP服务器基本通信...")

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
        # 等待服务器启动
        import time
        time.sleep(2)

        # 发送initialize请求
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

        request_str = json.dumps(init_request) + "\n"
        print(f"发送: {request_str.strip()}")
        process.stdin.write(request_str)
        process.stdin.flush()

        # 读取响应
        response_lines = []
        for i in range(10):  # 尝试读取10行
            line = process.stdout.readline()
            if line.strip():
                response_lines.append(line.strip())
                print(f"收到: {line.strip()}")

                # 如果收到完整的响应，解析它
                try:
                    response = json.loads(line.strip())
                    if response.get("id") == 1:
                        print("✅ 服务器响应成功!")
                        return True
                except json.JSONDecodeError:
                    continue

        print("❌ 没有收到有效的响应")
        return False

    except Exception as e:
        print(f"❌ 测试出错: {e}")
        return False
    finally:
        process.terminate()
        process.wait()

if __name__ == "__main__":
    success = test_basic_communication()
    sys.exit(0 if success else 1)