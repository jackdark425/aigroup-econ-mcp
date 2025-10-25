#!/usr/bin/env python3
"""
快速测试修复后的MCP服务器
"""

import subprocess
import json
import time
import sys

def test_server():
    """测试服务器功能"""
    print("🧪 快速测试MCP服务器...")
    
    # 启动服务器进程
    process = subprocess.Popen(
        ["uv", "run", "aigroup-econ-mcp", "main", "--transport", "stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    try:
        # 等待服务器启动
        time.sleep(2)
        
        # 发送初始化请求
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
        
        print("📤 发送初始化请求...")
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        # 读取响应
        time.sleep(1)
        response_line = process.stdout.readline()
        
        if response_line.strip():
            try:
                response = json.loads(response_line.strip())
                if response.get("id") == 1:
                    print("✅ 服务器初始化成功!")
                    print(f"   服务器名称: {response.get('result', {}).get('serverInfo', {}).get('name', 'Unknown')}")
                    return True
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析错误: {e}")
                print(f"   响应内容: {response_line}")
        else:
            print("❌ 没有收到响应")
            
        return False
        
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        return False
    finally:
        process.terminate()
        process.wait()

if __name__ == "__main__":
    success = test_server()
    sys.exit(0 if success else 1)