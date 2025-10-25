#!/usr/bin/env python3
"""
最终测试修复后的MCP服务器
"""

import subprocess
import json
import time
import sys

def test_server_connection():
    """测试服务器连接"""
    print("🧪 测试MCP服务器连接...")
    
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
        time.sleep(1)
        
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
                    server_name = response.get('result', {}).get('serverInfo', {}).get('name', 'Unknown')
                    print(f"   服务器名称: {server_name}")
                    
                    # 测试工具列表
                    tools_request = {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/list",
                        "params": {}
                    }
                    
                    print("🔧 测试工具列表...")
                    process.stdin.write(json.dumps(tools_request) + "\n")
                    process.stdin.flush()
                    
                    time.sleep(1)
                    tools_response = process.stdout.readline()
                    
                    if tools_response.strip():
                        tools_data = json.loads(tools_response.strip())
                        tools = tools_data.get("result", {}).get("tools", [])
                        print(f"✅ 找到 {len(tools)} 个工具:")
                        for tool in tools:
                            print(f"   - {tool.get('name', 'Unknown')}")
                        return True
                    else:
                        print("❌ 获取工具列表失败")
                        return False
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
    print("=" * 50)
    success = test_server_connection()
    print("=" * 50)
    
    if success:
        print("🎉 服务器修复成功！现在可以使用 uvx aigroup-econ-mcp 了")
        sys.exit(0)
    else:
        print("⚠️  服务器仍有问题，需要进一步调试")
        sys.exit(1)