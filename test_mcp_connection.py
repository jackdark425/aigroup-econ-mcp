#!/usr/bin/env python3
"""测试MCP连接是否正常"""
import subprocess
import json
import sys
import time

def test_mcp_stdio():
    """测试stdio模式的MCP通信"""
    print("正在启动MCP服务器...")
    
    # 启动MCP服务器进程
    process = subprocess.Popen(
        ['uvx', '--from', '.', 'aigroup-econ-mcp'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0
    )
    
    try:
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
        
        print("发送初始化请求...")
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        # 等待响应
        print("等待响应...")
        time.sleep(2)
        
        # 读取响应
        response_line = process.stdout.readline()
        if response_line:
            print("✓ 收到响应:")
            print(response_line)
            response = json.loads(response_line)
            if "result" in response:
                print("✓ MCP服务器初始化成功!")
                return True
        else:
            print("✗ 未收到响应")
            # 检查stderr
            error = process.stderr.read()
            if error:
                print(f"错误输出: {error}")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        stderr = process.stderr.read()
        if stderr:
            print(f"错误输出: {stderr}")
        return False
    finally:
        process.terminate()
        process.wait(timeout=5)

if __name__ == "__main__":
    success = test_mcp_stdio()
    sys.exit(0 if success else 1)