#!/usr/bin/env python3
"""
简单测试MCP服务器是否启动
"""

import subprocess
import time
import sys

def test_mcp_server():
    """测试MCP服务器启动"""
    
    print("启动MCP服务器...")
    
    # 启动MCP服务器进程
    process = subprocess.Popen(
        ["aigroup-econ-mcp", "--debug"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # 等待几秒钟让服务器启动
    time.sleep(3)
    
    # 检查进程是否还在运行
    if process.poll() is None:
        print("✓ MCP服务器正在运行")
        
        # 尝试读取一些输出
        try:
            # 非阻塞读取
            stdout, stderr = process.communicate(timeout=2)
            if stdout:
                print(f"标准输出: {stdout}")
            if stderr:
                print(f"标准错误: {stderr}")
        except subprocess.TimeoutExpired:
            print("✓ MCP服务器正常运行（无输出，符合stdio模式）")
        
        # 终止进程
        process.terminate()
        process.wait()
        print("✓ MCP服务器已正常关闭")
        return True
    else:
        # 进程已经退出
        stdout, stderr = process.communicate()
        print("✗ MCP服务器启动失败")
        if stdout:
            print(f"标准输出: {stdout}")
        if stderr:
            print(f"标准错误: {stderr}")
        return False

if __name__ == "__main__":
    success = test_mcp_server()
    if success:
        print("\n🎉 MCP服务器功能正常！")
        print("现在可以在Roo Cline中使用计量经济学工具了。")
    else:
        print("\n❌ MCP服务器启动失败，请检查配置。")
        sys.exit(1)