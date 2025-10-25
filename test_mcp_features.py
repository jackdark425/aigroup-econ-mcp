#!/usr/bin/env python3
"""全面测试MCP服务器的所有功能"""
import subprocess
import json
import sys
import time

class MCPTester:
    def __init__(self):
        self.process = None
        self.request_id = 0
        
    def start_server(self):
        """启动MCP服务器"""
        print("🚀 启动MCP服务器...")
        self.process = subprocess.Popen(
            ['uvx', '--from', '.', 'aigroup-econ-mcp'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=0
        )
        time.sleep(1)
        
    def send_request(self, method, params=None):
        """发送JSON-RPC请求"""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        
        # 读取响应（可能有多行通知消息）
        max_attempts = 10
        for _ in range(max_attempts):
            response_line = self.process.stdout.readline()
            if response_line:
                response = json.loads(response_line)
                # 如果是通知消息，继续读取下一行
                if "method" in response and response["method"].startswith("notifications/"):
                    continue
                # 如果有id字段，说明这是我们请求的响应
                if "id" in response:
                    return response
        return None
        
    def test_initialize(self):
        """测试初始化"""
        print("\n📋 测试1: 初始化服务器")
        response = self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })
        
        if response and "result" in response:
            print("  ✓ 初始化成功")
            print(f"  服务器名称: {response['result']['serverInfo']['name']}")
            print(f"  服务器版本: {response['result']['serverInfo']['version']}")
            return True
        else:
            print("  ✗ 初始化失败")
            return False
            
    def test_list_tools(self):
        """测试工具列表"""
        print("\n📋 测试2: 获取工具列表")
        response = self.send_request("tools/list")
        
        if response and "result" in response:
            tools = response['result'].get('tools', [])
            print(f"  ✓ 找到 {len(tools)} 个工具:")
            for tool in tools:
                print(f"    - {tool['name']}: {tool.get('description', 'N/A')}")
            return len(tools) > 0
        else:
            print("  ✗ 获取工具列表失败")
            return False
            
    def test_list_resources(self):
        """测试资源列表"""
        print("\n📋 测试3: 获取资源列表")
        response = self.send_request("resources/list")
        
        if response and "result" in response:
            resources = response['result'].get('resources', [])
            print(f"  ✓ 找到 {len(resources)} 个资源:")
            for resource in resources:
                print(f"    - {resource['uri']}: {resource.get('name', 'N/A')}")
            return True
        else:
            print("  ✗ 获取资源列表失败")
            return False
            
    def test_list_prompts(self):
        """测试提示词列表"""
        print("\n📋 测试4: 获取提示词列表")
        response = self.send_request("prompts/list")
        
        if response and "result" in response:
            prompts = response['result'].get('prompts', [])
            print(f"  ✓ 找到 {len(prompts)} 个提示词:")
            for prompt in prompts:
                print(f"    - {prompt['name']}: {prompt.get('description', 'N/A')}")
            return True
        else:
            print("  ✗ 获取提示词列表失败")
            return False
            
    def test_descriptive_statistics(self):
        """测试描述性统计工具"""
        print("\n📋 测试5: 描述性统计工具")
        response = self.send_request("tools/call", {
            "name": "descriptive_statistics",
            "arguments": {
                "data": {
                    "stock_returns": [0.02, -0.01, 0.015, -0.008, 0.012, 0.018, -0.005],
                    "market_returns": [0.018, -0.005, 0.012, -0.006, 0.010, 0.015, -0.003]
                }
            }
        })
        
        print(f"  收到响应: {json.dumps(response, indent=2, ensure_ascii=False)[:500]}...")
        
        if response and "result" in response:
            content = response['result'].get('content', [])
            if content and len(content) > 0:
                print("  ✓ 描述性统计计算成功")
                print(f"  结果预览: {content[0].get('text', '')[:200]}...")
                return True
        
        print("  ✗ 描述性统计计算失败")
        if response and "error" in response:
            print(f"  错误详情: {json.dumps(response['error'], indent=2, ensure_ascii=False)}")
        return False
        
    def test_correlation_analysis(self):
        """测试相关性分析工具"""
        print("\n📋 测试6: 相关性分析工具")
        response = self.send_request("tools/call", {
            "name": "correlation_analysis",
            "arguments": {
                "data": {
                    "GDP_Growth": [3.2, 2.8, 3.5, 2.9, 3.1],
                    "Inflation": [2.1, 2.3, 1.9, 2.4, 2.2],
                    "Unemployment": [4.5, 4.2, 4.0, 4.3, 4.1]
                },
                "method": "pearson"
            }
        })
        
        print(f"  收到响应: {json.dumps(response, indent=2, ensure_ascii=False)[:500]}...")
        
        if response and "result" in response:
            content = response['result'].get('content', [])
            if content and len(content) > 0:
                print("  ✓ 相关性分析成功")
                print(f"  结果预览: {content[0].get('text', '')[:200]}...")
                return True
        
        print("  ✗ 相关性分析失败")
        if response and "error" in response:
            print(f"  错误详情: {json.dumps(response['error'], indent=2, ensure_ascii=False)}")
        return False
        
    def test_get_resource(self):
        """测试获取资源"""
        print("\n📋 测试7: 获取示例数据集资源")
        response = self.send_request("resources/read", {
            "uri": "dataset://sample/economic_growth"
        })
        
        if response and "result" in response:
            contents = response['result'].get('contents', [])
            if contents and len(contents) > 0:
                print("  ✓ 资源获取成功")
                print(f"  数据预览: {contents[0].get('text', '')[:200]}...")
                return True
        
        print("  ✗ 资源获取失败")
        if response and "error" in response:
            print(f"  错误: {response['error']}")
        return False
        
    def cleanup(self):
        """清理资源"""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            
    def run_all_tests(self):
        """运行所有测试"""
        print("="*60)
        print("MCP服务器功能测试")
        print("="*60)
        
        try:
            self.start_server()
            
            results = []
            results.append(("初始化", self.test_initialize()))
            results.append(("工具列表", self.test_list_tools()))
            results.append(("资源列表", self.test_list_resources()))
            results.append(("提示词列表", self.test_list_prompts()))
            results.append(("描述性统计", self.test_descriptive_statistics()))
            results.append(("相关性分析", self.test_correlation_analysis()))
            results.append(("获取资源", self.test_get_resource()))
            
            print("\n" + "="*60)
            print("测试总结")
            print("="*60)
            
            passed = sum(1 for _, result in results if result)
            total = len(results)
            
            for name, result in results:
                status = "✓ 通过" if result else "✗ 失败"
                print(f"  {status} - {name}")
                
            print(f"\n总计: {passed}/{total} 测试通过")
            
            if passed == total:
                print("\n🎉 所有测试通过！MCP服务器功能正常！")
                return True
            else:
                print(f"\n⚠️  {total - passed} 个测试失败")
                return False
                
        except Exception as e:
            print(f"\n❌ 测试过程中出现异常: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup()

if __name__ == "__main__":
    tester = MCPTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)