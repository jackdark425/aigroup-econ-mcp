#!/usr/bin/env python3
"""
测试运行脚本
提供统一的测试运行接口和测试报告生成
"""

import sys
import os
import subprocess
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))


class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
    
    def run_unit_tests(self, verbose: bool = False) -> bool:
        """运行单元测试"""
        print("\n" + "="*80)
        print("运行单元测试")
        print("="*80)
        
        cmd = ["pytest", "tests/test_unit/", "-v" if verbose else "-q", "--tb=short"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            self.test_results["unit_tests"] = {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            return result.returncode == 0
        except Exception as e:
            print(f"运行单元测试时出错: {e}")
            return False
    
    def run_integration_tests(self, verbose: bool = False) -> bool:
        """运行集成测试"""
        print("\n" + "="*80)
        print("运行集成测试")
        print("="*80)
        
        cmd = ["pytest", "tests/test_integration/", "-v" if verbose else "-q", "--tb=short"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            self.test_results["integration_tests"] = {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            return result.returncode == 0
        except Exception as e:
            print(f"运行集成测试时出错: {e}")
            return False
    
    def run_file_input_tests(self, verbose: bool = False) -> bool:
        """运行文件输入测试"""
        print("\n" + "="*80)
        print("运行文件输入测试")
        print("="*80)
        
        cmd = ["pytest", "tests/test_integration/test_file_input.py", "-v" if verbose else "-q", "--tb=short"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            self.test_results["file_input_tests"] = {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            return result.returncode == 0
        except Exception as e:
            print(f"运行文件输入测试时出错: {e}")
            return False
    
    def run_all_tests(self, verbose: bool = False) -> bool:
        """运行所有测试"""
        print("\n" + "="*80)
        print("开始运行所有测试")
        print("="*80)
        
        self.start_time = datetime.now()
        
        # 运行各种类型的测试
        unit_success = self.run_unit_tests(verbose)
        integration_success = self.run_integration_tests(verbose)
        file_input_success = self.run_file_input_tests(verbose)
        
        self.end_time = datetime.now()
        
        # 生成测试报告
        self.generate_test_report()
        
        return all([unit_success, integration_success, file_input_success])
    
    def generate_test_report(self):
        """生成测试报告"""
        print("\n" + "="*80)
        print("测试报告")
        print("="*80)
        
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            print(f"测试时间: {duration}")
        
        # 统计测试结果
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for test_type, result in self.test_results.items():
            return_code = result["return_code"]
            if return_code == 0:
                passed_tests += 1
                status = "✅ 通过"
            else:
                failed_tests += 1
                status = "❌ 失败"
            
            total_tests += 1
            print(f"{status}: {test_type}")
        
        print("\n" + "-"*80)
        print(f"总计: {passed_tests}/{total_tests} 测试通过")
        
        if failed_tests == 0:
            print("\n🎉 所有测试通过！")
        else:
            print(f"\n⚠️  有 {failed_tests} 个测试失败")
        
        # 保存详细报告到文件
        self.save_detailed_report()
    
    def save_detailed_report(self):
        """保存详细测试报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_results": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "passed_tests": sum(1 for r in self.test_results.values() if r["return_code"] == 0),
                "failed_tests": sum(1 for r in self.test_results.values() if r["return_code"] != 0)
            }
        }
        
        report_file = "test_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 详细测试报告已保存到: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行aigroup-econ-mcp测试")
    parser.add_argument(
        "--unit", 
        action="store_true", 
        help="只运行单元测试"
    )
    parser.add_argument(
        "--integration", 
        action="store_true", 
        help="只运行集成测试"
    )
    parser.add_argument(
        "--file-input", 
        action="store_true", 
        help="只运行文件输入测试"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="详细输出"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="运行所有测试（默认）"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.unit:
        success = runner.run_unit_tests(args.verbose)
    elif args.integration:
        success = runner.run_integration_tests(args.verbose)
    elif args.file_input:
        success = runner.run_file_input_tests(args.verbose)
    else:
        # 默认运行所有测试
        success = runner.run_all_tests(args.verbose)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)