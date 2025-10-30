"""
优化示例模块
演示如何使用新的优化组件
"""

import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from .base import econometric_tool, validate_input, EconometricTool
from .validation import validate_econometric_data
from .cache import cache_result
from .monitoring import monitor_performance, track_progress
from ..config import get_config


class OptimizedResult(BaseModel):
    """优化结果"""
    mean: float
    std: float
    confidence_interval: Dict[str, float]
    sample_size: int
    performance_stats: Optional[Dict[str, Any]] = None


class OptimizedExampleTool(EconometricTool):
    """
    优化示例工具
    演示如何使用所有优化组件
    """
    
    def __init__(self):
        super().__init__("optimized_example")
    
    @validate_input(data_type="econometric")
    @econometric_tool("optimized_analysis")
    def analyze_data(self, data: Dict[str, List[float]]) -> OptimizedResult:
        """
        分析数据（使用所有优化组件）
        
        Args:
            data: 输入数据字典
            
        Returns:
            OptimizedResult: 分析结果
        """
        # 使用进度跟踪
        with self.execute_with_progress(3, "数据分析") as tracker:
            tracker.start_step("数据预处理")
            # 数据预处理
            processed_data = self._preprocess_data(data)
            tracker.complete_step()
            
            tracker.start_step("统计分析")
            # 统计分析
            result = self._perform_analysis(processed_data)
            tracker.complete_step()
            
            tracker.start_step("结果整理")
            # 添加性能统计
            performance_stats = self.get_performance_stats()
            result.performance_stats = performance_stats
            tracker.complete_step()
        
        return result
    
    def _preprocess_data(self, data: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
        """数据预处理"""
        processed = {}
        for key, values in data.items():
            # 转换为numpy数组并处理缺失值
            arr = np.array(values)
            arr = np.nan_to_num(arr, nan=np.nanmean(arr))
            processed[key] = arr
        return processed
    
    def _perform_analysis(self, data: Dict[str, np.ndarray]) -> OptimizedResult:
        """执行分析"""
        results = {}
        
        for key, values in data.items():
            mean = np.mean(values)
            std = np.std(values)
            n = len(values)
            
            # 计算置信区间
            confidence_level = 0.95
            z_score = 1.96  # 95%置信水平的z值
            margin_of_error = z_score * (std / np.sqrt(n))
            
            results[key] = {
                "mean": mean,
                "std": std,
                "confidence_interval": {
                    "lower": mean - margin_of_error,
                    "upper": mean + margin_of_error
                },
                "sample_size": n
            }
        
        # 返回第一个变量的结果作为示例
        first_key = list(results.keys())[0]
        result_data = results[first_key]
        
        return OptimizedResult(
            mean=result_data["mean"],
            std=result_data["std"],
            confidence_interval=result_data["confidence_interval"],
            sample_size=result_data["sample_size"]
        )
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        from .monitoring import global_performance_monitor
        
        stats = global_performance_monitor.get_function_stats("optimized_analysis")
        if stats:
            return {
                "execution_time": stats.execution_time,
                "call_count": stats.call_count,
                "average_time": stats.average_time,
                "last_execution": stats.last_execution.isoformat() if stats.last_execution else None
            }
        return {}


# 使用便捷装饰器的独立函数示例
@validate_input(data_type="econometric")
@econometric_tool("quick_analysis")
def quick_data_analysis(data: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    快速数据分析（使用装饰器）
    
    Args:
        data: 输入数据
        
    Returns:
        Dict[str, Any]: 分析结果
    """
    # 验证数据
    validated_data = validate_econometric_data(data)
    
    results = {}
    for key, values in validated_data.items():
        arr = np.array(values)
        results[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(arr)
        }
    
    return results


@cache_result(ttl=3600)  # 1小时缓存
@monitor_performance("cached_calculation")
def expensive_calculation(data: List[float], iterations: int = 1000) -> Dict[str, float]:
    """
    昂贵的计算（使用缓存和性能监控）
    
    Args:
        data: 输入数据
        iterations: 迭代次数
        
    Returns:
        Dict[str, float]: 计算结果
    """
    arr = np.array(data)
    result = 0.0
    
    # 模拟昂贵计算
    for i in range(iterations):
        result += np.sum(arr * i) / (i + 1)
    
    return {
        "result": float(result),
        "iterations": iterations,
        "data_length": len(data)
    }


# 使用配置的示例
def config_based_analysis(data: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    基于配置的分析
    
    Args:
        data: 输入数据
        
    Returns:
        Dict[str, Any]: 分析结果
    """
    # 获取配置
    cache_enabled = get_config("cache_enabled", True)
    monitoring_enabled = get_config("monitoring_enabled", True)
    min_sample_size = get_config("min_sample_size", 10)
    
    results = {}
    
    for key, values in data.items():
        # 检查样本大小
        if len(values) < min_sample_size:
            raise ValueError(f"样本量不足: {key} (需要至少{min_sample_size}个观测点)")
        
        # 执行分析
        arr = np.array(values)
        results[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "sample_size": len(arr),
            "config_used": {
                "cache_enabled": cache_enabled,
                "monitoring_enabled": monitoring_enabled,
                "min_sample_size": min_sample_size
            }
        }
    
    return results


# 导出主要类和函数
__all__ = [
    "OptimizedResult",
    "OptimizedExampleTool",
    "quick_data_analysis",
    "expensive_calculation",
    "config_based_analysis"
]