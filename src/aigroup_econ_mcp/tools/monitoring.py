"""
性能监控模块
提供性能监控、内存使用跟踪和进度报告功能
"""

import time
import psutil
import threading
import gc
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import warnings
import sys
import os


class PerformanceMetric(Enum):
    """性能指标类型"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    THREAD_COUNT = "thread_count"
    GC_COLLECTIONS = "gc_collections"


@dataclass
class PerformanceStats:
    """性能统计信息"""
    execution_time: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_percent: float = 0.0
    thread_count: int = 0
    gc_collections: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProgressReport:
    """进度报告"""
    current_step: int
    total_steps: int
    step_name: str
    progress_percent: float
    elapsed_time: float
    estimated_remaining: float
    status: str


class PerformanceMonitor:
    """
    性能监控器
    跟踪函数执行时间、内存使用等性能指标
    """
    
    def __init__(self, enabled: bool = True):
        """
        初始化性能监控器
        
        Args:
            enabled: 是否启用性能监控
        """
        self.enabled = enabled
        self._stats: Dict[str, PerformanceStats] = {}
        self._process = psutil.Process()
        self._lock = threading.RLock()
        
        # 初始内存基准
        self._initial_memory = self._get_memory_usage()
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        try:
            memory_info = self._process.memory_info()
            return memory_info.rss / 1024 / 1024  # 转换为MB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """获取当前CPU使用率"""
        try:
            return self._process.cpu_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def _get_thread_count(self) -> int:
        """获取线程数量"""
        try:
            return self._process.num_threads()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0
    
    def _get_gc_stats(self) -> int:
        """获取GC收集统计"""
        return sum(gc.get_count())
    
    @contextmanager
    def monitor_function(self, function_name: str):
        """
        监控函数执行的上下文管理器
        
        Args:
            function_name: 函数名称
            
        Yields:
            PerformanceStats: 性能统计信息
        """
        if not self.enabled:
            yield PerformanceStats()
            return
        
        # 记录初始状态
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()
        start_threads = self._get_thread_count()
        start_gc = self._get_gc_stats()
        
        peak_memory = start_memory
        
        try:
            yield PerformanceStats()
            
        finally:
            # 计算性能指标
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()
            end_threads = self._get_thread_count()
            end_gc = self._get_gc_stats()
            
            # 更新峰值内存
            peak_memory = max(peak_memory, end_memory)
            
            # 创建统计信息
            stats = PerformanceStats(
                execution_time=end_time - start_time,
                peak_memory_mb=peak_memory - self._initial_memory,
                cpu_percent=end_cpu,
                thread_count=end_threads,
                gc_collections=end_gc - start_gc,
                timestamp=end_time
            )
            
            # 保存统计信息
            with self._lock:
                self._stats[function_name] = stats
    
    def get_function_stats(self, function_name: str) -> Optional[PerformanceStats]:
        """
        获取函数性能统计
        
        Args:
            function_name: 函数名称
            
        Returns:
            Optional[PerformanceStats]: 性能统计信息
        """
        return self._stats.get(function_name)
    
    def get_all_stats(self) -> Dict[str, PerformanceStats]:
        """
        获取所有性能统计
        
        Returns:
            Dict[str, PerformanceStats]: 所有函数的性能统计
        """
        return self._stats.copy()
    
    def clear_stats(self) -> None:
        """清空性能统计"""
        with self._lock:
            self._stats.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取性能监控摘要
        
        Returns:
            Dict[str, Any]: 性能摘要
        """
        if not self._stats:
            return {}
        
        total_time = sum(stats.execution_time for stats in self._stats.values())
        avg_time = total_time / len(self._stats)
        max_time = max(stats.execution_time for stats in self._stats.values())
        min_time = min(stats.execution_time for stats in self._stats.values())
        
        total_memory = sum(stats.peak_memory_mb for stats in self._stats.values())
        avg_memory = total_memory / len(self._stats)
        max_memory = max(stats.peak_memory_mb for stats in self._stats.values())
        
        return {
            "total_functions": len(self._stats),
            "total_execution_time": total_time,
            "average_execution_time": avg_time,
            "max_execution_time": max_time,
            "min_execution_time": min_time,
            "average_memory_usage_mb": avg_memory,
            "max_memory_usage_mb": max_memory,
            "monitored_functions": list(self._stats.keys())
        }


class ProgressTracker:
    """
    进度跟踪器
    提供详细的进度报告和状态更新
    """
    
    def __init__(self, total_steps: int, description: str = ""):
        """
        初始化进度跟踪器
        
        Args:
            total_steps: 总步骤数
            description: 进度描述
        """
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0
        self.start_time = time.time()
        self._step_times: List[float] = []
        self._step_names: List[str] = []
        self._lock = threading.RLock()
    
    def start_step(self, step_name: str = "") -> None:
        """
        开始新步骤
        
        Args:
            step_name: 步骤名称
        """
        with self._lock:
            self.current_step += 1
            self._step_names.append(step_name)
            self._step_times.append(time.time())
    
    def complete_step(self) -> None:
        """完成当前步骤"""
        with self._lock:
            if self._step_times:
                step_start_time = self._step_times[-1]
                step_duration = time.time() - step_start_time
                self._step_times[-1] = step_duration
    
    def get_progress_report(self) -> ProgressReport:
        """
        获取进度报告
        
        Returns:
            ProgressReport: 进度报告
        """
        with self._lock:
            elapsed_time = time.time() - self.start_time
            
            # 计算进度百分比
            progress_percent = (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0
            
            # 计算预计剩余时间
            if self.current_step > 0:
                avg_time_per_step = elapsed_time / self.current_step
                remaining_steps = self.total_steps - self.current_step
                estimated_remaining = avg_time_per_step * remaining_steps
            else:
                estimated_remaining = 0
            
            # 当前步骤名称
            current_step_name = self._step_names[-1] if self._step_names else ""
            
            # 状态描述
            if self.current_step == 0:
                status = "等待开始"
            elif self.current_step < self.total_steps:
                status = "进行中"
            else:
                status = "已完成"
            
            return ProgressReport(
                current_step=self.current_step,
                total_steps=self.total_steps,
                step_name=current_step_name,
                progress_percent=progress_percent,
                elapsed_time=elapsed_time,
                estimated_remaining=estimated_remaining,
                status=status
            )
    
    def print_progress(self) -> None:
        """打印进度信息"""
        report = self.get_progress_report()
        
        progress_bar = self._create_progress_bar(report.progress_percent)
        
        print(f"\r{self.description}: {progress_bar} {report.progress_percent:.1f}% "
              f"({report.current_step}/{report.total_steps}) - {report.step_name} "
              f"[已用: {report.elapsed_time:.1f}s, 剩余: {report.estimated_remaining:.1f}s]",
              end="", flush=True)
        
        if report.status == "已完成":
            print()  # 完成后换行
    
    def _create_progress_bar(self, percent: float, length: int = 20) -> str:
        """
        创建进度条
        
        Args:
            percent: 进度百分比
            length: 进度条长度
            
        Returns:
            str: 进度条字符串
        """
        filled_length = int(length * percent / 100)
        bar = '█' * filled_length + '░' * (length - filled_length)
        return f"[{bar}]"
    
    def get_step_times(self) -> List[Dict[str, Any]]:
        """
        获取各步骤执行时间
        
        Returns:
            List[Dict[str, Any]]: 步骤时间信息
        """
        step_info = []
        for i, (step_name, step_time) in enumerate(zip(self._step_names, self._step_times)):
            step_info.append({
                "step": i + 1,
                "name": step_name,
                "duration": step_time,
                "percentage": (step_time / sum(self._step_times)) * 100 if self._step_times else 0
            })
        return step_info


class MemoryMonitor:
    """
    内存监控器
    跟踪内存使用情况和内存泄漏检测
    """
    
    def __init__(self, check_interval: float = 1.0):
        """
        初始化内存监控器
        
        Args:
            check_interval: 检查间隔（秒）
        """
        self.check_interval = check_interval
        self._process = psutil.Process()
        self._memory_samples: List[float] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start_monitoring(self) -> None:
        """开始内存监控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._stop_event.clear()
        self._memory_samples.clear()
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> List[float]:
        """
        停止内存监控
        
        Returns:
            List[float]: 内存使用样本
        """
        self._monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        return self._memory_samples.copy()
    
    def _monitor_loop(self) -> None:
        """内存监控循环"""
        while not self._stop_event.is_set():
            try:
                memory_usage = self._get_memory_usage()
                self._memory_samples.append(memory_usage)
                
                # 等待检查间隔
                self._stop_event.wait(self.check_interval)
                
            except Exception as e:
                warnings.warn(f"内存监控错误: {e}")
                break
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量（MB）"""
        try:
            memory_info = self._process.memory_info()
            return memory_info.rss / 1024 / 1024  # 转换为MB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """
        分析内存使用情况
        
        Returns:
            Dict[str, Any]: 内存使用分析
        """
        if not self._memory_samples:
            return {}
        
        samples = self._memory_samples
        initial_memory = samples[0] if samples else 0
        peak_memory = max(samples)
        final_memory = samples[-1] if samples else 0
        memory_increase = final_memory - initial_memory
        
        # 检测内存泄漏（持续增长）
        memory_growth_rate = self._calculate_growth_rate(samples)
        
        return {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": memory_increase,
            "memory_growth_rate": memory_growth_rate,
            "sample_count": len(samples),
            "potential_leak": memory_growth_rate > 0.1 and memory_increase > 10.0  # 10MB增长且增长率>10%
        }
    
    def _calculate_growth_rate(self, samples: List[float]) -> float:
        """计算内存增长率"""
        if len(samples) < 2:
            return 0.0
        
        # 使用线性回归计算增长率
        x = list(range(len(samples)))
        y = samples
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
        sum_x2 = sum(x_i ** 2 for x_i in x)
        
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope


# 全局监控实例
global_performance_monitor = PerformanceMonitor()
global_memory_monitor = MemoryMonitor()


# 便捷装饰器和上下文管理器
def monitor_performance(function_name: str = None):
    """
    性能监控装饰器
    
    Args:
        function_name: 函数名称，如果为None则使用函数名
        
    Returns:
        Callable: 装饰器函数
    """
    def decorator(func):
        name = function_name or func.__name__
        
        def wrapper(*args, **kwargs):
            with global_performance_monitor.monitor_function(name):
                return func(*args, **kwargs)
        
        # 保留原始函数的元数据
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        
        return wrapper
    
    return decorator


@contextmanager
def track_progress(total_steps: int, description: str = ""):
    """
    进度跟踪上下文管理器
    
    Args:
        total_steps: 总步骤数
        description: 进度描述
        
    Yields:
        ProgressTracker: 进度跟踪器
    """
    tracker = ProgressTracker(total_steps, description)
    
    try:
        yield tracker
    finally:
        # 确保进度显示完成
        if tracker.current_step < tracker.total_steps:
            tracker.current_step = tracker.total_steps
            tracker.print_progress()


def enable_memory_monitoring(check_interval: float = 1.0) -> MemoryMonitor:
    """
    启用内存监控
    
    Args:
        check_interval: 检查间隔（秒）
        
    Returns:
        MemoryMonitor: 内存监控器实例
    """
    global_memory_monitor.check_interval = check_interval
    global_memory_monitor.start_monitoring()
    return global_memory_monitor


def disable_memory_monitoring() -> Dict[str, Any]:
    """
    禁用内存监控并返回分析结果
    
    Returns:
        Dict[str, Any]: 内存使用分析
    """
    global_memory_monitor.stop_monitoring()
    return global_memory_monitor.analyze_memory_usage()


# 导出主要类和函数
__all__ = [
    "PerformanceMetric",
    "PerformanceStats",
    "ProgressReport",
    "PerformanceMonitor",
    "ProgressTracker",
    "MemoryMonitor",
    "global_performance_monitor",
    "global_memory_monitor",
    "monitor_performance",
    "track_progress",
    "enable_memory_monitoring",
    "disable_memory_monitoring"
]