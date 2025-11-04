"""


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
    """"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    THREAD_COUNT = "thread_count"
    GC_COLLECTIONS = "gc_collections"


@dataclass
class PerformanceStats:
    """"""
    execution_time: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_percent: float = 0.0
    thread_count: int = 0
    gc_collections: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProgressReport:
    """"""
    current_step: int
    total_steps: int
    step_name: str
    progress_percent: float
    elapsed_time: float
    estimated_remaining: float
    status: str


class PerformanceMonitor:
    """
    
    
    """
    
    def __init__(self, enabled: bool = True):
        """
        
        
        Args:
            enabled: 
        """
        self.enabled = enabled
        self._stats: Dict[str, PerformanceStats] = {}
        self._process = psutil.Process()
        self._lock = threading.RLock()
        
        # 
        self._initial_memory = self._get_memory_usage()
    
    def _get_memory_usage(self) -> float:
        """MB"""
        try:
            memory_info = self._process.memory_info()
            return memory_info.rss / 1024 / 1024  # MB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """CPU"""
        try:
            return self._process.cpu_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def _get_thread_count(self) -> int:
        """"""
        try:
            return self._process.num_threads()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0
    
    def _get_gc_stats(self) -> int:
        """GC"""
        return sum(gc.get_count())
    
    @contextmanager
    def monitor_function(self, function_name: str):
        """
        
        
        Args:
            function_name: 
            
        Yields:
            PerformanceStats: 
        """
        if not self.enabled:
            yield PerformanceStats()
            return
        
        # 
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self._get_cpu_usage()
        start_threads = self._get_thread_count()
        start_gc = self._get_gc_stats()
        
        peak_memory = start_memory
        
        try:
            yield PerformanceStats()
            
        finally:
            # 
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = self._get_cpu_usage()
            end_threads = self._get_thread_count()
            end_gc = self._get_gc_stats()
            
            # 
            peak_memory = max(peak_memory, end_memory)
            
            # 
            stats = PerformanceStats(
                execution_time=end_time - start_time,
                peak_memory_mb=peak_memory - self._initial_memory,
                cpu_percent=end_cpu,
                thread_count=end_threads,
                gc_collections=end_gc - start_gc,
                timestamp=end_time
            )
            
            # 
            with self._lock:
                self._stats[function_name] = stats
    
    def get_function_stats(self, function_name: str) -> Optional[PerformanceStats]:
        """
        
        
        Args:
            function_name: 
            
        Returns:
            Optional[PerformanceStats]: 
        """
        return self._stats.get(function_name)
    
    def get_all_stats(self) -> Dict[str, PerformanceStats]:
        """
        
        
        Returns:
            Dict[str, PerformanceStats]: 
        """
        return self._stats.copy()
    
    def clear_stats(self) -> None:
        """"""
        with self._lock:
            self._stats.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        
        
        Returns:
            Dict[str, Any]: 
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
    
    
    """
    
    def __init__(self, total_steps: int, description: str = ""):
        """
        
        
        Args:
            total_steps: 
            description: 
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
        
        
        Args:
            step_name: 
        """
        with self._lock:
            self.current_step += 1
            self._step_names.append(step_name)
            self._step_times.append(time.time())
    
    def complete_step(self) -> None:
        """"""
        with self._lock:
            if self._step_times:
                step_start_time = self._step_times[-1]
                step_duration = time.time() - step_start_time
                self._step_times[-1] = step_duration
    
    def get_progress_report(self) -> ProgressReport:
        """
        
        
        Returns:
            ProgressReport: 
        """
        with self._lock:
            elapsed_time = time.time() - self.start_time
            
            # 
            progress_percent = (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0
            
            # 
            if self.current_step > 0:
                avg_time_per_step = elapsed_time / self.current_step
                remaining_steps = self.total_steps - self.current_step
                estimated_remaining = avg_time_per_step * remaining_steps
            else:
                estimated_remaining = 0
            
            # 
            current_step_name = self._step_names[-1] if self._step_names else ""
            
            # 
            if self.current_step == 0:
                status = ""
            elif self.current_step < self.total_steps:
                status = ""
            else:
                status = ""
            
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
        """"""
        report = self.get_progress_report()
        
        progress_bar = self._create_progress_bar(report.progress_percent)
        
        print(f"\r{self.description}: {progress_bar} {report.progress_percent:.1f}% "
              f"({report.current_step}/{report.total_steps}) - {report.step_name} "
              f"[: {report.elapsed_time:.1f}s, : {report.estimated_remaining:.1f}s]",
              end="", flush=True)
        
        if report.status == "":
            print()  # 
    
    def _create_progress_bar(self, percent: float, length: int = 20) -> str:
        """
        
        
        Args:
            percent: 
            length: 
            
        Returns:
            str: 
        """
        filled_length = int(length * percent / 100)
        bar = '' * filled_length + '' * (length - filled_length)
        return f"[{bar}]"
    
    def get_step_times(self) -> List[Dict[str, Any]]:
        """
        
        
        Returns:
            List[Dict[str, Any]]: 
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
    
    
    """
    
    def __init__(self, check_interval: float = 1.0):
        """
        
        
        Args:
            check_interval: 
        """
        self.check_interval = check_interval
        self._process = psutil.Process()
        self._memory_samples: List[float] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start_monitoring(self) -> None:
        """"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._stop_event.clear()
        self._memory_samples.clear()
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> List[float]:
        """
        
        
        Returns:
            List[float]: 
        """
        self._monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        return self._memory_samples.copy()
    
    def _monitor_loop(self) -> None:
        """"""
        while not self._stop_event.is_set():
            try:
                memory_usage = self._get_memory_usage()
                self._memory_samples.append(memory_usage)
                
                # 
                self._stop_event.wait(self.check_interval)
                
            except Exception as e:
                warnings.warn(f": {e}")
                break
    
    def _get_memory_usage(self) -> float:
        """MB"""
        try:
            memory_info = self._process.memory_info()
            return memory_info.rss / 1024 / 1024  # MB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """
        
        
        Returns:
            Dict[str, Any]: 
        """
        if not self._memory_samples:
            return {}
        
        samples = self._memory_samples
        initial_memory = samples[0] if samples else 0
        peak_memory = max(samples)
        final_memory = samples[-1] if samples else 0
        memory_increase = final_memory - initial_memory
        
        # 
        memory_growth_rate = self._calculate_growth_rate(samples)
        
        return {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": memory_increase,
            "memory_growth_rate": memory_growth_rate,
            "sample_count": len(samples),
            "potential_leak": memory_growth_rate > 0.1 and memory_increase > 10.0  # 10MB>10%
        }
    
    def _calculate_growth_rate(self, samples: List[float]) -> float:
        """"""
        if len(samples) < 2:
            return 0.0
        
        # 
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


# 
global_performance_monitor = PerformanceMonitor()
global_memory_monitor = MemoryMonitor()


# 
def monitor_performance(function_name: str = None):
    """
    
    
    Args:
        function_name: None
        
    Returns:
        Callable: 
    """
    def decorator(func):
        name = function_name or func.__name__
        
        def wrapper(*args, **kwargs):
            with global_performance_monitor.monitor_function(name):
                return func(*args, **kwargs)
        
        # 
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        
        return wrapper
    
    return decorator


@contextmanager
def track_progress(total_steps: int, description: str = ""):
    """
    
    
    Args:
        total_steps: 
        description: 
        
    Yields:
        ProgressTracker: 
    """
    tracker = ProgressTracker(total_steps, description)
    
    try:
        yield tracker
    finally:
        # 
        if tracker.current_step < tracker.total_steps:
            tracker.current_step = tracker.total_steps
            tracker.print_progress()


def enable_memory_monitoring(check_interval: float = 1.0) -> MemoryMonitor:
    """
    
    
    Args:
        check_interval: 
        
    Returns:
        MemoryMonitor: 
    """
    global_memory_monitor.check_interval = check_interval
    global_memory_monitor.start_monitoring()
    return global_memory_monitor


def disable_memory_monitoring() -> Dict[str, Any]:
    """
    
    
    Returns:
        Dict[str, Any]: 
    """
    global_memory_monitor.stop_monitoring()
    return global_memory_monitor.analyze_memory_usage()


# 
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