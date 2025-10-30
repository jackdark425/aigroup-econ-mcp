"""
缓存机制模块
提供LRU缓存策略和结果缓存功能
"""

import time
import hashlib
import pickle
import threading
from typing import Any, Dict, List, Optional, Callable, Tuple
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import json
import warnings


class CachePolicy(Enum):
    """缓存策略"""
    LRU = "lru"  # 最近最少使用
    FIFO = "fifo"  # 先进先出
    LFU = "lfu"  # 最不经常使用


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    size: int = 0


class LRUCache:
    """
    LRU缓存实现
    使用最近最少使用策略管理缓存
    """
    
    def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
        """
        初始化LRU缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl: 缓存生存时间（秒），None表示永不过期
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存值，如果不存在或过期则返回None
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # 检查是否过期
            if self.ttl and (time.time() - entry.timestamp) > self.ttl:
                del self._cache[key]
                self._misses += 1
                return None
            
            # 更新访问时间和计数
            entry.timestamp = time.time()
            entry.access_count += 1
            self._cache.move_to_end(key)
            
            self._hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, size: int = 0) -> None:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            size: 缓存大小（字节），用于大小限制
        """
        with self._lock:
            # 如果键已存在，先删除
            if key in self._cache:
                del self._cache[key]
            
            # 创建新的缓存条目
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                size=size
            )
            
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            # 如果超过最大大小，移除最旧的条目
            while len(self._cache) > self.max_size:
                oldest_key, oldest_entry = next(iter(self._cache.items()))
                del self._cache[oldest_key]
                self._evictions += 1
    
    def delete(self, key: str) -> bool:
        """
        删除缓存条目
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否成功删除
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
    
    def contains(self, key: str) -> bool:
        """
        检查缓存是否包含指定键
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否包含
        """
        with self._lock:
            if key not in self._cache:
                return False
            
            # 检查是否过期
            entry = self._cache[key]
            if self.ttl and (time.time() - entry.timestamp) > self.ttl:
                del self._cache[key]
                return False
            
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
                "memory_usage": sum(entry.size for entry in self._cache.values())
            }
    
    def keys(self) -> List[str]:
        """
        获取所有缓存键
        
        Returns:
            List[str]: 缓存键列表
        """
        with self._lock:
            return list(self._cache.keys())


class ResultCache:
    """
    结果缓存管理器
    提供函数结果缓存和智能缓存管理
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        初始化结果缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl: 缓存生存时间（秒）
        """
        self.cache = LRUCache(max_size=max_size, ttl=ttl)
        self._function_cache: Dict[str, LRUCache] = {}
    
    def generate_key(self, func_name: str, *args, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            func_name: 函数名称
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            str: 缓存键
        """
        # 序列化参数
        try:
            # 使用JSON序列化，处理基本数据类型
            args_str = json.dumps(args, sort_keys=True, default=str)
            kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
            key_data = f"{func_name}:{args_str}:{kwargs_str}"
        except (TypeError, ValueError):
            # 如果JSON序列化失败，使用pickle
            key_data = f"{func_name}:{pickle.dumps((args, kwargs))}"
        
        # 生成哈希键
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cached(self, func: Callable = None, *, ttl: Optional[int] = None, max_size: int = 100):
        """
        缓存装饰器
        
        Args:
            func: 被装饰的函数
            ttl: 缓存生存时间（秒）
            max_size: 函数特定的最大缓存大小
            
        Returns:
            Callable: 装饰后的函数
        """
        def decorator(f):
            func_name = f.__name__
            
            # 为每个函数创建独立的缓存
            if func_name not in self._function_cache:
                self._function_cache[func_name] = LRUCache(
                    max_size=max_size,
                    ttl=ttl or self.cache.ttl
                )
            
            func_cache = self._function_cache[func_name]
            
            def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self.generate_key(func_name, *args, **kwargs)
                
                # 尝试从缓存获取结果
                cached_result = func_cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # 执行函数并缓存结果
                result = f(*args, **kwargs)
                func_cache.set(cache_key, result)
                
                return result
            
            # 保留原始函数的元数据
            wrapper.__name__ = f.__name__
            wrapper.__doc__ = f.__doc__
            wrapper.__module__ = f.__module__
            
            return wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    def get_function_cache_stats(self, func_name: str) -> Optional[Dict[str, Any]]:
        """
        获取函数缓存统计信息
        
        Args:
            func_name: 函数名称
            
        Returns:
            Optional[Dict[str, Any]]: 统计信息，如果函数没有缓存则返回None
        """
        if func_name in self._function_cache:
            return self._function_cache[func_name].get_stats()
        return None
    
    def clear_function_cache(self, func_name: str) -> bool:
        """
        清空指定函数的缓存
        
        Args:
            func_name: 函数名称
            
        Returns:
            bool: 是否成功清空
        """
        if func_name in self._function_cache:
            self._function_cache[func_name].clear()
            return True
        return False
    
    def clear_all(self) -> None:
        """清空所有缓存"""
        for cache in self._function_cache.values():
            cache.clear()
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        获取全局缓存统计信息
        
        Returns:
            Dict[str, Any]: 全局统计信息
        """
        total_hits = 0
        total_misses = 0
        total_size = 0
        function_stats = {}
        
        for func_name, cache in self._function_cache.items():
            stats = cache.get_stats()
            total_hits += stats["hits"]
            total_misses += stats["misses"]
            total_size += stats["memory_usage"]
            function_stats[func_name] = stats
        
        total_requests = total_hits + total_misses
        global_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            "total_functions": len(self._function_cache),
            "total_hits": total_hits,
            "total_misses": total_misses,
            "global_hit_rate": global_hit_rate,
            "total_memory_usage": total_size,
            "function_stats": function_stats
        }


class EconometricCache:
    """
    计量经济学专用缓存
    针对计量经济学计算结果的优化缓存
    """
    
    def __init__(self):
        self.result_cache = ResultCache(max_size=500, ttl=7200)  # 2小时TTL
        self._model_cache: Dict[str, Any] = {}
        self._data_cache: Dict[str, Any] = {}
    
    def cache_model_result(self, model_type: str, data_hash: str, parameters: Dict, result: Any) -> str:
        """
        缓存模型结果
        
        Args:
            model_type: 模型类型
            data_hash: 数据哈希
            parameters: 模型参数
            result: 模型结果
            
        Returns:
            str: 缓存键
        """
        cache_key = f"{model_type}:{data_hash}:{hash(str(parameters))}"
        self._model_cache[cache_key] = {
            "result": result,
            "timestamp": time.time(),
            "model_type": model_type,
            "parameters": parameters
        }
        return cache_key
    
    def get_cached_model_result(self, model_type: str, data_hash: str, parameters: Dict) -> Optional[Any]:
        """
        获取缓存的模型结果
        
        Args:
            model_type: 模型类型
            data_hash: 数据哈希
            parameters: 模型参数
            
        Returns:
            Optional[Any]: 缓存的结果，如果不存在则返回None
        """
        cache_key = f"{model_type}:{data_hash}:{hash(str(parameters))}"
        return self._model_cache.get(cache_key, {}).get("result")
    
    def cache_data_analysis(self, data_hash: str, analysis_type: str, result: Any) -> str:
        """
        缓存数据分析结果
        
        Args:
            data_hash: 数据哈希
            analysis_type: 分析类型
            result: 分析结果
            
        Returns:
            str: 缓存键
        """
        cache_key = f"analysis:{analysis_type}:{data_hash}"
        self._data_cache[cache_key] = {
            "result": result,
            "timestamp": time.time(),
            "analysis_type": analysis_type
        }
        return cache_key
    
    def get_cached_data_analysis(self, data_hash: str, analysis_type: str) -> Optional[Any]:
        """
        获取缓存的数据分析结果
        
        Args:
            data_hash: 数据哈希
            analysis_type: 分析类型
            
        Returns:
            Optional[Any]: 缓存的结果，如果不存在则返回None
        """
        cache_key = f"analysis:{analysis_type}:{data_hash}"
        return self._data_cache.get(cache_key, {}).get("result")
    
    def clear_old_entries(self, max_age: int = 86400) -> int:
        """
        清理过期的缓存条目
        
        Args:
            max_age: 最大年龄（秒）
            
        Returns:
            int: 清理的条目数量
        """
        current_time = time.time()
        removed_count = 0
        
        # 清理模型缓存
        keys_to_remove = []
        for key, entry in self._model_cache.items():
            if current_time - entry["timestamp"] > max_age:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._model_cache[key]
            removed_count += 1
        
        # 清理数据缓存
        keys_to_remove = []
        for key, entry in self._data_cache.items():
            if current_time - entry["timestamp"] > max_age:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._data_cache[key]
            removed_count += 1
        
        return removed_count
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息
        
        Returns:
            Dict[str, Any]: 缓存信息
        """
        return {
            "model_cache_size": len(self._model_cache),
            "data_cache_size": len(self._data_cache),
            "result_cache_stats": self.result_cache.get_global_stats()
        }


# 全局缓存实例
global_econometric_cache = EconometricCache()


# 便捷缓存装饰器
def cache_result(ttl: int = 3600, max_size: int = 100):
    """
    便捷的结果缓存装饰器
    
    Args:
        ttl: 缓存生存时间（秒）
        max_size: 最大缓存大小
        
    Returns:
        Callable: 装饰器函数
    """
    return global_econometric_cache.result_cache.cached(ttl=ttl, max_size=max_size)


def cache_model(ttl: int = 7200):
    """
    模型结果缓存装饰器
    
    Args:
        ttl: 缓存生存时间（秒）
        
    Returns:
        Callable: 装饰器函数
    """
    def decorator(func):
        @cache_result(ttl=ttl, max_size=50)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# 导出主要类和函数
__all__ = [
    "CachePolicy",
    "LRUCache",
    "ResultCache",
    "EconometricCache",
    "global_econometric_cache",
    "cache_result",
    "cache_model"
]