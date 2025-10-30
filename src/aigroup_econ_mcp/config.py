"""
配置管理系统
提供统一的配置管理和环境设置功能
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging
from dotenv import load_dotenv


class ConfigSource(Enum):
    """配置来源"""
    ENV = "environment"
    FILE = "file"
    DEFAULT = "default"


@dataclass
class ConfigValue:
    """配置值"""
    value: Any
    source: ConfigSource
    key: str
    description: str = ""
    required: bool = False
    validated: bool = False


class ConfigurationError(Exception):
    """配置错误"""
    pass


class ConfigManager:
    """
    配置管理器
    提供统一的配置加载、验证和管理功能
    """
    
    def __init__(self, config_dir: Optional[str] = None, env_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
            env_file: 环境变量文件路径
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self.env_file = Path(env_file) if env_file else Path.cwd() / ".env"
        
        # 创建配置目录（如果不存在）
        self.config_dir.mkdir(exist_ok=True)
        
        # 加载环境变量
        self._load_environment_variables()
        
        # 配置存储
        self._config: Dict[str, ConfigValue] = {}
        self._defaults: Dict[str, Any] = {}
        self._validators: Dict[str, callable] = {}
        
        # 日志设置
        self._setup_logging()
    
    def _load_environment_variables(self) -> None:
        """加载环境变量"""
        if self.env_file.exists():
            load_dotenv(self.env_file)
        else:
            load_dotenv()
    
    def _setup_logging(self) -> None:
        """设置日志"""
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger("aigroup_econ_mcp")
    
    def set_default(self, key: str, value: Any, description: str = "", required: bool = False) -> None:
        """
        设置默认配置值
        
        Args:
            key: 配置键
            value: 默认值
            description: 配置描述
            required: 是否必需
        """
        self._defaults[key] = {
            "value": value,
            "description": description,
            "required": required
        }
    
    def add_validator(self, key: str, validator: callable) -> None:
        """
        添加配置验证器
        
        Args:
            key: 配置键
            validator: 验证函数
        """
        self._validators[key] = validator
    
    def load_from_file(self, filename: str, file_type: str = "auto") -> None:
        """
        从文件加载配置
        
        Args:
            filename: 文件名
            file_type: 文件类型 (json, yaml, auto)
        """
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            self.logger.warning(f"配置文件不存在: {file_path}")
            return
        
        try:
            if file_type == "auto":
                if filename.endswith(".json"):
                    file_type = "json"
                elif filename.endswith((".yaml", ".yml")):
                    file_type = "yaml"
                else:
                    raise ConfigurationError(f"无法自动识别文件类型: {filename}")
            
            if file_type == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            elif file_type == "yaml":
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
            else:
                raise ConfigurationError(f"不支持的文件类型: {file_type}")
            
            # 更新配置
            self._update_config_from_dict(config_data, ConfigSource.FILE)
            
            self.logger.info(f"从文件加载配置: {file_path}")
            
        except Exception as e:
            raise ConfigurationError(f"加载配置文件失败: {file_path}, 错误: {e}")
    
    def load_from_environment(self, prefix: str = "AIGROUP_ECON_") -> None:
        """
        从环境变量加载配置
        
        Args:
            prefix: 环境变量前缀
        """
        env_config = {}
        
        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                # 移除前缀并转换为小写
                config_key = env_key[len(prefix):].lower()
                
                # 尝试解析值
                try:
                    # 尝试解析为JSON
                    parsed_value = json.loads(env_value)
                except (json.JSONDecodeError, ValueError):
                    # 如果不是JSON，保持原样
                    parsed_value = env_value
                
                env_config[config_key] = parsed_value
        
        self._update_config_from_dict(env_config, ConfigSource.ENV)
        self.logger.info(f"从环境变量加载配置，前缀: {prefix}")
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any], source: ConfigSource) -> None:
        """
        从字典更新配置
        
        Args:
            config_dict: 配置字典
            source: 配置来源
        """
        for key, value in config_dict.items():
            description = self._defaults.get(key, {}).get("description", "")
            required = self._defaults.get(key, {}).get("required", False)
            
            # 验证配置值
            if key in self._validators:
                try:
                    value = self._validators[key](value)
                    validated = True
                except Exception as e:
                    if required:
                        raise ConfigurationError(f"配置验证失败: {key}={value}, 错误: {e}")
                    else:
                        self.logger.warning(f"配置验证失败: {key}={value}, 错误: {e}")
                        validated = False
            else:
                validated = False
            
            self._config[key] = ConfigValue(
                value=value,
                source=source,
                key=key,
                description=description,
                required=required,
                validated=validated
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            Any: 配置值
        """
        if key in self._config:
            return self._config[key].value
        
        # 检查默认值
        if key in self._defaults:
            default_value = self._defaults[key]["value"]
            self._config[key] = ConfigValue(
                value=default_value,
                source=ConfigSource.DEFAULT,
                key=key,
                description=self._defaults[key]["description"],
                required=self._defaults[key]["required"],
                validated=False
            )
            return default_value
        
        # 使用提供的默认值
        if default is not None:
            return default
        
        # 如果配置不存在且没有默认值
        if key in self._defaults and self._defaults[key]["required"]:
            raise ConfigurationError(f"必需的配置项缺失: {key}")
        
        return None
    
    def set(self, key: str, value: Any, description: str = "") -> None:
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
            description: 配置描述
        """
        # 验证配置值
        validated = False
        if key in self._validators:
            try:
                value = self._validators[key](value)
                validated = True
            except Exception as e:
                raise ConfigurationError(f"配置验证失败: {key}={value}, 错误: {e}")
        
        required = self._defaults.get(key, {}).get("required", False)
        
        self._config[key] = ConfigValue(
            value=value,
            source=ConfigSource.FILE,  # 手动设置的配置视为文件配置
            key=key,
            description=description,
            required=required,
            validated=validated
        )
    
    def save_to_file(self, filename: str, file_type: str = "json") -> None:
        """
        保存配置到文件
        
        Args:
            filename: 文件名
            file_type: 文件类型 (json, yaml)
        """
        file_path = self.config_dir / filename
        
        try:
            config_dict = {}
            for key, config_value in self._config.items():
                if config_value.source != ConfigSource.ENV:  # 不保存环境变量配置
                    config_dict[key] = config_value.value
            
            if file_type == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            elif file_type == "yaml":
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ConfigurationError(f"不支持的文件类型: {file_type}")
            
            self.logger.info(f"配置保存到文件: {file_path}")
            
        except Exception as e:
            raise ConfigurationError(f"保存配置文件失败: {file_path}, 错误: {e}")
    
    def validate(self) -> List[str]:
        """
        验证配置
        
        Returns:
            List[str]: 验证错误列表
        """
        errors = []
        
        # 检查必需配置
        for key, default_info in self._defaults.items():
            if default_info["required"] and key not in self._config:
                errors.append(f"必需的配置项缺失: {key}")
        
        # 验证配置值
        for key, config_value in self._config.items():
            if key in self._validators and not config_value.validated:
                try:
                    self._validators[key](config_value.value)
                    config_value.validated = True
                except Exception as e:
                    errors.append(f"配置验证失败: {key}={config_value.value}, 错误: {e}")
        
        return errors
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        获取配置信息
        
        Returns:
            Dict[str, Any]: 配置信息
        """
        config_info = {}
        
        for key, config_value in self._config.items():
            config_info[key] = {
                "value": config_value.value,
                "source": config_value.source.value,
                "description": config_value.description,
                "required": config_value.required,
                "validated": config_value.validated
            }
        
        return config_info
    
    def reload(self) -> None:
        """重新加载配置"""
        self._config.clear()
        self._load_environment_variables()
        self.load_from_environment()


# 预定义配置验证器
class ConfigValidators:
    """配置验证器集合"""
    
    @staticmethod
    def validate_positive_int(value: Any) -> int:
        """验证正整数"""
        try:
            int_value = int(value)
            if int_value <= 0:
                raise ValueError("必须为正整数")
            return int_value
        except (ValueError, TypeError):
            raise ValueError("必须为有效的整数")
    
    @staticmethod
    def validate_positive_float(value: Any) -> float:
        """验证正浮点数"""
        try:
            float_value = float(value)
            if float_value <= 0:
                raise ValueError("必须为正数")
            return float_value
        except (ValueError, TypeError):
            raise ValueError("必须为有效的数字")
    
    @staticmethod
    def validate_boolean(value: Any) -> bool:
        """验证布尔值"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lower() in ("true", "1", "yes", "on"):
                return True
            elif value.lower() in ("false", "0", "no", "off"):
                return False
        raise ValueError("必须为布尔值 (true/false, 1/0, yes/no)")
    
    @staticmethod
    def validate_list(value: Any) -> list:
        """验证列表"""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return [item.strip() for item in value.split(",")]
        raise ValueError("必须为列表或逗号分隔的字符串")
    
    @staticmethod
    def validate_file_path(value: Any) -> str:
        """验证文件路径"""
        path = Path(str(value))
        if not path.exists():
            raise ValueError(f"文件不存在: {path}")
        return str(path.absolute())


# 计量经济学专用配置
class EconometricConfig:
    """计量经济学专用配置"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        初始化计量经济学配置
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config = config_manager
        
        # 设置默认配置
        self._set_defaults()
        
        # 添加验证器
        self._add_validators()
    
    def _set_defaults(self) -> None:
        """设置默认配置"""
        defaults = {
            # 性能配置
            "cache_enabled": (True, "是否启用缓存", False),
            "cache_ttl": (3600, "缓存生存时间（秒）", False),
            "cache_max_size": (1000, "最大缓存条目数", False),
            
            # 监控配置
            "monitoring_enabled": (True, "是否启用性能监控", False),
            "memory_monitoring_interval": (1.0, "内存监控间隔（秒）", False),
            
            # 模型配置
            "default_n_estimators": (100, "默认树的数量", False),
            "default_max_depth": (None, "默认最大深度", False),
            "default_learning_rate": (0.1, "默认学习率", False),
            "default_alpha": (1.0, "默认正则化强度", False),
            
            # 数据配置
            "min_sample_size": (10, "最小样本量", False),
            "max_feature_count": (100, "最大特征数量", False),
            "data_validation_strict": (True, "严格数据验证", False),
            
            # 日志配置
            "log_level": ("INFO", "日志级别", False),
            "log_file": (None, "日志文件路径", False),
            
            # 并行配置
            "parallel_processing": (False, "是否启用并行处理", False),
            "max_workers": (None, "最大工作线程数", False),
        }
        
        for key, (value, description, required) in defaults.items():
            self.config.set_default(key, value, description, required)
    
    def _add_validators(self) -> None:
        """添加验证器"""
        validators = {
            "cache_enabled": ConfigValidators.validate_boolean,
            "cache_ttl": ConfigValidators.validate_positive_int,
            "cache_max_size": ConfigValidators.validate_positive_int,
            "monitoring_enabled": ConfigValidators.validate_boolean,
            "memory_monitoring_interval": ConfigValidators.validate_positive_float,
            "default_n_estimators": ConfigValidators.validate_positive_int,
            "default_learning_rate": ConfigValidators.validate_positive_float,
            "default_alpha": ConfigValidators.validate_positive_float,
            "min_sample_size": ConfigValidators.validate_positive_int,
            "max_feature_count": ConfigValidators.validate_positive_int,
            "data_validation_strict": ConfigValidators.validate_boolean,
            "parallel_processing": ConfigValidators.validate_boolean,
        }
        
        for key, validator in validators.items():
            self.config.add_validator(key, validator)
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            "n_estimators": self.config.get("default_n_estimators"),
            "max_depth": self.config.get("default_max_depth"),
            "learning_rate": self.config.get("default_learning_rate"),
            "alpha": self.config.get("default_alpha"),
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """获取缓存配置"""
        return {
            "enabled": self.config.get("cache_enabled"),
            "ttl": self.config.get("cache_ttl"),
            "max_size": self.config.get("cache_max_size"),
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """获取监控配置"""
        return {
            "enabled": self.config.get("monitoring_enabled"),
            "memory_interval": self.config.get("memory_monitoring_interval"),
        }


# 全局配置实例
global_config_manager = ConfigManager()
econometric_config = EconometricConfig(global_config_manager)

# 便捷配置访问函数
def get_config(key: str, default: Any = None) -> Any:
    """便捷配置访问函数"""
    return global_config_manager.get(key, default)

def set_config(key: str, value: Any, description: str = "") -> None:
    """便捷配置设置函数"""
    global_config_manager.set(key, value, description)

def load_config_files() -> None:
    """加载配置文件"""
    # 加载环境变量配置
    global_config_manager.load_from_environment()
    
    # 尝试加载配置文件
    config_files = ["config.json", "config.yaml", "config.yml"]
    for config_file in config_files:
        if (global_config_manager.config_dir / config_file).exists():
            global_config_manager.load_from_file(config_file)
            break

# 初始化时加载配置
load_config_files()

# 导出主要类和函数
__all__ = [
    "ConfigSource",
    "ConfigValue",
    "ConfigurationError",
    "ConfigManager",
    "ConfigValidators",
    "EconometricConfig",
    "global_config_manager",
    "econometric_config",
    "get_config",
    "set_config",
    "load_config_files"
]