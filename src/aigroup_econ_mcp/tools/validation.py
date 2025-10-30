"""
统一的参数验证系统
提供类型检查、数据验证和参数验证功能
"""

import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pydantic import BaseModel, Field, validator
from enum import Enum
import warnings


class ValidationError(Exception):
    """参数验证错误"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


class DataType(Enum):
    """支持的数据类型"""
    NUMERIC = "numeric"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    DATAFRAME = "dataframe"
    SERIES = "series"


class ValidationRule(BaseModel):
    """验证规则定义"""
    required: bool = Field(default=True, description="是否必需")
    data_type: DataType = Field(description="数据类型")
    min_value: Optional[float] = Field(default=None, description="最小值")
    max_value: Optional[float] = Field(default=None, description="最大值")
    min_length: Optional[int] = Field(default=None, description="最小长度")
    max_length: Optional[int] = Field(default=None, description="最大长度")
    pattern: Optional[str] = Field(default=None, description="正则表达式模式")
    allowed_values: Optional[List[Any]] = Field(default=None, description="允许的值")
    custom_validator: Optional[Callable] = Field(default=None, description="自定义验证函数")


class ParameterValidator:
    """
    参数验证器
    提供统一的参数验证和类型检查功能
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        初始化参数验证器
        
        Args:
            strict_mode: 严格模式，如果为True，验证失败时抛出异常
        """
        self.strict_mode = strict_mode
        self._validation_rules: Dict[str, ValidationRule] = {}
    
    def add_rule(self, field_name: str, rule: ValidationRule):
        """
        添加验证规则
        
        Args:
            field_name: 字段名称
            rule: 验证规则
        """
        self._validation_rules[field_name] = rule
    
    def validate_parameter(self, field_name: str, value: Any) -> Tuple[bool, Optional[str]]:
        """
        验证单个参数
        
        Args:
            field_name: 字段名称
            value: 参数值
            
        Returns:
            Tuple[bool, Optional[str]]: (是否验证通过, 错误信息)
        """
        if field_name not in self._validation_rules:
            return True, None
        
        rule = self._validation_rules[field_name]
        
        # 检查必需性
        if rule.required and value is None:
            return False, f"参数 '{field_name}' 是必需的"
        
        if value is None:
            return True, None
        
        # 检查数据类型
        type_valid, type_error = self._validate_data_type(value, rule.data_type)
        if not type_valid:
            return False, type_error
        
        # 检查数值范围
        if rule.min_value is not None or rule.max_value is not None:
            range_valid, range_error = self._validate_numeric_range(value, rule.min_value, rule.max_value)
            if not range_valid:
                return False, range_error
        
        # 检查长度范围
        if rule.min_length is not None or rule.max_length is not None:
            length_valid, length_error = self._validate_length(value, rule.min_length, rule.max_length)
            if not length_valid:
                return False, length_error
        
        # 检查正则表达式模式
        if rule.pattern is not None:
            pattern_valid, pattern_error = self._validate_pattern(value, rule.pattern)
            if not pattern_valid:
                return False, pattern_error
        
        # 检查允许的值
        if rule.allowed_values is not None:
            allowed_valid, allowed_error = self._validate_allowed_values(value, rule.allowed_values)
            if not allowed_valid:
                return False, allowed_error
        
        # 自定义验证
        if rule.custom_validator is not None:
            try:
                custom_valid = rule.custom_validator(value)
                if not custom_valid:
                    return False, f"参数 '{field_name}' 未通过自定义验证"
            except Exception as e:
                return False, f"参数 '{field_name}' 自定义验证失败: {str(e)}"
        
        return True, None
    
    def _validate_data_type(self, value: Any, data_type: DataType) -> Tuple[bool, Optional[str]]:
        """验证数据类型"""
        try:
            if data_type == DataType.NUMERIC:
                if not isinstance(value, (int, float, np.number)):
                    return False, f"期望数值类型，实际类型: {type(value).__name__}"
            
            elif data_type == DataType.INTEGER:
                if not isinstance(value, (int, np.integer)):
                    return False, f"期望整数类型，实际类型: {type(value).__name__}"
            
            elif data_type == DataType.FLOAT:
                if not isinstance(value, (float, np.floating)):
                    return False, f"期望浮点数类型，实际类型: {type(value).__name__}"
            
            elif data_type == DataType.STRING:
                if not isinstance(value, str):
                    return False, f"期望字符串类型，实际类型: {type(value).__name__}"
            
            elif data_type == DataType.BOOLEAN:
                if not isinstance(value, bool):
                    return False, f"期望布尔类型，实际类型: {type(value).__name__}"
            
            elif data_type == DataType.LIST:
                if not isinstance(value, (list, tuple, np.ndarray)):
                    return False, f"期望列表类型，实际类型: {type(value).__name__}"
            
            elif data_type == DataType.DICT:
                if not isinstance(value, dict):
                    return False, f"期望字典类型，实际类型: {type(value).__name__}"
            
            elif data_type == DataType.DATAFRAME:
                if not isinstance(value, pd.DataFrame):
                    return False, f"期望DataFrame类型，实际类型: {type(value).__name__}"
            
            elif data_type == DataType.SERIES:
                if not isinstance(value, pd.Series):
                    return False, f"期望Series类型，实际类型: {type(value).__name__}"
            
            return True, None
            
        except Exception as e:
            return False, f"数据类型验证失败: {str(e)}"
    
    def _validate_numeric_range(self, value: Any, min_val: Optional[float], max_val: Optional[float]) -> Tuple[bool, Optional[str]]:
        """验证数值范围"""
        try:
            numeric_value = float(value)
            
            if min_val is not None and numeric_value < min_val:
                return False, f"数值 {numeric_value} 小于最小值 {min_val}"
            
            if max_val is not None and numeric_value > max_val:
                return False, f"数值 {numeric_value} 大于最大值 {max_val}"
            
            return True, None
            
        except (ValueError, TypeError):
            return False, "无法转换为数值进行范围验证"
    
    def _validate_length(self, value: Any, min_len: Optional[int], max_len: Optional[int]) -> Tuple[bool, Optional[str]]:
        """验证长度范围"""
        try:
            length = len(value)
            
            if min_len is not None and length < min_len:
                return False, f"长度 {length} 小于最小长度 {min_len}"
            
            if max_len is not None and length > max_len:
                return False, f"长度 {length} 大于最大长度 {max_len}"
            
            return True, None
            
        except TypeError:
            return False, "无法获取长度信息"
    
    def _validate_pattern(self, value: Any, pattern: str) -> Tuple[bool, Optional[str]]:
        """验证正则表达式模式"""
        try:
            if not isinstance(value, str):
                return False, "模式验证仅适用于字符串类型"
            
            if not re.match(pattern, value):
                return False, f"字符串 '{value}' 不匹配模式 '{pattern}'"
            
            return True, None
            
        except re.error as e:
            return False, f"正则表达式模式错误: {str(e)}"
    
    def _validate_allowed_values(self, value: Any, allowed_values: List[Any]) -> Tuple[bool, Optional[str]]:
        """验证允许的值"""
        if value not in allowed_values:
            return False, f"值 '{value}' 不在允许的值列表中: {allowed_values}"
        
        return True, None
    
    def validate_all(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证所有参数
        
        Args:
            parameters: 参数字典
            
        Returns:
            Dict[str, Any]: 验证后的参数（包含类型转换）
            
        Raises:
            ValidationError: 验证失败时抛出
        """
        validated_params = {}
        errors = []
        
        for field_name, rule in self._validation_rules.items():
            value = parameters.get(field_name)
            
            # 验证参数
            is_valid, error_msg = self.validate_parameter(field_name, value)
            
            if not is_valid:
                if self.strict_mode:
                    raise ValidationError(error_msg, field_name, value)
                else:
                    errors.append(f"{field_name}: {error_msg}")
                    continue
            
            # 类型转换
            if value is not None and rule.data_type:
                try:
                    converted_value = self._convert_type(value, rule.data_type)
                    validated_params[field_name] = converted_value
                except Exception as e:
                    error_msg = f"参数 '{field_name}' 类型转换失败: {str(e)}"
                    if self.strict_mode:
                        raise ValidationError(error_msg, field_name, value)
                    else:
                        errors.append(error_msg)
            else:
                validated_params[field_name] = value
        
        if errors and not self.strict_mode:
            warnings.warn(f"参数验证警告: {', '.join(errors)}")
        
        return validated_params
    
    def _convert_type(self, value: Any, data_type: DataType) -> Any:
        """类型转换"""
        if data_type == DataType.INTEGER:
            return int(value)
        elif data_type == DataType.FLOAT:
            return float(value)
        elif data_type == DataType.STRING:
            return str(value)
        elif data_type == DataType.BOOLEAN:
            return bool(value)
        elif data_type == DataType.LIST:
            if isinstance(value, (tuple, np.ndarray)):
                return list(value)
            return value
        else:
            return value


# 预定义的验证器实例
class EconometricValidator:
    """计量经济学专用验证器"""
    
    @staticmethod
    def create_data_validator() -> ParameterValidator:
        """创建数据验证器"""
        validator = ParameterValidator()
        
        # 数据验证规则
        validator.add_rule("data", ValidationRule(
            required=True,
            data_type=DataType.DICT,
            min_length=1,
            custom_validator=lambda x: all(isinstance(v, (list, np.ndarray)) for v in x.values())
        ))
        
        validator.add_rule("y_data", ValidationRule(
            required=True,
            data_type=DataType.LIST,
            min_length=10,
            custom_validator=lambda x: all(isinstance(v, (int, float)) for v in x)
        ))
        
        validator.add_rule("x_data", ValidationRule(
            required=True,
            data_type=DataType.LIST,
            min_length=1,
            custom_validator=lambda x: all(isinstance(row, (list, np.ndarray)) and len(row) > 0 for row in x)
        ))
        
        return validator
    
    @staticmethod
    def create_model_validator() -> ParameterValidator:
        """创建模型参数验证器"""
        validator = ParameterValidator()
        
        # 模型参数验证规则
        validator.add_rule("n_estimators", ValidationRule(
            required=False,
            data_type=DataType.INTEGER,
            min_value=1,
            max_value=10000
        ))
        
        validator.add_rule("max_depth", ValidationRule(
            required=False,
            data_type=DataType.INTEGER,
            min_value=1,
            max_value=100
        ))
        
        validator.add_rule("learning_rate", ValidationRule(
            required=False,
            data_type=DataType.FLOAT,
            min_value=0.001,
            max_value=1.0
        ))
        
        validator.add_rule("alpha", ValidationRule(
            required=False,
            data_type=DataType.FLOAT,
            min_value=0.0,
            max_value=100.0
        ))
        
        return validator


# 便捷验证函数
def validate_econometric_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证计量经济学数据
    
    Args:
        data: 输入数据，格式为 {变量名: [数值列表]}
        
    Returns:
        Dict[str, Any]: 验证后的数据
    """
    # 创建专门的数据验证器
    validator = ParameterValidator()
    
    # 添加数据验证规则
    validator.add_rule("data", ValidationRule(
        required=True,
        data_type=DataType.DICT,
        min_length=1,
        custom_validator=lambda x: all(isinstance(v, (list, np.ndarray)) for v in x.values())
    ))
    
    # 验证数据格式
    validated = validator.validate_all({"data": data})
    return validated["data"]


def validate_model_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证模型参数
    
    Args:
        parameters: 模型参数
        
    Returns:
        Dict[str, Any]: 验证后的参数
    """
    validator = EconometricValidator.create_model_validator()
    return validator.validate_all(parameters)


def validate_time_series_data(data: List[float], min_length: int = 10) -> List[float]:
    """
    验证时间序列数据
    
    Args:
        data: 时间序列数据
        min_length: 最小长度要求
        
    Returns:
        List[float]: 验证后的数据
        
    Raises:
        ValidationError: 验证失败
    """
    if not isinstance(data, (list, np.ndarray)):
        raise ValidationError("时间序列数据必须是列表或数组")
    
    if len(data) < min_length:
        raise ValidationError(f"时间序列数据长度不足，至少需要 {min_length} 个观测点")
    
    if not all(isinstance(x, (int, float)) for x in data):
        raise ValidationError("时间序列数据必须全部为数值")
    
    # 检查缺失值
    if any(pd.isna(x) for x in data):
        raise ValidationError("时间序列数据包含缺失值")
    
    return list(data)

def validate_numeric_data(data: List[float], data_name: str = "数据") -> bool:
    """
    验证数值数据
    
    Args:
        data: 数值数据列表
        data_name: 数据名称，用于错误信息
        
    Returns:
        bool: 验证是否通过
        
    Raises:
        ValidationError: 验证失败时抛出
    """
    if not isinstance(data, (list, np.ndarray)):
        raise ValidationError(f"{data_name} 必须是列表或数组")
    
    if len(data) == 0:
        raise ValidationError(f"{data_name} 不能为空")
    
    if not all(isinstance(x, (int, float)) for x in data):
        raise ValidationError(f"{data_name} 必须全部为数值")
    
    # 检查缺失值
    if any(pd.isna(x) for x in data):
        raise ValidationError(f"{data_name} 包含缺失值")
    
    return True

# 导出主要类和函数
__all__ = [
    "ValidationError",
    "DataType",
    "ValidationRule", 
    "ParameterValidator",
    "EconometricValidator",
    "validate_econometric_data",
    "validate_model_parameters",
    "validate_time_series_data",
    "validate_numeric_data"
]