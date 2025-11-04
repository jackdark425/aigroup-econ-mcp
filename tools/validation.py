"""


"""

import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pydantic import BaseModel, Field, validator
from enum import Enum
import warnings


class ValidationError(Exception):
    """"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


class DataType(Enum):
    """"""
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
    """"""
    required: bool = Field(default=True, description="")
    data_type: DataType = Field(description="")
    min_value: Optional[float] = Field(default=None, description="")
    max_value: Optional[float] = Field(default=None, description="")
    min_length: Optional[int] = Field(default=None, description="")
    max_length: Optional[int] = Field(default=None, description="")
    pattern: Optional[str] = Field(default=None, description="")
    allowed_values: Optional[List[Any]] = Field(default=None, description="")
    custom_validator: Optional[Callable] = Field(default=None, description="")


class ParameterValidator:
    """
    
    
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        
        
        Args:
            strict_mode: True
        """
        self.strict_mode = strict_mode
        self._validation_rules: Dict[str, ValidationRule] = {}
    
    def add_rule(self, field_name: str, rule: ValidationRule):
        """
        
        
        Args:
            field_name: 
            rule: 
        """
        self._validation_rules[field_name] = rule
    
    def validate_parameter(self, field_name: str, value: Any) -> Tuple[bool, Optional[str]]:
        """
        
        
        Args:
            field_name: 
            value: 
            
        Returns:
            Tuple[bool, Optional[str]]: (, )
        """
        if field_name not in self._validation_rules:
            return True, None
        
        rule = self._validation_rules[field_name]
        
        # 
        if rule.required and value is None:
            return False, f" '{field_name}' "
        
        if value is None:
            return True, None
        
        # 
        type_valid, type_error = self._validate_data_type(value, rule.data_type)
        if not type_valid:
            return False, type_error
        
        # 
        if rule.min_value is not None or rule.max_value is not None:
            range_valid, range_error = self._validate_numeric_range(value, rule.min_value, rule.max_value)
            if not range_valid:
                return False, range_error
        
        # 
        if rule.min_length is not None or rule.max_length is not None:
            length_valid, length_error = self._validate_length(value, rule.min_length, rule.max_length)
            if not length_valid:
                return False, length_error
        
        # 
        if rule.pattern is not None:
            pattern_valid, pattern_error = self._validate_pattern(value, rule.pattern)
            if not pattern_valid:
                return False, pattern_error
        
        # 
        if rule.allowed_values is not None:
            allowed_valid, allowed_error = self._validate_allowed_values(value, rule.allowed_values)
            if not allowed_valid:
                return False, allowed_error
        
        # 
        if rule.custom_validator is not None:
            try:
                custom_valid = rule.custom_validator(value)
                if not custom_valid:
                    return False, f" '{field_name}' "
            except Exception as e:
                return False, f" '{field_name}' : {str(e)}"
        
        return True, None
    
    def _validate_data_type(self, value: Any, data_type: DataType) -> Tuple[bool, Optional[str]]:
        """"""
        try:
            if data_type == DataType.NUMERIC:
                if not isinstance(value, (int, float, np.number)):
                    return False, f": {type(value).__name__}"
            
            elif data_type == DataType.INTEGER:
                if not isinstance(value, (int, np.integer)):
                    return False, f": {type(value).__name__}"
            
            elif data_type == DataType.FLOAT:
                if not isinstance(value, (float, np.floating)):
                    return False, f": {type(value).__name__}"
            
            elif data_type == DataType.STRING:
                if not isinstance(value, str):
                    return False, f": {type(value).__name__}"
            
            elif data_type == DataType.BOOLEAN:
                if not isinstance(value, bool):
                    return False, f": {type(value).__name__}"
            
            elif data_type == DataType.LIST:
                if not isinstance(value, (list, tuple, np.ndarray)):
                    return False, f": {type(value).__name__}"
            
            elif data_type == DataType.DICT:
                if not isinstance(value, dict):
                    return False, f": {type(value).__name__}"
            
            elif data_type == DataType.DATAFRAME:
                if not isinstance(value, pd.DataFrame):
                    return False, f"DataFrame: {type(value).__name__}"
            
            elif data_type == DataType.SERIES:
                if not isinstance(value, pd.Series):
                    return False, f"Series: {type(value).__name__}"
            
            return True, None
            
        except Exception as e:
            return False, f": {str(e)}"
    
    def _validate_numeric_range(self, value: Any, min_val: Optional[float], max_val: Optional[float]) -> Tuple[bool, Optional[str]]:
        """"""
        try:
            numeric_value = float(value)
            
            if min_val is not None and numeric_value < min_val:
                return False, f" {numeric_value}  {min_val}"
            
            if max_val is not None and numeric_value > max_val:
                return False, f" {numeric_value}  {max_val}"
            
            return True, None
            
        except (ValueError, TypeError):
            return False, ""
    
    def _validate_length(self, value: Any, min_len: Optional[int], max_len: Optional[int]) -> Tuple[bool, Optional[str]]:
        """"""
        try:
            length = len(value)
            
            if min_len is not None and length < min_len:
                return False, f" {length}  {min_len}"
            
            if max_len is not None and length > max_len:
                return False, f" {length}  {max_len}"
            
            return True, None
            
        except TypeError:
            return False, ""
    
    def _validate_pattern(self, value: Any, pattern: str) -> Tuple[bool, Optional[str]]:
        """"""
        try:
            if not isinstance(value, str):
                return False, ""
            
            if not re.match(pattern, value):
                return False, f" '{value}'  '{pattern}'"
            
            return True, None
            
        except re.error as e:
            return False, f": {str(e)}"
    
    def _validate_allowed_values(self, value: Any, allowed_values: List[Any]) -> Tuple[bool, Optional[str]]:
        """"""
        if value not in allowed_values:
            return False, f" '{value}' : {allowed_values}"
        
        return True, None
    
    def validate_all(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        
        
        Args:
            parameters: 
            
        Returns:
            Dict[str, Any]: 
            
        Raises:
            ValidationError: 
        """
        validated_params = {}
        errors = []
        
        for field_name, rule in self._validation_rules.items():
            value = parameters.get(field_name)
            
            # 
            is_valid, error_msg = self.validate_parameter(field_name, value)
            
            if not is_valid:
                if self.strict_mode:
                    raise ValidationError(error_msg, field_name, value)
                else:
                    errors.append(f"{field_name}: {error_msg}")
                    continue
            
            # 
            if value is not None and rule.data_type:
                try:
                    converted_value = self._convert_type(value, rule.data_type)
                    validated_params[field_name] = converted_value
                except Exception as e:
                    error_msg = f" '{field_name}' : {str(e)}"
                    if self.strict_mode:
                        raise ValidationError(error_msg, field_name, value)
                    else:
                        errors.append(error_msg)
            else:
                validated_params[field_name] = value
        
        if errors and not self.strict_mode:
            warnings.warn(f": {', '.join(errors)}")
        
        return validated_params
    
    def _convert_type(self, value: Any, data_type: DataType) -> Any:
        """"""
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


# 
class EconometricValidator:
    """"""
    
    @staticmethod
    def create_data_validator() -> ParameterValidator:
        """"""
        validator = ParameterValidator()
        
        # 
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
        """"""
        validator = ParameterValidator()
        
        # 
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


# 
def validate_econometric_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    
    
    Args:
        data:  {: []}
        
    Returns:
        Dict[str, Any]: 
    """
    # 
    validator = ParameterValidator()
    
    # 
    validator.add_rule("data", ValidationRule(
        required=True,
        data_type=DataType.DICT,
        min_length=1,
        custom_validator=lambda x: all(isinstance(v, (list, np.ndarray)) for v in x.values())
    ))
    
    # 
    validated = validator.validate_all({"data": data})
    return validated["data"]


def validate_model_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    
    
    Args:
        parameters: 
        
    Returns:
        Dict[str, Any]: 
    """
    validator = EconometricValidator.create_model_validator()
    return validator.validate_all(parameters)


def validate_time_series_data(data: List[float], min_length: int = 10) -> List[float]:
    """
    
    
    Args:
        data: 
        min_length: 
        
    Returns:
        List[float]: 
        
    Raises:
        ValidationError: 
    """
    if not isinstance(data, (list, np.ndarray)):
        raise ValidationError("")
    
    if len(data) < min_length:
        raise ValidationError(f" {min_length} ")
    
    if not all(isinstance(x, (int, float)) for x in data):
        raise ValidationError("")
    
    # 
    if any(pd.isna(x) for x in data):
        raise ValidationError("")
    
    return list(data)

def validate_numeric_data(data: List[float], data_name: str = "") -> bool:
    """
    
    
    Args:
        data: 
        data_name: 
        
    Returns:
        bool: 
        
    Raises:
        ValidationError: 
    """
    if not isinstance(data, (list, np.ndarray)):
        raise ValidationError(f"{data_name} ")
    
    if len(data) == 0:
        raise ValidationError(f"{data_name} ")
    
    if not all(isinstance(x, (int, float)) for x in data):
        raise ValidationError(f"{data_name} ")
    
    # 
    if any(pd.isna(x) for x in data):
        raise ValidationError(f"{data_name} ")
    
    return True

# 
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