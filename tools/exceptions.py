"""
异常类模块
提供计量经济学工具专用的异常类
"""


class EconometricToolError(Exception):
    """计量经济学工具异常基类"""
    
    def __init__(self, message: str, tool_name: str = None, original_error: Exception = None):
        self.message = message
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(self.message)
    
    def __str__(self):
        base_msg = f"计量经济学工具错误"
        if self.tool_name:
            base_msg += f" ({self.tool_name})"
        base_msg += f": {self.message}"
        if self.original_error:
            base_msg += f"\n原始错误: {self.original_error}"
        return base_msg


class DataValidationError(EconometricToolError):
    """数据验证错误"""
    pass


class ModelFittingError(EconometricToolError):
    """模型拟合错误"""
    pass


class ConfigurationError(EconometricToolError):
    """配置错误"""
    pass


# 导出
__all__ = [
    "EconometricToolError",
    "DataValidationError", 
    "ModelFittingError",
    "ConfigurationError"
]