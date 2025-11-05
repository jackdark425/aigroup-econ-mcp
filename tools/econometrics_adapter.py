"""
计量经济学核心算法适配器
复用 econometrics/ 中的核心实现，避免代码重复
"""

from typing import List, Optional, Union
import sys
from pathlib import Path
import json

# 确保可以导入econometrics模块
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入核心算法实现
from econometrics.basic_parametric_estimation.ols.ols_model import (
    ols_regression as core_ols_regression,
    OLSResult as CoreOLSResult
)
from econometrics.basic_parametric_estimation.mle.mle_model import (
    mle_estimation as core_mle_estimation,
    MLEResult as CoreMLEResult
)
from econometrics.basic_parametric_estimation.gmm.gmm_model import (
    gmm_estimation as core_gmm_estimation,
    GMMResult as CoreGMMResult
)

# 导入数据加载和格式化组件
from .data_loader import DataLoader, MLEDataLoader
from .output_formatter import OutputFormatter


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_ols_gmm_data(y_data: List[float], x_data: List[List[float]], feature_names: Optional[List[str]] = None):
        """验证OLS和GMM数据格式"""
        if len(y_data) != len(x_data):
            raise ValueError(f"因变量长度({len(y_data)})与自变量长度({len(x_data)})不一致")
        
        # 检查所有x_data行的长度是否一致
        if x_data:
            first_row_len = len(x_data[0])
            for i, row in enumerate(x_data):
                if len(row) != first_row_len:
                    raise ValueError(f"自变量第{i}行长度({len(row)})与第一行长度({first_row_len})不一致")
        
        # 验证feature_names
        if feature_names and len(feature_names) != len(x_data[0]) if x_data else 0:
            raise ValueError(f"特征名称数量({len(feature_names)})与自变量列数({len(x_data[0]) if x_data else 0})不一致")
    
    @staticmethod
    def convert_to_2d_list(data: Union[List[float], List[List[float]]]) -> List[List[float]]:
        """将数据转换为二维列表格式"""
        if not data:
            return []
        
        # 如果是一维列表，转换为二维列表
        if isinstance(data[0], (int, float)):
            return [[x] for x in data]
        
        # 已经是二维列表
        return data


class EconometricsAdapter:
    """
    计量经济学适配器
    将core算法适配为MCP工具，支持文件输入和多种输出格式
    """
    
    @staticmethod
    def ols_regression(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        constant: bool = True,
        confidence_level: float = 0.95,
        output_format: str = "json",
        save_path: Optional[str] = None
    ) -> str:
        """
        OLS回归适配器
        
        优势：复用econometrics/核心算法，避免代码重复
        """
        # 1. 数据准备
        if file_path:
            data = DataLoader.load_from_file(file_path)
            y_data = data["y_data"]
            x_data = data["x_data"]
            feature_names = data.get("feature_names") or feature_names
        elif y_data is None or x_data is None:
            raise ValueError("必须提供文件路径(file_path)或直接数据(y_data和x_data)")
        
        # 数据验证和转换
        x_data = DataValidator.convert_to_2d_list(x_data)
        DataValidator.validate_ols_gmm_data(y_data, x_data, feature_names)
        
        # 2. 调用核心算法（复用！）
        result: CoreOLSResult = core_ols_regression(
            y_data=y_data,
            x_data=x_data,
            feature_names=feature_names,
            constant=constant,
            confidence_level=confidence_level
        )
        
        # 3. 格式化输出
        if output_format == "json":
            json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
            if save_path:
                OutputFormatter.save_to_file(json_result, save_path)
                return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
            return json_result
        else:
            # 尝试使用格式化器，失败则回退到JSON
            try:
                formatted = OutputFormatter.format_ols_result(result, output_format)
                if save_path:
                    OutputFormatter.save_to_file(formatted, save_path)
                    return f"分析完成！\n\n{formatted}\n\n已保存到: {save_path}"
                return formatted
            except Exception as e:
                # 回退到JSON格式
                json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
                warning = f"警告: {output_format}格式化失败({str(e)})，返回JSON格式\n\n"
                if save_path:
                    OutputFormatter.save_to_file(json_result, save_path)
                    return f"{warning}分析完成！结果已保存到: {save_path}\n\n{json_result}"
                return warning + json_result
    
    @staticmethod
    def mle_estimation(
        data: Optional[List[float]] = None,
        file_path: Optional[str] = None,
        distribution: str = "normal",
        initial_params: Optional[List[float]] = None,
        confidence_level: float = 0.95,
        output_format: str = "json",
        save_path: Optional[str] = None
    ) -> str:
        """
        MLE估计适配器
        
        优势：复用econometrics/核心算法
        """
        # 1. 数据准备
        if file_path:
            data_dict = MLEDataLoader.load_from_file(file_path)
            data = data_dict["data"]
        elif data is None:
            raise ValueError("必须提供文件路径(file_path)或直接数据(data)")
        
        # 2. 调用核心算法（复用！）
        result: CoreMLEResult = core_mle_estimation(
            data=data,
            distribution=distribution,
            initial_params=initial_params,
            confidence_level=confidence_level
        )
        
        # 3. 格式化输出
        if output_format == "json":
            json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
            if save_path:
                OutputFormatter.save_to_file(json_result, save_path)
                return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
            return json_result
        else:
            # 尝试使用格式化器，失败则回退到JSON
            try:
                formatted = OutputFormatter.format_mle_result(result, output_format)
                if save_path:
                    OutputFormatter.save_to_file(formatted, save_path)
                    return f"分析完成！\n\n{formatted}\n\n已保存到: {save_path}"
                return formatted
            except Exception as e:
                # 回退到JSON格式
                json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
                warning = f"警告: {output_format}格式化失败({str(e)})，返回JSON格式\n\n"
                if save_path:
                    OutputFormatter.save_to_file(json_result, save_path)
                    return f"{warning}分析完成！结果已保存到: {save_path}\n\n{json_result}"
                return warning + json_result
    
    @staticmethod
    def gmm_estimation(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        instruments: Optional[List[List[float]]] = None,
        feature_names: Optional[List[str]] = None,
        constant: bool = True,
        confidence_level: float = 0.95,
        output_format: str = "json",
        save_path: Optional[str] = None
    ) -> str:
        """
        GMM估计适配器
        
        优势：复用econometrics/核心算法
        增强：添加数值稳定性检查
        """
        # 1. 数据准备
        if file_path:
            data = DataLoader.load_from_file(file_path)
            y_data = data["y_data"]
            x_data = data["x_data"]
            feature_names = data.get("feature_names") or feature_names
        elif y_data is None or x_data is None:
            raise ValueError("必须提供文件路径(file_path)或直接数据(y_data和x_data)")
        
        # 数据验证和转换
        x_data = DataValidator.convert_to_2d_list(x_data)
        DataValidator.validate_ols_gmm_data(y_data, x_data, feature_names)
        
        # 转换工具变量格式
        if instruments:
            instruments = DataValidator.convert_to_2d_list(instruments)
        
        # 2. 调用核心算法（复用！）
        try:
            result: CoreGMMResult = core_gmm_estimation(
                y_data=y_data,
                x_data=x_data,
                instruments=instruments,
                feature_names=feature_names,
                constant=constant,
                confidence_level=confidence_level
            )
        except Exception as e:
            # 提供更详细的错误信息
            error_msg = f"GMM估计失败: {str(e)}\n"
            error_msg += "可能原因:\n"
            error_msg += "1. 数据存在多重共线性\n"
            error_msg += "2. 工具变量不足或无效\n"
            error_msg += "3. 矩阵奇异（数值不稳定）\n"
            error_msg += "建议:\n"
            error_msg += "- 检查数据质量\n"
            error_msg += "- 增加工具变量数量\n"
            error_msg += "- 尝试标准化数据\n"
            raise ValueError(error_msg) from e
        
        # 3. 格式化输出
        if output_format == "json":
            json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
            if save_path:
                OutputFormatter.save_to_file(json_result, save_path)
                return f"分析完成！结果已保存到: {save_path}\n\n{json_result}"
            return json_result
        else:
            # 尝试使用格式化器，失败则回退到JSON
            try:
                formatted = OutputFormatter.format_gmm_result(result, output_format)
                if save_path:
                    OutputFormatter.save_to_file(formatted, save_path)
                    return f"分析完成！\n\n{formatted}\n\n已保存到: {save_path}"
                return formatted
            except Exception as e:
                # 回退到JSON格式
                json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
                warning = f"警告: {output_format}格式化失败({str(e)})，返回JSON格式\n\n"
                if save_path:
                    OutputFormatter.save_to_file(json_result, save_path)
                    return f"{warning}分析完成！结果已保存到: {save_path}\n\n{json_result}"
                return warning + json_result


# 便捷别名
ols_adapter = EconometricsAdapter.ols_regression
mle_adapter = EconometricsAdapter.mle_estimation

# 导入模型规范、诊断和稳健推断适配器
from .model_specification_adapter import (
    diagnostic_tests_adapter,
    gls_adapter,
    wls_adapter,
    robust_errors_adapter,
    model_selection_adapter,
    regularization_adapter,
    simultaneous_equations_adapter
)
gmm_adapter = EconometricsAdapter.gmm_estimation