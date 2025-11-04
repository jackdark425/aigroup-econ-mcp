"""
计量经济学核心算法适配器
复用 econometrics/ 中的核心实现，避免代码重复
"""

from typing import List, Optional, Union
import sys
from pathlib import Path

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
            raise ValueError("Must provide either file_path or (y_data and x_data)")
        
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
            import json
            return json.dumps(result.dict(), ensure_ascii=False, indent=2)
        else:
            formatted = OutputFormatter.format_ols_result(result, output_format)
            if save_path:
                OutputFormatter.save_to_file(formatted, save_path)
                return f"Analysis complete!\n\n{formatted}\n\nSaved to: {save_path}"
            return formatted
    
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
            raise ValueError("Must provide either file_path or data")
        
        # 2. 调用核心算法（复用！）
        result: CoreMLEResult = core_mle_estimation(
            data=data,
            distribution=distribution,
            initial_params=initial_params,
            confidence_level=confidence_level
        )
        
        # 3. 格式化输出
        if output_format == "json":
            import json
            return json.dumps(result.dict(), ensure_ascii=False, indent=2)
        else:
            formatted = OutputFormatter.format_mle_result(result, output_format)
            if save_path:
                OutputFormatter.save_to_file(formatted, save_path)
                return f"Analysis complete!\n\n{formatted}\n\nSaved to: {save_path}"
            return formatted
    
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
        注意：需要修复core代码的j_p_value bug
        """
        # 1. 数据准备
        if file_path:
            data = DataLoader.load_from_file(file_path)
            y_data = data["y_data"]
            x_data = data["x_data"]
            feature_names = data.get("feature_names") or feature_names
        elif y_data is None or x_data is None:
            raise ValueError("Must provide either file_path or (y_data and x_data)")
        
        # 2. 调用核心算法（复用！）
        result: CoreGMMResult = core_gmm_estimation(
            y_data=y_data,
            x_data=x_data,
            instruments=instruments,
            feature_names=feature_names,
            constant=constant,
            confidence_level=confidence_level
        )
        
        # 3. 修复j_p_value bug（如果core还没修复）
        # TODO: 应该在core代码中修复这个问题
        
        # 4. 格式化输出
        if output_format == "json":
            import json
            return json.dumps(result.dict(), ensure_ascii=False, indent=2)
        else:
            formatted = OutputFormatter.format_gmm_result(result, output_format)
            if save_path:
                OutputFormatter.save_to_file(formatted, save_path)
                return f"Analysis complete!\n\n{formatted}\n\nSaved to: {save_path}"
            return formatted


# 便捷别名
ols_adapter = EconometricsAdapter.ols_regression
mle_adapter = EconometricsAdapter.mle_estimation
gmm_adapter = EconometricsAdapter.gmm_estimation