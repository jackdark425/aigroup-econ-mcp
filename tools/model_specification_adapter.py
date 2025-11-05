"""
模型规范、诊断和稳健推断适配器
将econometrics/model_specification_diagnostics_robust_inference中的模型适配为MCP工具
"""

from typing import List, Optional, Union, Dict, Any
import sys
from pathlib import Path
import json

# 确保可以导入econometrics模块
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入模型规范、诊断和稳健推断模型
from econometrics.model_specification_diagnostics_robust_inference.diagnostic_tests.diagnostic_tests_model import (
    diagnostic_tests as core_diagnostic_tests,
    DiagnosticTestsResult as CoreDiagnosticTestsResult
)

from econometrics.model_specification_diagnostics_robust_inference.generalized_least_squares.gls_model import (
    gls_regression as core_gls_regression,
    GLSResult as CoreGLSResult
)

from econometrics.model_specification_diagnostics_robust_inference.weighted_least_squares.wls_model import (
    wls_regression as core_wls_regression,
    WLSResult as CoreWLSResult
)

from econometrics.model_specification_diagnostics_robust_inference.robust_errors.robust_errors_model import (
    robust_errors_regression as core_robust_errors_regression,
    RobustErrorsResult as CoreRobustErrorsResult
)

from econometrics.model_specification_diagnostics_robust_inference.model_selection.model_selection_model import (
    model_selection_criteria as core_model_selection_criteria,
    granger_causality_test as core_granger_causality_test,
    ModelSelectionResult as CoreModelSelectionResult,
    GrangerCausalityResult as CoreGrangerCausalityResult
)

from econometrics.model_specification_diagnostics_robust_inference.regularization.regularization_model import (
    regularized_regression as core_regularized_regression,
    RegularizationResult as CoreRegularizationResult
)

from econometrics.model_specification_diagnostics_robust_inference.simultaneous_equations.simultaneous_equations_model import (
    two_stage_least_squares as core_two_stage_least_squares,
    SimultaneousEquationsResult as CoreSimultaneousEquationsResult
)

# 导入数据加载和格式化组件
from .data_loader import DataLoader
from .output_formatter import OutputFormatter


class ModelSpecificationAdapter:
    """
    模型规范、诊断和稳健推断适配器
    将core算法适配为MCP工具，支持文件输入和多种输出格式
    """
    
    @staticmethod
    def diagnostic_tests(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        constant: bool = True,
        output_format: str = "json",
        save_path: Optional[str] = None
    ) -> str:
        """
        模型诊断检验适配器
        
        Args:
            y_data: 因变量数据
            x_data: 自变量数据
            file_path: 数据文件路径
            feature_names: 特征名称
            constant: 是否包含常数项
            output_format: 输出格式 ("json", "markdown", "html")
            save_path: 保存路径
            
        Returns:
            str: 格式化的分析结果
        """
        # 1. 数据准备
        if file_path:
            data = DataLoader.load_from_file(file_path)
            y_data = data["y_data"]
            x_data = data["x_data"]
            feature_names = data.get("feature_names") or feature_names
        elif y_data is None or x_data is None:
            raise ValueError("必须提供文件路径(file_path)或直接数据(y_data和x_data)")
        
        # 2. 调用核心算法
        result: CoreDiagnosticTestsResult = core_diagnostic_tests(
            y_data=y_data,
            x_data=x_data,
            feature_names=feature_names,
            constant=constant
        )
        
        # 3. 格式化输出
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"模型诊断检验完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    
    @staticmethod
    def gls_regression(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        sigma: Optional[List[List[float]]] = None,
        feature_names: Optional[List[str]] = None,
        constant: bool = True,
        confidence_level: float = 0.95,
        output_format: str = "json",
        save_path: Optional[str] = None
    ) -> str:
        """
        GLS回归适配器
        """
        # 1. 数据准备
        if file_path:
            data = DataLoader.load_from_file(file_path)
            y_data = data["y_data"]
            x_data = data["x_data"]
            feature_names = data.get("feature_names") or feature_names
        elif y_data is None or x_data is None:
            raise ValueError("必须提供文件路径(file_path)或直接数据(y_data和x_data)")
        
        # 2. 调用核心算法
        result: CoreGLSResult = core_gls_regression(
            y_data=y_data,
            x_data=x_data,
            sigma=sigma,
            feature_names=feature_names,
            constant=constant,
            confidence_level=confidence_level
        )
        
        # 3. 格式化输出
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"GLS回归完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    
    @staticmethod
    def wls_regression(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        weights: Optional[List[float]] = None,
        feature_names: Optional[List[str]] = None,
        constant: bool = True,
        confidence_level: float = 0.95,
        output_format: str = "json",
        save_path: Optional[str] = None
    ) -> str:
        """
        WLS回归适配器
        """
        # 1. 数据准备
        if file_path:
            data = DataLoader.load_from_file(file_path)
            y_data = data["y_data"]
            x_data = data["x_data"]
            weights = data.get("weights") or weights
            feature_names = data.get("feature_names") or feature_names
        elif y_data is None or x_data is None or weights is None:
            raise ValueError("必须提供文件路径(file_path)或直接数据(y_data、x_data和weights)")
        
        # 2. 调用核心算法
        result: CoreWLSResult = core_wls_regression(
            y_data=y_data,
            x_data=x_data,
            weights=weights,
            feature_names=feature_names,
            constant=constant,
            confidence_level=confidence_level
        )
        
        # 3. 格式化输出
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"WLS回归完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    
    @staticmethod
    def robust_errors_regression(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        constant: bool = True,
        confidence_level: float = 0.95,
        cov_type: str = "HC1",
        output_format: str = "json",
        save_path: Optional[str] = None
    ) -> str:
        """
        稳健标准误回归适配器
        """
        # 1. 数据准备
        if file_path:
            data = DataLoader.load_from_file(file_path)
            y_data = data["y_data"]
            x_data = data["x_data"]
            feature_names = data.get("feature_names") or feature_names
        elif y_data is None or x_data is None:
            raise ValueError("必须提供文件路径(file_path)或直接数据(y_data和x_data)")
        
        # 2. 调用核心算法
        result: CoreRobustErrorsResult = core_robust_errors_regression(
            y_data=y_data,
            x_data=x_data,
            feature_names=feature_names,
            constant=constant,
            confidence_level=confidence_level,
            cov_type=cov_type
        )
        
        # 3. 格式化输出
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"稳健标准误回归完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    
    @staticmethod
    def model_selection_criteria(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        constant: bool = True,
        cv_folds: Optional[int] = None,
        output_format: str = "json",
        save_path: Optional[str] = None
    ) -> str:
        """
        模型选择准则适配器
        """
        # 1. 数据准备
        if file_path:
            data = DataLoader.load_from_file(file_path)
            y_data = data["y_data"]
            x_data = data["x_data"]
            feature_names = data.get("feature_names") or feature_names
        elif y_data is None or x_data is None:
            raise ValueError("必须提供文件路径(file_path)或直接数据(y_data和x_data)")
        
        # 2. 调用核心算法
        result: CoreModelSelectionResult = core_model_selection_criteria(
            y_data=y_data,
            x_data=x_data,
            feature_names=feature_names,
            constant=constant,
            cv_folds=cv_folds
        )
        
        # 3. 格式化输出
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"模型选择分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    
    @staticmethod
    def regularized_regression(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        method: str = "ridge",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        feature_names: Optional[List[str]] = None,
        fit_intercept: bool = True,
        output_format: str = "json",
        save_path: Optional[str] = None
    ) -> str:
        """
        正则化回归适配器
        """
        # 1. 数据准备
        if file_path:
            data = DataLoader.load_from_file(file_path)
            y_data = data["y_data"]
            x_data = data["x_data"]
            feature_names = data.get("feature_names") or feature_names
        elif y_data is None or x_data is None:
            raise ValueError("必须提供文件路径(file_path)或直接数据(y_data和x_data)")
        
        # 2. 调用核心算法
        result: CoreRegularizationResult = core_regularized_regression(
            y_data=y_data,
            x_data=x_data,
            method=method,
            alpha=alpha,
            l1_ratio=l1_ratio,
            feature_names=feature_names,
            fit_intercept=fit_intercept
        )
        
        # 3. 格式化输出
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"正则化回归({method})完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result
    
    @staticmethod
    def simultaneous_equations(
        y_data: Optional[List[List[float]]] = None,
        x_data: Optional[List[List[List[float]]]] = None,
        file_path: Optional[str] = None,
        instruments: Optional[List[List[float]]] = None,
        equation_names: Optional[List[str]] = None,
        endogenous_vars: Optional[List[str]] = None,
        exogenous_vars: Optional[List[str]] = None,
        constant: bool = True,
        output_format: str = "json",
        save_path: Optional[str] = None
    ) -> str:
        """
        联立方程模型适配器
        """
        # 1. 数据准备
        if file_path:
            data = DataLoader.load_from_file(file_path)
            y_data = data.get("y_data") or y_data
            x_data = data.get("x_data") or x_data
            instruments = data.get("instruments") or instruments
            equation_names = data.get("equation_names") or equation_names
        elif y_data is None or x_data is None or instruments is None:
            raise ValueError("必须提供文件路径(file_path)或直接数据(y_data、x_data和instruments)")
        
        # 2. 调用核心算法
        result: CoreSimultaneousEquationsResult = core_two_stage_least_squares(
            y_data=y_data,
            x_data=x_data,
            instruments=instruments,
            equation_names=equation_names,
            endogenous_vars=endogenous_vars,
            exogenous_vars=exogenous_vars,
            constant=constant
        )
        
        # 3. 格式化输出
        json_result = json.dumps(result.dict(), ensure_ascii=False, indent=2)
        if save_path:
            OutputFormatter.save_to_file(json_result, save_path)
            return f"联立方程模型(2SLS)分析完成！结果已保存到: {save_path}\n\n{json_result}"
        return json_result


# 便捷别名
diagnostic_tests_adapter = ModelSpecificationAdapter.diagnostic_tests
gls_adapter = ModelSpecificationAdapter.gls_regression
wls_adapter = ModelSpecificationAdapter.wls_regression
robust_errors_adapter = ModelSpecificationAdapter.robust_errors_regression
model_selection_adapter = ModelSpecificationAdapter.model_selection_criteria
regularization_adapter = ModelSpecificationAdapter.regularized_regression
simultaneous_equations_adapter = ModelSpecificationAdapter.simultaneous_equations