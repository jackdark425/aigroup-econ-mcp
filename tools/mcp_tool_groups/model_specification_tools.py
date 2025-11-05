"""
模型规范、诊断和稳健推断工具组
包含诊断检验、GLS、WLS、稳健标准误、模型选择、正则化、联立方程等方法
"""

from typing import List, Optional, Union, Dict, Any
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from ..mcp_tools_registry import ToolGroup
from ..econometrics_adapter import (
    diagnostic_tests_adapter,
    gls_adapter,
    wls_adapter,
    robust_errors_adapter,
    model_selection_adapter,
    regularization_adapter,
    simultaneous_equations_adapter
)


class ModelSpecificationTools(ToolGroup):
    """模型规范、诊断和稳健推断工具组"""
    
    name = "MODEL SPECIFICATION, DIAGNOSTICS & ROBUST INFERENCE"
    description = "模型规范检验、诊断测试和稳健推断方法工具"
    version = "2.0.0"
    
    @classmethod
    def get_tools(cls) -> List[Dict[str, Any]]:
        """返回工具列表"""
        return [
            {
                "name": "model_diagnostic_tests",
                "handler": cls.diagnostic_tests_tool,
                "description": "Model Diagnostic Tests (Heteroskedasticity, Autocorrelation, Normality, VIF)"
            },
            {
                "name": "generalized_least_squares",
                "handler": cls.gls_tool,
                "description": "Generalized Least Squares (GLS) Regression"
            },
            {
                "name": "weighted_least_squares",
                "handler": cls.wls_tool,
                "description": "Weighted Least Squares (WLS) Regression"
            },
            {
                "name": "robust_errors_regression",
                "handler": cls.robust_errors_tool,
                "description": "Robust Standard Errors Regression (Heteroskedasticity-Robust)"
            },
            {
                "name": "model_selection_criteria",
                "handler": cls.model_selection_tool,
                "description": "Model Selection Criteria (AIC, BIC, HQIC, Cross-Validation)"
            },
            {
                "name": "regularized_regression",
                "handler": cls.regularization_tool,
                "description": "Regularized Regression (Ridge, LASSO, Elastic Net)"
            },
            {
                "name": "simultaneous_equations_model",
                "handler": cls.simultaneous_equations_tool,
                "description": "Simultaneous Equations Model (2SLS)"
            }
        ]
    
    @classmethod
    def get_help_text(cls) -> str:
        """返回帮助文档"""
        return """
15. Model Diagnostic Tests (model_diagnostic_tests)
    - Heteroskedasticity: Breusch-Pagan, White tests
    - Autocorrelation: Durbin-Watson test
    - Normality: Jarque-Bera test
    - Multicollinearity: VIF calculation
    - Comprehensive model diagnostics

16. Generalized Least Squares (generalized_least_squares)
    - GLS: Handle heteroskedasticity and autocorrelation
    - Covariance matrix: User-specified or estimated
    - Efficient: More efficient than OLS under GLS assumptions

17. Weighted Least Squares (weighted_least_squares)
    - WLS: Handle heteroskedasticity with known weights
    - Weights: Inverse of variance or other weighting schemes
    - Applications: Survey data, grouped data

18. Robust Standard Errors (robust_errors_regression)
    - Heteroskedasticity-robust: HC0, HC1, HC2, HC3
    - Inference: Valid inference under heteroskedasticity
    - Applications: Cross-sectional data with heteroskedasticity

19. Model Selection Criteria (model_selection_criteria)
    - Information criteria: AIC, BIC, HQIC
    - Cross-validation: K-fold, leave-one-out
    - Granger causality: Test for causality in time series
    - Model comparison: Select best model specification

20. Regularized Regression (regularized_regression)
    - Ridge regression: L2 penalty for multicollinearity
    - LASSO: L1 penalty for variable selection
    - Elastic Net: Combined L1 and L2 penalties
    - Applications: High-dimensional data, feature selection

21. Simultaneous Equations Model (simultaneous_equations_model)
    - 2SLS: Two-stage least squares for simultaneous equations
    - Endogeneity: Handle endogenous regressors
    - Instrumental variables: Valid instruments required
    - Applications: Supply-demand models, macroeconomic models
"""

    @staticmethod
    async def diagnostic_tests_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[Union[List[float], List[List[float]]]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        constant: bool = True,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Model Diagnostic Tests"""
        try:
            if ctx:
                await ctx.info("Starting model diagnostic tests...")
            
            result = diagnostic_tests_adapter(
                y_data=y_data,
                x_data=x_data,
                file_path=file_path,
                feature_names=feature_names,
                constant=constant,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Model diagnostic tests complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def gls_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[Union[List[float], List[List[float]]]] = None,
        file_path: Optional[str] = None,
        sigma: Optional[List[List[float]]] = None,
        feature_names: Optional[List[str]] = None,
        constant: bool = True,
        confidence_level: float = 0.95,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Generalized Least Squares Regression"""
        try:
            if ctx:
                await ctx.info("Starting GLS regression...")
            
            result = gls_adapter(
                y_data=y_data,
                x_data=x_data,
                file_path=file_path,
                sigma=sigma,
                feature_names=feature_names,
                constant=constant,
                confidence_level=confidence_level,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("GLS regression complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def wls_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[Union[List[float], List[List[float]]]] = None,
        file_path: Optional[str] = None,
        weights: Optional[List[float]] = None,
        feature_names: Optional[List[str]] = None,
        constant: bool = True,
        confidence_level: float = 0.95,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Weighted Least Squares Regression"""
        try:
            if ctx:
                await ctx.info("Starting WLS regression...")
            
            result = wls_adapter(
                y_data=y_data,
                x_data=x_data,
                file_path=file_path,
                weights=weights,
                feature_names=feature_names,
                constant=constant,
                confidence_level=confidence_level,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("WLS regression complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def robust_errors_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[Union[List[float], List[List[float]]]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        constant: bool = True,
        confidence_level: float = 0.95,
        cov_type: str = "HC1",
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Robust Standard Errors Regression"""
        try:
            if ctx:
                await ctx.info("Starting robust errors regression...")
            
            result = robust_errors_adapter(
                y_data=y_data,
                x_data=x_data,
                file_path=file_path,
                feature_names=feature_names,
                constant=constant,
                confidence_level=confidence_level,
                cov_type=cov_type,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Robust errors regression complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def model_selection_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[Union[List[float], List[List[float]]]] = None,
        file_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        constant: bool = True,
        cv_folds: Optional[int] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Model Selection Criteria"""
        try:
            if ctx:
                await ctx.info("Starting model selection analysis...")
            
            result = model_selection_adapter(
                y_data=y_data,
                x_data=x_data,
                file_path=file_path,
                feature_names=feature_names,
                constant=constant,
                cv_folds=cv_folds,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Model selection analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def regularization_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[Union[List[float], List[List[float]]]] = None,
        file_path: Optional[str] = None,
        method: str = "ridge",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        feature_names: Optional[List[str]] = None,
        fit_intercept: bool = True,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Regularized Regression"""
        try:
            if ctx:
                await ctx.info(f"Starting {method} regularized regression...")
            
            result = regularization_adapter(
                y_data=y_data,
                x_data=x_data,
                file_path=file_path,
                method=method,
                alpha=alpha,
                l1_ratio=l1_ratio,
                feature_names=feature_names,
                fit_intercept=fit_intercept,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info(f"{method} regularized regression complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise
    
    @staticmethod
    async def simultaneous_equations_tool(
        y_data: Optional[List[List[float]]] = None,
        x_data: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        instruments: Optional[List[List[float]]] = None,
        equation_names: Optional[List[str]] = None,
        instrument_names: Optional[List[str]] = None,
        constant: bool = True,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """
        Simultaneous Equations Model (2SLS)
        
        数据格式说明:
        - y_data: 因变量数据，二维列表，每个子列表代表一个方程的因变量时间序列
        - x_data: 自变量数据，二维列表，每个子列表代表一个观测的所有自变量值
        - instruments: 工具变量数据，二维列表，每个子列表代表一个观测的所有工具变量值
        
        重要: 所有数据的观测数量必须相同
        
        示例调用:
        {
          "y_data": [[1.0, 1.2, 1.4, 1.6], [2.0, 2.2, 2.4, 2.6]],
          "x_data": [[1.5, 2.5], [1.7, 2.7], [1.9, 2.9], [2.1, 3.1]],
          "instruments": [[1.8, 2.8], [2.0, 3.0], [2.2, 3.2], [2.4, 3.4]],
          "equation_names": ["Demand", "Supply"],
          "instrument_names": ["Income", "Price"],
          "constant": true,
          "output_format": "json"
        }
        """
        try:
            if ctx:
                await ctx.info("Starting simultaneous equations model analysis...")
            
            result = simultaneous_equations_adapter(
                y_data=y_data,
                x_data=x_data,
                file_path=file_path,
                instruments=instruments,
                equation_names=equation_names,
                instrument_names=instrument_names,
                constant=constant,
                output_format=output_format,
                save_path=save_path
            )
            
            if ctx:
                await ctx.info("Simultaneous equations model analysis complete")
            
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise