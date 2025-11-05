"""
时间序列和面板数据工具组
包含 ARIMA、GARCH、单位根检验、VAR/SVAR、协整分析、动态面板模型等
"""

from typing import List, Optional, Union, Dict, Any
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from ..mcp_tools_registry import ToolGroup
from ..time_series_panel_data_adapter import (
    arima_adapter,
    exp_smoothing_adapter,
    garch_adapter,
    unit_root_adapter,
    var_svar_adapter,
    cointegration_adapter,
    dynamic_panel_adapter,
    panel_diagnostics_adapter,
    panel_var_adapter,
    structural_break_adapter,
    time_varying_parameter_adapter
)


class TimeSeriesTools(ToolGroup):
    """时间序列和面板数据工具组"""
    
    name = "TIME SERIES & PANEL DATA"
    description = "时间序列分析和面板数据模型工具"
    version = "2.0.0"
    
    @classmethod
    def get_tools(cls) -> List[Dict[str, Any]]:
        """返回工具列表"""
        return [
            {
                "name": "time_series_arima_model",
                "handler": cls.arima_tool,
                "description": "ARIMA Time Series Model"
            },
            {
                "name": "time_series_exponential_smoothing",
                "handler": cls.exp_smoothing_tool,
                "description": "Exponential Smoothing Model"
            },
            {
                "name": "time_series_garch_model",
                "handler": cls.garch_tool,
                "description": "GARCH Volatility Model"
            },
            {
                "name": "time_series_unit_root_tests",
                "handler": cls.unit_root_tool,
                "description": "Unit Root Tests (ADF/PP/KPSS)"
            },
            {
                "name": "time_series_var_svar_model",
                "handler": cls.var_svar_tool,
                "description": "VAR/SVAR Model"
            },
            {
                "name": "time_series_cointegration_analysis",
                "handler": cls.cointegration_tool,
                "description": "Cointegration Analysis"
            },
            {
                "name": "panel_data_dynamic_model",
                "handler": cls.dynamic_panel_tool,
                "description": "Dynamic Panel Data Model"
            },
            {
                "name": "panel_data_diagnostics",
                "handler": cls.panel_diagnostics_tool,
                "description": "Panel Data Diagnostic Tests"
            },
            {
                "name": "panel_var_model",
                "handler": cls.panel_var_tool,
                "description": "Panel VAR Model"
            },
            {
                "name": "structural_break_tests",
                "handler": cls.structural_break_tool,
                "description": "Structural Break Tests"
            },
            {
                "name": "time_varying_parameter_models",
                "handler": cls.time_varying_parameter_tool,
                "description": "Time-Varying Parameter Models"
            }
        ]
    
    @classmethod
    def get_help_text(cls) -> str:
        """返回帮助文档"""
        return """
4. ARIMA Model (time_series_arima_model)
   - Order: (p,d,q) parameters
   - Forecasting: multi-step prediction
   
5. Exponential Smoothing (time_series_exponential_smoothing)
   - Components: trend, seasonal
   - Forecasting: multi-step prediction
   
6. GARCH Model (time_series_garch_model)
   - Volatility: conditional variance modeling
   - Order: (p,q) parameters
   
7. Unit Root Tests (time_series_unit_root_tests)
   - Tests: ADF, PP, KPSS
   - Stationarity: check for unit roots
   
8. VAR/SVAR Model (time_series_var_svar_model)
   - Models: VAR, SVAR
   - Multivariate: multiple time series analysis
   
9. Cointegration Analysis (time_series_cointegration_analysis)
   - Tests: Engle-Granger, Johansen
   - Models: VECM
   - Long-run: equilibrium relationships
   
10. Dynamic Panel Models (panel_data_dynamic_model)
    - Models: Difference GMM, System GMM
    - Panel: cross-sectional and time series data
    
11. Panel Data Diagnostics (panel_data_diagnostics)
    - Tests: Hausman, Pooling F, LM, Within Correlation
    - Model Selection: FE vs RE vs Pooled
    
12. Panel VAR Model (panel_var_model)
    - Panel Vector Autoregression
    - Individual and Time Effects
    
13. Structural Break Tests (structural_break_tests)
    - Tests: Chow, Quandt-Andrews, Bai-Perron
    - Detect: structural changes in time series
    
14. Time-Varying Parameter Models (time_varying_parameter_models)
    - Models: TAR, STAR, Markov Switching
    - Regime-switching: threshold-based transitions
"""

    @staticmethod
    async def arima_tool(
        data: Optional[List[float]] = None,
        file_path: Optional[str] = None,
        order: tuple = (1, 1, 1),
        forecast_steps: int = 1,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """ARIMA Time Series Model"""
        try:
            if ctx:
                await ctx.info("Starting ARIMA model analysis...")
            result = arima_adapter(data, file_path, order, forecast_steps, output_format, save_path)
            if ctx:
                await ctx.info("ARIMA analysis complete")
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def exp_smoothing_tool(
        data: Optional[List[float]] = None,
        file_path: Optional[str] = None,
        trend: bool = True,
        seasonal: bool = False,
        seasonal_periods: Optional[int] = None,
        forecast_steps: int = 1,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Exponential Smoothing Model"""
        try:
            if ctx:
                await ctx.info("Starting exponential smoothing analysis...")
            result = exp_smoothing_adapter(data, file_path, trend, seasonal, seasonal_periods, forecast_steps, output_format, save_path)
            if ctx:
                await ctx.info("Exponential smoothing complete")
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def garch_tool(
        data: Optional[List[float]] = None,
        file_path: Optional[str] = None,
        order: tuple = (1, 1),
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """GARCH Volatility Model"""
        try:
            if ctx:
                await ctx.info("Starting GARCH model analysis...")
            result = garch_adapter(data, file_path, order, output_format, save_path)
            if ctx:
                await ctx.info("GARCH analysis complete")
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def unit_root_tool(
        data: Optional[List[float]] = None,
        file_path: Optional[str] = None,
        test_type: str = "adf",
        max_lags: Optional[int] = None,
        regression_type: str = "c",
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Unit Root Tests"""
        try:
            if ctx:
                await ctx.info(f"Starting {test_type.upper()} unit root test...")
            result = unit_root_adapter(data, file_path, test_type, max_lags, regression_type, output_format, save_path)
            if ctx:
                await ctx.info(f"{test_type.upper()} test complete")
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def var_svar_tool(
        data: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        model_type: str = "var",
        lags: int = 1,
        variables: Optional[List[str]] = None,
        a_matrix: Optional[List[List[float]]] = None,
        b_matrix: Optional[List[List[float]]] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """
        VAR/SVAR Model
        
        数据格式说明:
        - data: 多元时间序列数据，格式为二维列表
        - 每个子列表代表一个时间点的所有变量值
        - 示例: [[var1_t1, var2_t1, var3_t1], [var1_t2, var2_t2, var3_t2], ...]
        - variables: 变量名称列表，如 ["GDP", "Inflation", "Interest"]
        
        示例调用:
        {
          "data": [[1.0, 2.5, 1.8], [1.2, 2.7, 2.0], [1.4, 2.9, 2.2]],
          "model_type": "var",
          "lags": 1,
          "variables": ["GDP", "Inflation", "Interest"],
          "output_format": "json"
        }
        """
        try:
            if ctx:
                await ctx.info(f"Starting {model_type.upper()} model analysis...")
            
            # 数据验证和转换
            if data is not None:
                # 确保数据是二维列表格式
                if isinstance(data[0], (int, float)):
                    data = [data]  # 如果是一维数据，转换为二维
                elif isinstance(data[0], list) and len(data) > 0 and isinstance(data[0][0], (int, float)):
                    # 已经是正确的二维格式
                    pass
                else:
                    raise ValueError("数据格式不正确，应为二维列表")
            
            result = var_svar_adapter(data, file_path, model_type, lags, variables, a_matrix, b_matrix, output_format, save_path)
            if ctx:
                await ctx.info(f"{model_type.upper()} analysis complete")
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def panel_diagnostics_tool(
        test_type: str = "hausman",
        fe_coefficients: Optional[List[float]] = None,
        re_coefficients: Optional[List[float]] = None,
        fe_covariance: Optional[List[List[float]]] = None,
        re_covariance: Optional[List[List[float]]] = None,
        pooled_ssrs: Optional[float] = None,
        fixed_ssrs: Optional[float] = None,
        random_ssrs: Optional[float] = None,
        n_individuals: Optional[int] = None,
        n_params: Optional[int] = None,
        n_obs: Optional[int] = None,
        n_periods: Optional[int] = None,
        residuals: Optional[List[List[float]]] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Panel Data Diagnostic Tests"""
        try:
            if ctx:
                await ctx.info(f"Starting {test_type} panel diagnostic test...")
            result = panel_diagnostics_adapter(
                test_type, fe_coefficients, re_coefficients, fe_covariance, re_covariance,
                pooled_ssrs, fixed_ssrs, random_ssrs, n_individuals, n_params, n_obs,
                n_periods, residuals, output_format, save_path
            )
            if ctx:
                await ctx.info(f"{test_type} test complete")
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def panel_var_tool(
        data: Optional[List[List[float]]] = None,
        entity_ids: Optional[List[int]] = None,
        time_periods: Optional[List[int]] = None,
        file_path: Optional[str] = None,
        lags: int = 1,
        variables: Optional[List[str]] = None,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Panel VAR Model"""
        try:
            if ctx:
                await ctx.info("Starting Panel VAR model analysis...")
            result = panel_var_adapter(
                data, entity_ids, time_periods, file_path, lags, variables,
                output_format, save_path
            )
            if ctx:
                await ctx.info("Panel VAR analysis complete")
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def structural_break_tool(
        data: Optional[List[float]] = None,
        file_path: Optional[str] = None,
        test_type: str = "chow",
        break_point: Optional[int] = None,
        max_breaks: int = 5,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Structural Break Tests"""
        try:
            if ctx:
                await ctx.info(f"Starting {test_type} structural break test...")
            result = structural_break_adapter(
                data, file_path, test_type, break_point, max_breaks,
                output_format, save_path
            )
            if ctx:
                await ctx.info(f"{test_type} test complete")
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def time_varying_parameter_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        model_type: str = "tar",
        threshold_variable: Optional[List[float]] = None,
        n_regimes: int = 2,
        star_type: str = "logistic",
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Time-Varying Parameter Models"""
        try:
            if ctx:
                await ctx.info(f"Starting {model_type.upper()} time-varying parameter model...")
            result = time_varying_parameter_adapter(
                y_data, x_data, file_path, model_type, threshold_variable,
                n_regimes, star_type, output_format, save_path
            )
            if ctx:
                await ctx.info(f"{model_type.upper()} model complete")
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def cointegration_tool(
        data: Optional[List[List[float]]] = None,
        file_path: Optional[str] = None,
        analysis_type: str = "johansen",
        variables: Optional[List[str]] = None,
        coint_rank: int = 1,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """Cointegration Analysis"""
        try:
            if ctx:
                await ctx.info(f"Starting {analysis_type} cointegration analysis...")
            
            # 数据验证和转换
            if data is not None:
                # 确保数据是二维列表格式，每行代表一个时间点的多个变量
                if isinstance(data[0], (int, float)):
                    data = [data]  # 如果是一维数据，转换为二维
                elif isinstance(data[0], list) and len(data) > 0 and isinstance(data[0][0], (int, float)):
                    # 已经是正确的二维格式
                    pass
                else:
                    raise ValueError("数据格式不正确，应为二维列表，每行代表一个时间点的多个变量值")
            
            result = cointegration_adapter(data, file_path, analysis_type, variables, coint_rank, output_format, save_path)
            if ctx:
                await ctx.info(f"{analysis_type} analysis complete")
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise

    @staticmethod
    async def dynamic_panel_tool(
        y_data: Optional[List[float]] = None,
        x_data: Optional[List[List[float]]] = None,
        entity_ids: Optional[List[int]] = None,
        time_periods: Optional[List[int]] = None,
        file_path: Optional[str] = None,
        model_type: str = "diff_gmm",
        lags: int = 1,
        output_format: str = "json",
        save_path: Optional[str] = None,
        ctx: Context[ServerSession, None] = None
    ) -> str:
        """
        Dynamic Panel Data Model
        
        数据格式说明:
        - y_data: 因变量数据，一维列表，所有个体的因变量时间序列
        - x_data: 自变量数据，二维列表，每个子列表代表一个自变量的完整时间序列
        - entity_ids: 个体标识符，一维列表，标识每个观测属于哪个个体
        - time_periods: 时间标识符，一维列表，标识每个观测的时间点
        
        重要: 所有数据的观测数量必须相同
        
        示例调用:
        {
          "y_data": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
          "x_data": [[1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3]],
          "entity_ids": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
          "time_periods": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
          "model_type": "diff_gmm",
          "lags": 1,
          "output_format": "json"
        }
        """
        try:
            if ctx:
                await ctx.info(f"Starting {model_type} dynamic panel model...")
            result = dynamic_panel_adapter(y_data, x_data, entity_ids, time_periods, file_path, model_type, lags, output_format, save_path)
            if ctx:
                await ctx.info(f"{model_type} model complete")
            return result
        except Exception as e:
            if ctx:
                await ctx.error(f"Error: {str(e)}")
            raise