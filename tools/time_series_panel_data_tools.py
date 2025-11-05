"""
时间序列和面板数据工具注册
将时间序列和面板数据模型注册为MCP工具
"""

from typing import List, Optional
from .time_series_panel_data_adapter import TimeSeriesPanelDataAdapter


def register_time_series_panel_data_tools():
    """
    注册时间序列和面板数据工具
    """
    # 这个函数可以用于初始化时注册工具
    pass


# 工具函数别名，便于直接调用
arima_model = TimeSeriesPanelDataAdapter.arima_model
exponential_smoothing_model = TimeSeriesPanelDataAdapter.exponential_smoothing_model
garch_model = TimeSeriesPanelDataAdapter.garch_model
unit_root_tests = TimeSeriesPanelDataAdapter.unit_root_tests
var_svar_model = TimeSeriesPanelDataAdapter.var_svar_model
cointegration_analysis = TimeSeriesPanelDataAdapter.cointegration_analysis
dynamic_panel_model = TimeSeriesPanelDataAdapter.dynamic_panel_model


# 工具列表
TIME_SERIES_PANEL_DATA_TOOLS = [
    {
        "name": "arima_model",
        "function": arima_model,
        "description": "ARIMA模型，用于时间序列建模和预测"
    },
    {
        "name": "exponential_smoothing_model",
        "function": exponential_smoothing_model,
        "description": "指数平滑模型，用于时间序列建模和预测"
    },
    {
        "name": "garch_model",
        "function": garch_model,
        "description": "GARCH模型，用于波动率建模"
    },
    {
        "name": "unit_root_tests",
        "function": unit_root_tests,
        "description": "单位根检验，用于检验时间序列的平稳性"
    },
    {
        "name": "var_svar_model",
        "function": var_svar_model,
        "description": "向量自回归(VAR)和结构向量自回归(SVAR)模型"
    },
    {
        "name": "cointegration_analysis",
        "function": cointegration_analysis,
        "description": "协整分析，包括Engle-Granger检验、Johansen检验和VECM模型"
    },
    {
        "name": "dynamic_panel_model",
        "function": dynamic_panel_model,
        "description": "动态面板模型，包括差分GMM和系统GMM"
    }
]