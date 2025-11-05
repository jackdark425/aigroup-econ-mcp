"""
指数平滑法模型实现
"""

from typing import List, Optional
from pydantic import BaseModel, Field
import numpy as np


class ExponentialSmoothingResult(BaseModel):
    """指数平滑法模型结果"""
    model_type: str = Field(..., description="模型类型")
    smoothing_level: Optional[float] = Field(None, description="水平平滑参数")
    smoothing_trend: Optional[float] = Field(None, description="趋势平滑参数")
    smoothing_seasonal: Optional[float] = Field(None, description="季节平滑参数")
    coefficients: List[float] = Field(..., description="模型系数")
    std_errors: Optional[List[float]] = Field(None, description="系数标准误")
    t_values: Optional[List[float]] = Field(None, description="t统计量")
    p_values: Optional[List[float]] = Field(None, description="p值")
    aic: Optional[float] = Field(None, description="赤池信息准则")
    bic: Optional[float] = Field(None, description="贝叶斯信息准则")
    sse: Optional[float] = Field(None, description="误差平方和")
    mse: Optional[float] = Field(None, description="均方误差")
    rmse: Optional[float] = Field(None, description="均方根误差")
    mae: Optional[float] = Field(None, description="平均绝对误差")
    n_obs: int = Field(..., description="观测数量")
    forecast: Optional[List[float]] = Field(None, description="预测值")


def exponential_smoothing_model(
    data: List[float],
    trend: bool = True,
    seasonal: bool = False,
    seasonal_periods: Optional[int] = None,
    forecast_steps: int = 1
) -> ExponentialSmoothingResult:
    """
    指数平滑法模型实现
    
    Args:
        data: 时间序列数据
        trend: 是否包含趋势成分
        seasonal: 是否包含季节成分
        seasonal_periods: 季节周期长度
        forecast_steps: 预测步数
        
    Returns:
        ExponentialSmoothingResult: 指数平滑法模型结果
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # 输入验证
        if not data:
            raise ValueError("时间序列数据不能为空")
            
        if len(data) < 2:
            raise ValueError("时间序列数据长度必须至少为2")
            
        # 检查数据有效性
        data_array = np.array(data, dtype=np.float64)
        if np.isnan(data_array).any():
            raise ValueError("数据中包含缺失值(NaN)")
            
        if np.isinf(data_array).any():
            raise ValueError("数据中包含无穷大值")
        
        # 检查季节性参数
        if seasonal and seasonal_periods is None:
            raise ValueError("启用季节性时，必须指定季节周期长度")
            
        if seasonal_periods is not None and seasonal_periods <= 0:
            raise ValueError("季节周期长度必须为正整数")
            
        if seasonal_periods is not None and seasonal_periods > len(data) // 2:
            raise ValueError("季节周期长度不能超过数据长度的一半")
        
        # 检查预测步数
        if forecast_steps <= 0:
            raise ValueError("预测步数必须为正整数")
            
        # 构建模型
        model = ExponentialSmoothing(
            data, 
            trend="add" if trend else None, 
            seasonal="add" if seasonal else None, 
            seasonal_periods=seasonal_periods
        )
        
        # 拟合模型
        fitted_model = model.fit()
        
        # 获取参数
        smoothing_level = float(fitted_model.params['smoothing_level']) if 'smoothing_level' in fitted_model.params else None
        smoothing_trend = float(fitted_model.params['smoothing_trend']) if 'smoothing_trend' in fitted_model.params else None
        smoothing_seasonal = float(fitted_model.params['smoothing_seasonal']) if 'smoothing_seasonal' in fitted_model.params else None
        
        # 提取所有参数作为系数
        coefficients = []
        for param_name, param_value in fitted_model.params.items():
            # 确保参数值有效
            if np.isscalar(param_value) and np.isfinite(param_value):
                coefficients.append(float(param_value))
            elif hasattr(param_value, 'size') and param_value.size > 0:
                # 如果是数组且非空，取第一个元素
                val = param_value.item() if hasattr(param_value, 'item') else param_value[0]
                if np.isfinite(val):
                    coefficients.append(float(val))
        
        # 如果没有参数或参数无效，使用默认值
        if not coefficients or any(not np.isfinite(coeff) for coeff in coefficients):
            coefficients = [smoothing_level or 0.5]
            if smoothing_trend is not None:
                coefficients.append(smoothing_trend)
            if smoothing_seasonal is not None:
                coefficients.append(smoothing_seasonal)
        
        # 获取拟合值和残差用于计算指标
        fitted_values = fitted_model.fittedvalues
        
        # 检查拟合值有效性
        if np.isnan(fitted_values).any() or np.isinf(fitted_values).any():
            raise ValueError("模型拟合值包含无效值")
        
        residuals = np.array(data) - np.array(fitted_values)
        
        # 计算各种评估指标
        sse = float(np.sum(residuals**2))
        mse = float(mean_squared_error(data, fitted_values))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(data, fitted_values))
        
        # 检查指标有效性
        if not np.isfinite(sse) or not np.isfinite(mse) or not np.isfinite(rmse) or not np.isfinite(mae):
            raise ValueError("计算出的评估指标包含无效值")
        
        # 进行预测
        forecast = fitted_model.forecast(steps=forecast_steps).tolist()
        
        # 检查预测值有效性
        if np.isnan(forecast).any() or np.isinf(forecast).any():
            raise ValueError("预测值包含无效值")
        
        # 构建模型类型描述
        model_type = "Exponential Smoothing"
        if trend:
            model_type += " with Trend"
        if seasonal:
            model_type += " with Seasonal"
            
        # 获取信息准则
        aic = float(fitted_model.aic) if hasattr(fitted_model, 'aic') and np.isfinite(fitted_model.aic) else None
        bic = float(fitted_model.bic) if hasattr(fitted_model, 'bic') and np.isfinite(fitted_model.bic) else None
        
        return ExponentialSmoothingResult(
            model_type=model_type,
            smoothing_level=smoothing_level,
            smoothing_trend=smoothing_trend,
            smoothing_seasonal=smoothing_seasonal,
            coefficients=coefficients,
            std_errors=None,  # statsmodels的指数平滑不提供标准误
            t_values=None,
            p_values=None,
            aic=aic,
            bic=bic,
            sse=sse,
            mse=mse,
            rmse=rmse,
            mae=mae,
            n_obs=len(data),
            forecast=forecast
        )
    except Exception as e:
        # 出现错误时抛出异常
        raise ValueError(f"指数平滑模型拟合失败: {str(e)}")
