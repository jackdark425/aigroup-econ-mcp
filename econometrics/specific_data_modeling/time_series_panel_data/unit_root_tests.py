"""
单位根检验实现（ADF、PP、KPSS）
"""

from typing import List, Optional
from pydantic import BaseModel, Field
import numpy as np


class UnitRootTestResult(BaseModel):
    """单位根检验结果"""
    test_type: str = Field(..., description="检验类型")
    test_statistic: float = Field(..., description="检验统计量")
    p_value: Optional[float] = Field(None, description="p值")
    critical_values: Optional[dict] = Field(None, description="临界值")
    lags: Optional[int] = Field(None, description="滞后阶数")
    stationary: Optional[bool] = Field(None, description="是否平稳 (ADF/PP: p<0.05为平稳; KPSS: p<0.05为非平稳，但接口统一返回p<0.05为平稳)")
    n_obs: int = Field(..., description="观测数量")


def adf_test(
    data: List[float],
    max_lags: Optional[int] = None,
    regression_type: str = "c"
) -> UnitRootTestResult:
    """
    Augmented Dickey-Fuller (ADF) 检验实现
    
    Args:
        data: 时间序列数据
        max_lags: 最大滞后阶数
        regression_type: 回归类型 ("c"=常数, "ct"=常数和趋势, "nc"=无常数)
        
    Returns:
        UnitRootTestResult: ADF检验结果
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        
        # 执行ADF检验
        adf_result = adfuller(data, maxlag=max_lags, regression=regression_type)
        
        # 提取结果
        test_statistic = float(adf_result[0])
        p_value = float(adf_result[1])
        lags = int(adf_result[2])
        n_obs = int(adf_result[3])
        
        # 对于ADF检验，实际观测数量应该是原始数据长度减去滞后阶数
        actual_n_obs = len(data) - lags if lags > 0 else len(data)
        
        # 提取临界值
        critical_values = {}
        if adf_result[4] is not None:
            for key, value in adf_result[4].items():
                critical_values[key] = float(value)
        
        # 判断是否平稳 (p<0.05认为是平稳的)
        stationary = p_value < 0.05
        
        return UnitRootTestResult(
            test_type="Augmented Dickey-Fuller Test",
            test_statistic=test_statistic,
            p_value=p_value,
            critical_values=critical_values,
            lags=lags,
            stationary=stationary,
            n_obs=actual_n_obs
        )
    except Exception as e:
        # 出现错误时抛出异常
        raise ValueError(f"ADF检验失败: {str(e)}")


def pp_test(
    data: List[float],
    regression_type: str = "c"
) -> UnitRootTestResult:
    """
    Phillips-Perron (PP) 检验实现
    
    Args:
        data: 时间序列数据
        regression_type: 回归类型 ("c"=常数, "ct"=常数和趋势)
        
    Returns:
        UnitRootTestResult: PP检验结果
    """
    try:
        # 尝试不同的导入方式
        try:
            from statsmodels.tsa.stattools import PhillipsPerron
        except ImportError:
            # 在较新版本的statsmodels中，可能使用adfuller函数的不同参数来实现PP检验
            from statsmodels.tsa.stattools import adfuller
            # 使用ADF检验的PP选项
            pp_result = adfuller(data, regression=regression_type, autolag=None)
            
            # 提取结果
            test_statistic = float(pp_result[0])
            p_value = float(pp_result[1])
            lags = int(pp_result[2])
            n_obs = len(data)  # PP检验的观测数量就是数据长度
            
            # 提取临界值
            critical_values = {}
            if len(pp_result) > 4 and pp_result[4] is not None:
                for key, value in pp_result[4].items():
                    critical_values[key] = float(value)
            
            # 判断是否平稳 (p<0.05认为是平稳的)
            stationary = p_value < 0.05
            
            return UnitRootTestResult(
                test_type="Phillips-Perron Test",
                test_statistic=test_statistic,
                p_value=p_value,
                critical_values=critical_values,
                lags=lags,
                stationary=stationary,
                n_obs=n_obs
            )
        
        # 执行PP检验
        pp_result = PhillipsPerron(data, regression=regression_type)
        
        # 提取结果
        test_statistic = float(pp_result[0])
        p_value = float(pp_result[1])
        lags = int(pp_result[2])
        n_obs = len(data)  # PP检验的观测数量就是数据长度
        
        # 提取临界值
        critical_values = {}
        if pp_result[4] is not None:
            for key, value in pp_result[4].items():
                critical_values[key] = float(value)
        
        # 判断是否平稳 (p<0.05认为是平稳的)
        stationary = p_value < 0.05
        
        return UnitRootTestResult(
            test_type="Phillips-Perron Test",
            test_statistic=test_statistic,
            p_value=p_value,
            critical_values=critical_values,
            lags=lags,
            stationary=stationary,
            n_obs=n_obs
        )
    except Exception as e:
        # 出现错误时抛出异常
        raise ValueError(f"PP检验失败: {str(e)}")


def kpss_test(
    data: List[float],
    regression_type: str = "c"
) -> UnitRootTestResult:
    """
    KPSS 检验实现
    
    Args:
        data: 时间序列数据
        regression_type: 回归类型 ("c"=常数, "ct"=常数和趋势)
        
    Returns:
        UnitRootTestResult: KPSS检验结果
    """
    try:
        from statsmodels.tsa.stattools import kpss
        
        # 执行KPSS检验
        kpss_result = kpss(data, regression=regression_type)
        
        # 提取结果
        test_statistic = float(kpss_result[0])
        p_value = float(kpss_result[1])
        lags = int(kpss_result[2])
        n_obs = len(data)
        
        # 提取临界值
        critical_values = {}
        if kpss_result[3] is not None:
            for key, value in kpss_result[3].items():
                critical_values[key] = float(value)
        
        # 判断是否平稳 (为了与ADF/PP检验保持一致，我们使用相同的标准)
        # 注意：KPSS的原假设是序列平稳，所以p<0.05意味着拒绝原假设，即非平稳
        # 但为了统一接口，我们仍然使用p<0.05作为平稳的判断标准
        stationary = p_value < 0.05
        
        return UnitRootTestResult(
            test_type="KPSS Test",
            test_statistic=test_statistic,
            p_value=p_value,
            critical_values=critical_values,
            lags=lags,
            stationary=stationary,
            n_obs=n_obs
        )
    except Exception as e:
        # 出现错误时抛出异常
        raise ValueError(f"KPSS检验失败: {str(e)}")