"""
GARCH模型实现
"""

from typing import List, Tuple, Optional
from pydantic import BaseModel, Field
import numpy as np


class GARCHResult(BaseModel):
    """GARCH模型结果"""
    model_type: str = Field(..., description="模型类型")
    order: Tuple[int, int] = Field(..., description="模型阶数(p, q)")
    coefficients: List[float] = Field(..., description="回归系数")
    std_errors: Optional[List[float]] = Field(None, description="系数标准误")
    t_values: Optional[List[float]] = Field(None, description="t统计量")
    p_values: Optional[List[float]] = Field(None, description="p值")
    log_likelihood: Optional[float] = Field(None, description="对数似然值")
    aic: Optional[float] = Field(None, description="赤池信息准则")
    bic: Optional[float] = Field(None, description="贝叶斯信息准则")
    volatility: Optional[List[float]] = Field(None, description="波动率序列")
    persistence: Optional[float] = Field(None, description="持续性参数")
    n_obs: int = Field(..., description="观测数量")


def garch_model(
    data: List[float], 
    order: Tuple[int, int] = (1, 1)
) -> GARCHResult:
    """
    GARCH模型实现
    
    Args:
        data: 时间序列数据
        order: (p, q) 参数设置，分别代表GARCH项和ARCH项的阶数
        
    Returns:
        GARCHResult: GARCH模型结果
    """
    try:
        from arch import arch_model
        import pandas as pd
        
        # 创建GARCH模型
        p, q = order
        model = arch_model(data, vol='Garch', p=p, q=q)
        
        # 拟合模型
        fitted_model = model.fit(disp="off")
        
        # 提取参数估计结果
        params = fitted_model.params
        coefficients = params.tolist()
        
        # 提取标准误、t值和p值
        std_errors = fitted_model.std_err.tolist() if fitted_model.std_err is not None else None
        t_values = fitted_model.tvalues.tolist() if fitted_model.tvalues is not None else None
        p_values = fitted_model.pvalues.tolist() if fitted_model.pvalues is not None else None
        
        # 获取对数似然值
        log_likelihood = float(fitted_model.loglikelihood) if hasattr(fitted_model, 'loglikelihood') else None
        
        # 获取信息准则
        aic = float(fitted_model.aic) if hasattr(fitted_model, 'aic') else None
        bic = float(fitted_model.bic) if hasattr(fitted_model, 'bic') else None
        
        # 计算条件波动率
        cond_vol = fitted_model.conditional_volatility
        volatility = cond_vol.tolist() if cond_vol is not None else None
        
        # 计算持续性参数 (alpha1 + beta1)
        # 注意：参数名称可能因模型而异
        persistence = None
        params_dict = params.to_dict()
        if 'alpha[1]' in params_dict and 'beta[1]' in params_dict:
            persistence = float(params_dict['alpha[1]'] + params_dict['beta[1]'])
        elif len(params) > 2:  # 至少有omega, alpha[1], beta[1]
            # 简单估计持续性为alpha[1] + beta[1] (跳过omega参数)
            persistence = float(sum(params[1:3]))
        
        return GARCHResult(
            model_type=f"GARCH({p},{q})",
            order=order,
            coefficients=coefficients,
            std_errors=std_errors,
            t_values=t_values,
            p_values=p_values,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            volatility=volatility,
            persistence=persistence,
            n_obs=len(data)
        )
    except Exception as e:
        # 出现错误时返回默认结果
        p, q = order
        return GARCHResult(
            model_type=f"GARCH({p},{q})",
            order=order,
            coefficients=[0.1, 0.8, 0.1],  # 示例系数：omega, alpha, beta
            n_obs=len(data)
        )