"""
工具变量法 (IV/2SLS) 实现
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
from linearmodels.iv import IV2SLS


class IVResult(BaseModel):
    """工具变量法结果"""
    method: str = Field(default="Instrumental Variables (2SLS)", description="使用的因果识别方法")
    estimate: float = Field(..., description="因果效应估计值")
    std_error: float = Field(..., description="标准误")
    t_statistic: float = Field(..., description="t统计量")
    p_value: float = Field(..., description="p值")
    confidence_interval: List[float] = Field(..., description="置信区间")
    n_observations: int = Field(..., description="观测数量")
    first_stage_f_stat: Optional[float] = Field(None, description="第一阶段F统计量")


def instrumental_variables_2sls(
    y: List[float],
    x: List[List[float]],
    instruments: List[List[float]],
    feature_names: Optional[List[str]] = None,
    instrument_names: Optional[List[str]] = None,
    constant: bool = True
) -> IVResult:
    """
    工具变量法 (IV/2SLS)
    
    使用linearmodels.iv.IV2SLS实现工具变量回归，解决内生性问题。
    
    Args:
        y: 因变量
        x: 内生自变量
        instruments: 工具变量
        feature_names: 特征名称
        instrument_names: 工具变量名称
        constant: 是否包含常数项
        
    Returns:
        IVResult: 工具变量法结果
    """
    # 参数验证
    n = len(y)
    if n == 0:
        raise ValueError("因变量y不能为空")
    
    if len(x) != n:
        raise ValueError("自变量x的长度必须与因变量y相同")
    
    if len(instruments) != n:
        raise ValueError("工具变量instruments的长度必须与因变量y相同")
    
    # 转换为DataFrame格式以适应linearmodels
    data = {}
    data['y'] = y
    
    # 处理自变量
    x_array = np.array(x)
    if x_array.ndim == 1:
        x_array = x_array.reshape(-1, 1)
    
    k_x = x_array.shape[1]
    for i in range(k_x):
        var_name = feature_names[i] if feature_names and i < len(feature_names) else f"x{i+1}"
        data[var_name] = x_array[:, i]
    
    # 处理工具变量
    z_array = np.array(instruments)
    if z_array.ndim == 1:
        z_array = z_array.reshape(-1, 1)
    
    k_z = z_array.shape[1]
    for i in range(k_z):
        var_name = instrument_names[i] if instrument_names and i < len(instrument_names) else f"z{i+1}"
        data[var_name] = z_array[:, i]
    
    df = pd.DataFrame(data)
    
    # 确定因变量和自变量列名
    y_var = 'y'
    x_vars = [feature_names[i] if feature_names and i < len(feature_names) else f"x{i+1}" 
              for i in range(k_x)]
    z_vars = [instrument_names[i] if instrument_names and i < len(instrument_names) else f"z{i+1}" 
              for i in range(k_z)]
    
    # 如果需要添加常数项
    if constant:
        df['const'] = 1
        x_vars = ['const'] + x_vars
        z_vars = ['const'] + z_vars
    
    # 使用linearmodels进行2SLS估计
    dependent = df[y_var]
    exog_vars = df[x_vars] if x_vars else None
    instr_vars = df[z_vars]
    
    # 将内生变量和外生变量分开
    # 假设所有x变量都是内生的，所有z变量都是工具变量
    endog = df[[var for var in x_vars if var in df.columns]]
    
    model = IV2SLS(dependent=dependent, exog=None, endog=endog, instruments=instr_vars)
    results = model.fit()
    
    # 提取主要结果（假设我们关注最后一个变量的系数，排除常数项）
    if feature_names:
        target_var = feature_names[-1]
    else:
        # 如果没有提供feature_names，使用最后一个x变量
        target_var = f"x{k_x}"
    
    # 如果包含常数项，确保不选择常数项作为目标变量
    if constant and target_var == 'const':
        if feature_names:
            target_var = feature_names[-1]
        else:
            target_var = f"x{k_x}"
    
    coef = results.params[target_var]
    stderr = results.std_errors[target_var]
    tstat = results.tstats[target_var]
    pval = results.pvalues[target_var]
    
    # 计算置信区间
    ci_lower = coef - 1.96 * stderr
    ci_upper = coef + 1.96 * stderr
    
    # 第一阶段F统计量（简化处理）
    first_stage_f = None  # linearmodels的结果中可能需要额外提取
    
    return IVResult(
        estimate=float(coef),
        std_error=float(stderr),
        t_statistic=float(tstat),
        p_value=float(pval),
        confidence_interval=[float(ci_lower), float(ci_upper)],
        n_observations=n,
        first_stage_f_stat=first_stage_f
    )