"""
动态面板模型实现（差分GMM、系统GMM）
"""

from typing import List, Optional
from pydantic import BaseModel, Field
import numpy as np


class DynamicPanelResult(BaseModel):
    """动态面板模型结果"""
    model_type: str = Field(..., description="模型类型")
    coefficients: List[float] = Field(..., description="回归系数")
    std_errors: Optional[List[float]] = Field(None, description="系数标准误")
    t_values: Optional[List[float]] = Field(None, description="t统计量")
    p_values: Optional[List[float]] = Field(None, description="p值")
    conf_int_lower: Optional[List[float]] = Field(None, description="置信区间下界")
    conf_int_upper: Optional[List[float]] = Field(None, description="置信区间上界")
    instruments: Optional[int] = Field(None, description="工具变量数量")
    j_statistic: Optional[float] = Field(None, description="过度识别约束检验统计量")
    j_p_value: Optional[float] = Field(None, description="过度识别约束检验p值")
    n_obs: int = Field(..., description="观测数量")
    n_individuals: int = Field(..., description="个体数量")
    n_time_periods: int = Field(..., description="时间期数")


def diff_gmm_model(
    y_data: List[float],
    x_data: List[List[float]],
    entity_ids: List[int],
    time_periods: List[int],
    lags: int = 1
) -> DynamicPanelResult:
    """
    差分GMM模型实现（Arellano-Bond估计）
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据 (格式: 每个子列表代表一个自变量的时间序列)
        entity_ids: 个体标识符
        time_periods: 时间标识符
        lags: 滞后期数
        
    Returns:
        DynamicPanelResult: 差分GMM模型结果
    """
    try:
        from linearmodels.panel import DifferenceGMM
        import pandas as pd
        
        # 输入验证
        if not y_data:
            raise ValueError("因变量数据不能为空")
            
        if not x_data:
            raise ValueError("自变量数据不能为空")
            
        if not all(isinstance(series, (list, tuple)) for series in x_data):
            raise ValueError("自变量数据必须是二维列表格式")
            
        if not entity_ids:
            raise ValueError("个体标识符不能为空")
            
        if not time_periods:
            raise ValueError("时间标识符不能为空")
            
        # 检查数据长度一致性
        lengths = [len(y_data), len(entity_ids), len(time_periods)]
        for i, x_series in enumerate(x_data):
            lengths.append(len(x_series))
            
        if len(set(lengths)) > 1:
            raise ValueError(f"所有数据序列的长度必须一致，当前长度分别为: {lengths}")
        
        # 创建面板数据结构
        # 构建MultiIndex
        index = pd.MultiIndex.from_arrays([entity_ids, time_periods], names=['entity', 'time'])
        
        # 检查索引有效性
        if index.has_duplicates:
            raise ValueError("存在重复的个体-时间索引")
            
        # 构建因变量DataFrame
        y_df = pd.DataFrame({'y': y_data}, index=index)
        
        # 构建自变量DataFrame
        x_dict = {}
        for i, x in enumerate(x_data):
            x_dict[f'x{i}'] = x
        x_df = pd.DataFrame(x_dict, index=index)
        
        # 检查面板数据结构
        if y_df.empty or x_df.empty:
            raise ValueError("构建的面板数据为空")
        
        # 创建并拟合差分GMM模型
        model = DifferenceGMM(y_df, x_df, lags=lags)
        fitted_model = model.fit()
        
        # 提取参数估计结果
        params = fitted_model.params.tolist()
        
        # 提取标准误
        std_errors = fitted_model.std_errors.tolist() if fitted_model.std_errors is not None else None
        
        # 提取t值
        t_values = fitted_model.tstats.tolist() if fitted_model.tstats is not None else None
        
        # 提取p值
        p_values = fitted_model.pvalues.tolist() if fitted_model.pvalues is not None else None
        
        # 计算置信区间 (95%)
        if fitted_model.conf_int() is not None:
            conf_int = fitted_model.conf_int()
            conf_int_lower = conf_int.iloc[:, 0].tolist()
            conf_int_upper = conf_int.iloc[:, 1].tolist()
        else:
            conf_int_lower = None
            conf_int_upper = None
        
        # 提取工具变量数量
        instruments = None
        try:
            if hasattr(fitted_model, 'summary') and len(fitted_model.summary.tables) > 0:
                instruments = int(fitted_model.summary.tables[0].data[6][1])
        except (IndexError, ValueError, TypeError):
            # 如果无法提取工具变量数量，则保持为None
            instruments = None
        
        # 提取J统计量（过度识别约束检验）
        j_statistic = float(fitted_model.j_stat.stat) if hasattr(fitted_model, 'j_stat') and hasattr(fitted_model.j_stat, 'stat') else None
        j_p_value = float(fitted_model.j_stat.pval) if hasattr(fitted_model, 'j_stat') and hasattr(fitted_model.j_stat, 'pval') else None
        
        return DynamicPanelResult(
            model_type="Difference GMM (Arellano-Bond)",
            coefficients=params,
            std_errors=std_errors,
            t_values=t_values,
            p_values=p_values,
            conf_int_lower=conf_int_lower,
            conf_int_upper=conf_int_upper,
            instruments=instruments,
            j_statistic=j_statistic,
            j_p_value=j_p_value,
            n_obs=len(y_data),
            n_individuals=len(set(entity_ids)),
            n_time_periods=len(set(time_periods))
        )
    except Exception as e:
        # 出现错误时抛出异常
        raise ValueError(f"差分GMM模型拟合失败: {str(e)}")


def sys_gmm_model(
    y_data: List[float],
    x_data: List[List[float]],
    entity_ids: List[int],
    time_periods: List[int],
    lags: int = 1
) -> DynamicPanelResult:
    """
    系统GMM模型实现（Blundell-Bond估计）
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        entity_ids: 个体标识符
        time_periods: 时间标识符
        lags: 滞后期数
        
    Returns:
        DynamicPanelResult: 系统GMM模型结果
    """
    try:
        from linearmodels.panel import SystemGMM
        import pandas as pd
        
        # 创建面板数据结构
        # 构建MultiIndex
        index = pd.MultiIndex.from_arrays([entity_ids, time_periods], names=['entity', 'time'])
        
        # 构建因变量DataFrame
        y_df = pd.DataFrame({'y': y_data}, index=index)
        
        # 构建自变量DataFrame
        x_dict = {}
        for i, x in enumerate(x_data):
            x_dict[f'x{i}'] = x
        x_df = pd.DataFrame(x_dict, index=index)
        
        # 创建并拟合系统GMM模型
        model = SystemGMM(y_df, x_df, lags=lags)
        fitted_model = model.fit()
        
        # 提取参数估计结果
        params = fitted_model.params.tolist()
        
        # 提取标准误
        std_errors = fitted_model.std_errors.tolist() if fitted_model.std_errors is not None else None
        
        # 提取t值
        t_values = fitted_model.tstats.tolist() if fitted_model.tstats is not None else None
        
        # 提取p值
        p_values = fitted_model.pvalues.tolist() if fitted_model.pvalues is not None else None
        
        # 计算置信区间 (95%)
        if fitted_model.conf_int() is not None:
            conf_int = fitted_model.conf_int()
            conf_int_lower = conf_int.iloc[:, 0].tolist()
            conf_int_upper = conf_int.iloc[:, 1].tolist()
        else:
            conf_int_lower = None
            conf_int_upper = None
        
        # 提取工具变量数量
        instruments = None
        try:
            if hasattr(fitted_model, 'summary') and len(fitted_model.summary.tables) > 0:
                instruments = int(fitted_model.summary.tables[0].data[6][1])
        except (IndexError, ValueError, TypeError):
            # 如果无法提取工具变量数量，则保持为None
            instruments = None
        
        # 提取J统计量（过度识别约束检验）
        j_statistic = float(fitted_model.j_stat.stat) if hasattr(fitted_model, 'j_stat') and hasattr(fitted_model.j_stat, 'stat') else None
        j_p_value = float(fitted_model.j_stat.pval) if hasattr(fitted_model, 'j_stat') and hasattr(fitted_model.j_stat, 'pval') else None
        
        return DynamicPanelResult(
            model_type="System GMM (Blundell-Bond)",
            coefficients=params,
            std_errors=std_errors,
            t_values=t_values,
            p_values=p_values,
            conf_int_lower=conf_int_lower,
            conf_int_upper=conf_int_upper,
            instruments=instruments,
            j_statistic=j_statistic,
            j_p_value=j_p_value,
            n_obs=len(y_data),
            n_individuals=len(set(entity_ids)),
            n_time_periods=len(set(time_periods))
        )
    except Exception as e:
        # 出现错误时抛出异常
        raise ValueError(f"系统GMM模型拟合失败: {str(e)}")
