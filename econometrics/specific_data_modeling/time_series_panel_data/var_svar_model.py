"""
VAR/SVAR模型实现
"""

from typing import List, Optional
from pydantic import BaseModel, Field
import numpy as np


class VARResult(BaseModel):
    """VAR/SVAR模型结果"""
    model_type: str = Field(..., description="模型类型")
    lags: int = Field(..., description="滞后期数")
    variables: List[str] = Field(..., description="变量名称")
    coefficients: List[float] = Field(..., description="回归系数")
    std_errors: Optional[List[float]] = Field(None, description="系数标准误")
    t_values: Optional[List[float]] = Field(None, description="t统计量")
    p_values: Optional[List[float]] = Field(None, description="p值")
    aic: Optional[float] = Field(None, description="赤池信息准则")
    bic: Optional[float] = Field(None, description="贝叶斯信息准则")
    fpe: Optional[float] = Field(None, description="最终预测误差")
    hqic: Optional[float] = Field(None, description="汉南-奎因信息准则")
    irf: Optional[List[float]] = Field(None, description="脉冲响应函数")
    fevd: Optional[List[float]] = Field(None, description="方差分解")
    n_obs: int = Field(..., description="观测数量")


def var_model(
    data: List[List[float]],
    lags: int = 1,
    variables: Optional[List[str]] = None
) -> VARResult:
    """
    向量自回归(VAR)模型实现
    
    Args:
        data: 多元时间序列数据 (格式: 每个子列表代表一个变量的时间序列)
        lags: 滞后期数
        variables: 变量名称列表
        
    Returns:
        VARResult: VAR模型结果
    """
    try:
        from statsmodels.tsa.vector_ar.var_model import VAR
        import pandas as pd
        
        # 输入验证
        if not data:
            raise ValueError("数据不能为空")
            
        if not all(isinstance(series, (list, tuple)) for series in data):
            raise ValueError("数据必须是二维列表格式，每个子列表代表一个变量的时间序列")
            
        # 检查所有时间序列长度是否一致
        series_lengths = [len(series) for series in data]
        if len(set(series_lengths)) > 1:
            raise ValueError(f"所有时间序列的长度必须一致，当前长度分别为: {series_lengths}")
            
        # 转换数据格式
        data_array = np.array(data, dtype=np.float64).T  # 转置以匹配VAR模型要求的格式
        
        # 检查数据有效性
        if np.isnan(data_array).any():
            raise ValueError("数据中包含缺失值(NaN)")
            
        if np.isinf(data_array).any():
            raise ValueError("数据中包含无穷大值")
        
        # 创建变量名
        if variables is None:
            variables = [f"Variable_{i}" for i in range(len(data))]
        
        # 检查变量数量是否与数据一致
        if len(variables) != len(data):
            raise ValueError(f"变量名称数量({len(variables)})与数据列数({len(data)})不一致")
        
        # 创建DataFrame
        df = pd.DataFrame(data_array, columns=variables)
        
        # 检查滞后期数是否合理
        if lags <= 0:
            raise ValueError("滞后期数必须为正整数")
            
        if lags >= len(df):
            raise ValueError("滞后期数必须小于样本数量")
        
        # 创建并拟合VAR模型
        model = VAR(df)
        fitted_model = model.fit(lags)
        
        # 提取参数估计结果
        # VAR模型的系数是矩阵形式，需要展平
        coeffs = []
        std_errors = []
        t_values = []
        p_values = []
        
        for i in range(len(variables)):
            # 对于每个方程
            coeffs.extend(fitted_model.coefs[:, :, i].flatten().tolist())
            # 安全地提取标准误
            try:
                if fitted_model.stderr is not None:
                    # 检查stderr的维度
                    if hasattr(fitted_model.stderr, 'shape') and len(fitted_model.stderr.shape) == 3:
                        std_errors.extend(fitted_model.stderr[:, :, i].flatten().tolist())
                    elif hasattr(fitted_model.stderr, '__iter__'):
                        # 如果stderr是可迭代的，尝试安全转换
                        for se in fitted_model.stderr:
                            if np.isscalar(se) and np.isfinite(se):
                                std_errors.append(float(se))
            except (IndexError, TypeError, AttributeError, ValueError):
                # 如果无法提取标准误，保持列表为空
                pass
                
            # 安全地提取t值
            try:
                if fitted_model.tvalues is not None:
                    # 检查tvalues的维度
                    if hasattr(fitted_model.tvalues, 'shape') and len(fitted_model.tvalues.shape) == 3:
                        t_values.extend(fitted_model.tvalues[:, :, i].flatten().tolist())
                    elif hasattr(fitted_model.tvalues, '__iter__'):
                        # 如果tvalues是可迭代的，尝试安全转换
                        for tv in fitted_model.tvalues:
                            if np.isscalar(tv) and np.isfinite(tv):
                                t_values.append(float(tv))
            except (IndexError, TypeError, AttributeError, ValueError):
                # 如果无法提取t值，保持列表为空
                pass
                
            # 安全地提取p值
            try:
                if fitted_model.pvalues is not None:
                    # 检查pvalues的维度
                    if hasattr(fitted_model.pvalues, 'shape') and len(fitted_model.pvalues.shape) == 3:
                        p_values.extend(fitted_model.pvalues[:, :, i].flatten().tolist())
                    elif hasattr(fitted_model.pvalues, '__iter__'):
                        # 如果pvalues是可迭代的，尝试安全转换
                        for pv in fitted_model.pvalues:
                            if np.isscalar(pv) and np.isfinite(pv):
                                p_values.append(float(pv))
            except (IndexError, TypeError, AttributeError, ValueError):
                # 如果无法提取p值，保持列表为空
                pass
        
        # 获取信息准则
        aic = float(fitted_model.aic) if hasattr(fitted_model, 'aic') else None
        bic = float(fitted_model.bic) if hasattr(fitted_model, 'bic') else None
        fpe = float(fitted_model.fpe) if hasattr(fitted_model, 'fpe') else None
        hqic = float(fitted_model.hqic) if hasattr(fitted_model, 'hqic') else None
        
        # 计算脉冲响应函数 (前10期)
        irf_result = fitted_model.irf(10)
        irf = irf_result.irfs.flatten().tolist() if irf_result.irfs is not None else None
        
        # 计算方差分解 (前10期)
        fevd_result = fitted_model.fevd(10)
        fevd = fevd_result.decomp.flatten().tolist() if fevd_result.decomp is not None else None
        
        return VARResult(
            model_type=f"VAR({lags})",
            lags=lags,
            variables=variables,
            coefficients=coeffs,
            std_errors=std_errors if std_errors else None,
            t_values=t_values if t_values else None,
            p_values=p_values if p_values else None,
            aic=aic,
            bic=bic,
            fpe=fpe,
            hqic=hqic,
            irf=irf,
            fevd=fevd,
            n_obs=len(data[0]) if data else 0
        )
    except Exception as e:
        # 出现错误时抛出异常
        raise ValueError(f"VAR模型拟合失败: {str(e)}")


def svar_model(
    data: List[List[float]],
    lags: int = 1,
    variables: Optional[List[str]] = None,
    a_matrix: Optional[List[List[float]]] = None,
    b_matrix: Optional[List[List[float]]] = None
) -> VARResult:
    """
    结构向量自回归(SVAR)模型实现
    
    Args:
        data: 多元时间序列数据
        lags: 滞后期数
        variables: 变量名称列表
        a_matrix: A约束矩阵
        b_matrix: B约束矩阵
        
    Returns:
        VARResult: SVAR模型结果
    """
    try:
        from statsmodels.tsa.vector_ar.svar_model import SVAR
        import pandas as pd
        import numpy as np
        
        # 输入验证
        if not data:
            raise ValueError("数据不能为空")
            
        # 转换数据格式
        data_array = np.array(data, dtype=np.float64).T  # 转置以匹配SVAR模型要求的格式
        
        # 检查数据有效性
        if np.isnan(data_array).any():
            raise ValueError("数据中包含缺失值(NaN)")
            
        if np.isinf(data_array).any():
            raise ValueError("数据中包含无穷大值")
        
        # 创建变量名
        if variables is None:
            variables = [f"Variable_{i}" for i in range(len(data))]
        
        # 检查变量数量是否与数据一致
        if len(variables) != len(data):
            raise ValueError("变量名称数量与数据列数不一致")
        
        # 创建DataFrame
        df = pd.DataFrame(data_array, columns=variables)
        
        # 检查滞后期数是否合理
        if lags <= 0:
            raise ValueError("滞后期数必须为正整数")
            
        if lags >= len(df):
            raise ValueError("滞后期数必须小于样本数量")
        
        # 处理约束矩阵
        A = np.array(a_matrix, dtype=np.float64) if a_matrix is not None else None
        B = np.array(b_matrix, dtype=np.float64) if b_matrix is not None else None
        
        # 检查约束矩阵维度
        if A is not None and A.shape != (len(variables), len(variables)):
            raise ValueError(f"A矩阵维度不正确，应为({len(variables)}, {len(variables)})")
            
        if B is not None and B.shape != (len(variables), len(variables)):
            raise ValueError(f"B矩阵维度不正确，应为({len(variables)}, {len(variables)})")
        
        # 创建并拟合SVAR模型
        model = SVAR(df, svar_type='AB', A=A, B=B)
        fitted_model = model.fit(lags, maxiter=1000)
        
        # 提取参数估计结果
        # SVAR模型的系数是矩阵形式，需要展平
        coeffs = []
        std_errors = []
        t_values = []
        p_values = []
        
        for i in range(len(variables)):
            # 对于每个方程
            coeffs.extend(fitted_model.coefs[:, :, i].flatten().tolist())
            # 安全地提取标准误
            try:
                if fitted_model.stderr is not None:
                    # 检查stderr的维度
                    if hasattr(fitted_model.stderr, 'shape') and len(fitted_model.stderr.shape) == 3:
                        std_errors.extend(fitted_model.stderr[:, :, i].flatten().tolist())
                    elif hasattr(fitted_model.stderr, '__iter__'):
                        # 如果stderr是可迭代的，尝试安全转换
                        for se in fitted_model.stderr:
                            if np.isscalar(se) and np.isfinite(se):
                                std_errors.append(float(se))
            except (IndexError, TypeError, AttributeError, ValueError):
                # 如果无法提取标准误，保持列表为空
                pass
                
            # 安全地提取t值
            try:
                if fitted_model.tvalues is not None:
                    # 检查tvalues的维度
                    if hasattr(fitted_model.tvalues, 'shape') and len(fitted_model.tvalues.shape) == 3:
                        t_values.extend(fitted_model.tvalues[:, :, i].flatten().tolist())
                    elif hasattr(fitted_model.tvalues, '__iter__'):
                        # 如果tvalues是可迭代的，尝试安全转换
                        for tv in fitted_model.tvalues:
                            if np.isscalar(tv) and np.isfinite(tv):
                                t_values.append(float(tv))
            except (IndexError, TypeError, AttributeError, ValueError):
                # 如果无法提取t值，保持列表为空
                pass
                
            # 安全地提取p值
            try:
                if fitted_model.pvalues is not None:
                    # 检查pvalues的维度
                    if hasattr(fitted_model.pvalues, 'shape') and len(fitted_model.pvalues.shape) == 3:
                        p_values.extend(fitted_model.pvalues[:, :, i].flatten().tolist())
                    elif hasattr(fitted_model.pvalues, '__iter__'):
                        # 如果pvalues是可迭代的，尝试安全转换
                        for pv in fitted_model.pvalues:
                            if np.isscalar(pv) and np.isfinite(pv):
                                p_values.append(float(pv))
            except (IndexError, TypeError, AttributeError, ValueError):
                # 如果无法提取p值，保持列表为空
                pass
        
        # 获取信息准则
        aic = float(fitted_model.aic) if hasattr(fitted_model, 'aic') else None
        bic = float(fitted_model.bic) if hasattr(fitted_model, 'bic') else None
        fpe = float(fitted_model.fpe) if hasattr(fitted_model, 'fpe') else None
        hqic = float(fitted_model.hqic) if hasattr(fitted_model, 'hqic') else None
        
        # 计算脉冲响应函数 (前10期)
        irf_result = fitted_model.irf(10)
        irf = irf_result.irfs.flatten().tolist() if irf_result.irfs is not None else None
        
        # 计算方差分解 (前10期)
        fevd_result = fitted_model.fevd(10)
        fevd = fevd_result.decomp.flatten().tolist() if fevd_result.decomp is not None else None
        
        return VARResult(
            model_type=f"SVAR({lags})",
            lags=lags,
            variables=variables,
            coefficients=coeffs,
            std_errors=std_errors if std_errors else None,
            t_values=t_values if t_values else None,
            p_values=p_values if p_values else None,
            aic=aic,
            bic=bic,
            fpe=fpe,
            hqic=hqic,
            irf=irf,
            fevd=fevd,
            n_obs=len(data[0]) if data else 0
        )
    except Exception as e:
        # 出现错误时抛出异常
        raise ValueError(f"SVAR模型拟合失败: {str(e)}")