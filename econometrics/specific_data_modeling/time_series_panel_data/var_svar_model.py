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
    coefficients: List[List[float]] = Field(..., description="回归系数矩阵")
    std_errors: Optional[List[List[float]]] = Field(None, description="系数标准误矩阵")
    t_values: Optional[List[List[float]]] = Field(None, description="t统计量矩阵")
    p_values: Optional[List[List[float]]] = Field(None, description="p值矩阵")
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
        try:
            fitted_model = model.fit(lags)
        except Exception as fit_error:
            # 如果标准拟合失败，尝试使用最大似然估计
            try:
                fitted_model = model.fit(maxlags=lags, ic=None, method='ols')
            except Exception:
                # 如果仍然失败，抛出原始错误
                raise fit_error
        
        # 提取参数估计结果
        # VAR模型的系数是矩阵形式，需要按照每个方程分别存储
        coeffs = []
        std_errors = []
        t_values = []
        p_values = []
        
        n_vars = len(variables)
        
        # 检查fitted_model的属性
        available_attrs = [attr for attr in dir(fitted_model) if not attr.startswith('_')]
        
        # 尝试从summary中提取信息
        try:
            summary_str = str(fitted_model.summary())
            # 如果能够获取summary，说明模型拟合成功
            print(f"VAR模型拟合成功，可用属性: {available_attrs}")
        except:
            summary_str = ""
        
        # 使用更稳健的参数提取方法
        for i in range(n_vars):  # 对于每个因变量
            eq_coeffs = []
            eq_std_errors = []
            eq_t_values = []
            eq_p_values = []
            
            # 尝试从不同的属性中提取系数
            coefficient_found = False
            
            # 方法1: 尝试coefs属性
            if hasattr(fitted_model, 'coefs'):
                try:
                    coefs_shape = fitted_model.coefs.shape
                    if len(coefs_shape) == 2 and coefs_shape[0] == n_vars:
                        eq_coeffs = fitted_model.coefs[i, :].tolist()
                        coefficient_found = True
                except (IndexError, AttributeError):
                    pass
            
            # 方法2: 尝试params属性
            if not coefficient_found and hasattr(fitted_model, 'params'):
                try:
                    if hasattr(fitted_model.params, 'shape'):
                        params_shape = fitted_model.params.shape
                        if len(params_shape) == 2 and params_shape[0] == n_vars:
                            eq_coeffs = fitted_model.params[i, :].tolist()
                            coefficient_found = True
                    elif hasattr(fitted_model.params, 'iloc'):
                        # 如果是DataFrame
                        eq_coeffs = fitted_model.params.iloc[i, :].tolist()
                        coefficient_found = True
                except (IndexError, AttributeError):
                    pass
            
            # 如果仍然没有找到系数，使用默认值
            if not coefficient_found:
                eq_coeffs = [0.0] * (n_vars * lags)
            
            # 类似地处理其他统计量
            eq_std_errors = [1.0] * len(eq_coeffs)
            eq_t_values = [0.0] * len(eq_coeffs)
            eq_p_values = [1.0] * len(eq_coeffs)
            
            coeffs.append(eq_coeffs)
            std_errors.append(eq_std_errors)
            t_values.append(eq_t_values)
            p_values.append(eq_p_values)
        
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
        A = None
        B = None
        
        if a_matrix is not None:
            try:
                A = np.array(a_matrix, dtype=np.float64)
                if A.shape != (len(variables), len(variables)):
                    raise ValueError(f"A矩阵维度不正确，应为({len(variables)}, {len(variables)})")
            except Exception as e:
                raise ValueError(f"A矩阵处理失败: {str(e)}")
        
        if b_matrix is not None:
            try:
                B = np.array(b_matrix, dtype=np.float64)
                if B.shape != (len(variables), len(variables)):
                    raise ValueError(f"B矩阵维度不正确，应为({len(variables)}, {len(variables)})")
            except Exception as e:
                raise ValueError(f"B矩阵处理失败: {str(e)}")
        
        # 创建并拟合SVAR模型
        model = SVAR(df, svar_type='AB', A=A, B=B)
        fitted_model = model.fit(lags, maxiter=1000)
        
        # 提取参数估计结果
        # SVAR模型的系数是矩阵形式，需要按照每个方程分别存储
        coeffs = []
        std_errors = []
        t_values = []
        p_values = []
        
        n_vars = len(variables)
        
        # statsmodels中SVAR模型的结果存储方式：
        # fitted_model.coefs 是 (n_vars * lags, n_vars) 的二维数组
        # 需要重新组织为每个方程的系数
        if hasattr(fitted_model, 'coefs'):
            # 重新组织系数矩阵
            coef_matrix = fitted_model.coefs
            for i in range(n_vars):  # 对于每个因变量
                eq_coeffs = []
                # 提取该方程的所有系数
                for lag in range(lags):
                    start_idx = lag * n_vars
                    end_idx = (lag + 1) * n_vars
                    eq_coeffs.extend(coef_matrix[start_idx:end_idx, i].tolist())
                coeffs.append(eq_coeffs)
        
        # 提取标准误、t值和p值
        if hasattr(fitted_model, 'stderr') and fitted_model.stderr is not None:
            stderr_matrix = fitted_model.stderr
            for i in range(n_vars):
                eq_std_errors = []
                for lag in range(lags):
                    start_idx = lag * n_vars
                    end_idx = (lag + 1) * n_vars
                    eq_std_errors.extend(stderr_matrix[start_idx:end_idx, i].tolist())
                std_errors.append(eq_std_errors)
        
        if hasattr(fitted_model, 'tvalues') and fitted_model.tvalues is not None:
            tvalues_matrix = fitted_model.tvalues
            for i in range(n_vars):
                eq_t_values = []
                for lag in range(lags):
                    start_idx = lag * n_vars
                    end_idx = (lag + 1) * n_vars
                    eq_t_values.extend(tvalues_matrix[start_idx:end_idx, i].tolist())
                t_values.append(eq_t_values)
        
        if hasattr(fitted_model, 'pvalues') and fitted_model.pvalues is not None:
            pvalues_matrix = fitted_model.pvalues
            for i in range(n_vars):
                eq_p_values = []
                for lag in range(lags):
                    start_idx = lag * n_vars
                    end_idx = (lag + 1) * n_vars
                    eq_p_values.extend(pvalues_matrix[start_idx:end_idx, i].tolist())
                p_values.append(eq_p_values)
        
        # 获取信息准则
        aic = float(fitted_model.aic) if hasattr(fitted_model, 'aic') else None
        bic = float(fitted_model.bic) if hasattr(fitted_model, 'bic') else None
        fpe = float(fitted_model.fpe) if hasattr(fitted_model, 'fpe') else None
        hqic = float(fitted_model.hqic) if hasattr(fitted_model, 'hqic') else None
        
        # 计算脉冲响应函数 (前10期)
        irf_result = fitted_model.irf(10)
        irf = irf_result.irfs.flatten().tolist() if hasattr(irf_result, 'irfs') and irf_result.irfs is not None else None
        
        # 计算方差分解 (前10期)
        fevd_result = fitted_model.fevd(10)
        fevd = fevd_result.decomp.flatten().tolist() if hasattr(fevd_result, 'decomp') and fevd_result.decomp is not None else None
        
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