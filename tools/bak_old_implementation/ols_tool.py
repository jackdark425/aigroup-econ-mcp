"""
OLS回归工具实现
基于现有计量经济学代码重构为MCP工具
支持文件输入和多种输出格式
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from scipy import stats
from .data_loader import DataLoader
from .output_formatter import OutputFormatter


class OLSResult:
    """OLS回归结果"""
    
    def __init__(
        self,
        coefficients: List[float],
        std_errors: List[float],
        t_values: List[float],
        p_values: List[float],
        conf_int_lower: List[float],
        conf_int_upper: List[float],
        r_squared: float,
        adj_r_squared: float,
        f_statistic: float,
        f_p_value: float,
        n_obs: int,
        feature_names: List[str]
    ):
        self.coefficients = coefficients
        self.std_errors = std_errors
        self.t_values = t_values
        self.p_values = p_values
        self.conf_int_lower = conf_int_lower
        self.conf_int_upper = conf_int_upper
        self.r_squared = r_squared
        self.adj_r_squared = adj_r_squared
        self.f_statistic = f_statistic
        self.f_p_value = f_p_value
        self.n_obs = n_obs
        self.feature_names = feature_names
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "coefficients": self.coefficients,
            "std_errors": self.std_errors,
            "t_values": self.t_values,
            "p_values": self.p_values,
            "conf_int_lower": self.conf_int_lower,
            "conf_int_upper": self.conf_int_upper,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "f_statistic": self.f_statistic,
            "f_p_value": self.f_p_value,
            "n_obs": self.n_obs,
            "feature_names": self.feature_names
        }


def ols_regression_from_file(
    file_path: str,
    constant: bool = True,
    confidence_level: float = 0.95,
    output_format: str = "markdown",
    save_path: Optional[str] = None
) -> Union[OLSResult, str]:
    """
    从文件执行OLS回归（支持txt/json/csv/excel）
    
    Args:
        file_path: 数据文件路径
        constant: 是否包含常数项
        confidence_level: 置信水平
        output_format: 输出格式 ("markdown" 或 "txt")
        save_path: 可选的保存路径
        
    Returns:
        如果指定了save_path，返回格式化的字符串；否则返回OLSResult对象
    """
    # 从文件加载数据
    data = DataLoader.load_from_file(file_path)
    
    # 执行OLS回归
    result = ols_regression(
        y_data=data["y_data"],
        x_data=data["x_data"],
        feature_names=data.get("feature_names"),
        constant=constant,
        confidence_level=confidence_level
    )
    
    # 格式化输出
    formatted_output = OutputFormatter.format_ols_result(result, output_format)
    
    # 如果指定了保存路径，保存并返回消息
    if save_path:
        OutputFormatter.save_to_file(formatted_output, save_path)
        return f"分析完成！\n\n{formatted_output}\n\n{OutputFormatter.save_to_file(formatted_output, save_path)}"
    
    return formatted_output


def ols_regression(
    y_data: List[float],
    x_data: List[List[float]],
    feature_names: Optional[List[str]] = None,
    constant: bool = True,
    confidence_level: float = 0.95
) -> OLSResult:
    """
    普通最小二乘法回归（直接数据输入）
    
    Args:
        y_data: 因变量数据
        x_data: 自变量数据
        feature_names: 特征名称
        constant: 是否包含常数项
        confidence_level: 置信水平
        
    Returns:
        OLSResult: OLS回归结果
    """
    # 转换为numpy数组
    y = np.array(y_data)
    X = np.array(x_data)
    
    # 添加常数项
    if constant:
        X = np.column_stack([np.ones(len(X)), X])
        if feature_names:
            feature_names = ["const"] + feature_names
        else:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    else:
        if not feature_names:
            feature_names = [f"x{i}" for i in range(X.shape[1])]
    
    # 执行OLS回归: β = (X'X)^(-1)X'y
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    
    # 计算预测值和残差
    y_pred = X @ beta
    residuals = y - y_pred
    
    # 计算各种统计量
    n = len(y)
    k = len(beta)
    df_resid = n - k
    
    # 残差平方和
    ssr = np.sum(residuals ** 2)
    
    # 总平方和
    sst = np.sum((y - np.mean(y)) ** 2)
    
    # 回归平方和
    ssr_reg = sst - ssr
    
    # 均方误差
    mse = ssr / df_resid
    
    # R方和调整R方
    r_squared = 1 - (ssr / sst)
    adj_r_squared = 1 - ((ssr / df_resid) / (sst / (n - 1)))
    
    # 系数标准误
    var_beta = mse * XtX_inv
    std_errors = np.sqrt(np.diag(var_beta))
    
    # t统计量和p值
    t_values = beta / std_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df_resid))
    
    # 置信区间
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, df_resid)
    conf_int_lower = beta - t_critical * std_errors
    conf_int_upper = beta + t_critical * std_errors
    
    # F统计量
    f_statistic = (ssr_reg / (k - 1)) / mse
    f_p_value = 1 - stats.f.cdf(f_statistic, k - 1, df_resid)
    
    return OLSResult(
        coefficients=beta.tolist(),
        std_errors=std_errors.tolist(),
        t_values=t_values.tolist(),
        p_values=p_values.tolist(),
        conf_int_lower=conf_int_lower.tolist(),
        conf_int_upper=conf_int_upper.tolist(),
        r_squared=float(r_squared),
        adj_r_squared=float(adj_r_squared),
        f_statistic=float(f_statistic),
        f_p_value=float(f_p_value),
        n_obs=n,
        feature_names=feature_names
    )