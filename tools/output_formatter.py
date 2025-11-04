"""
输出格式化组件 - 支持Markdown和TXT格式
"""

from typing import Any, Dict, List
from datetime import datetime
from pathlib import Path


class OutputFormatter:
    """输出格式化器基类"""
    
    @staticmethod
    def format_ols_result(result: Any, format_type: str = "markdown") -> str:
        """
        格式化OLS结果
        
        Args:
            result: OLS结果对象
            format_type: 输出格式 ("markdown" 或 "txt")
            
        Returns:
            格式化的字符串
        """
        if format_type.lower() == "markdown":
            return MarkdownFormatter.format_ols(result)
        else:
            return TextFormatter.format_ols(result)
    
    @staticmethod
    def format_mle_result(result: Any, format_type: str = "markdown") -> str:
        """格式化MLE结果"""
        if format_type.lower() == "markdown":
            return MarkdownFormatter.format_mle(result)
        else:
            return TextFormatter.format_mle(result)
    
    @staticmethod
    def format_gmm_result(result: Any, format_type: str = "markdown") -> str:
        """格式化GMM结果"""
        if format_type.lower() == "markdown":
            return MarkdownFormatter.format_gmm(result)
        else:
            return TextFormatter.format_gmm(result)
    
    @staticmethod
    def save_to_file(content: str, file_path: str) -> str:
        """
        保存内容到文件
        
        Args:
            content: 要保存的内容
            file_path: 文件路径
            
        Returns:
            保存成功的消息
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"结果已保存到: {file_path}"


class MarkdownFormatter:
    """Markdown格式化器"""
    
    @staticmethod
    def format_ols(result: Any) -> str:
        """格式化OLS结果为Markdown"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md = f"""# OLS回归分析结果

**生成时间**: {timestamp}

## 模型概览

- **观测数量**: {result.n_obs}
- **R²**: {result.r_squared:.4f}
- **调整R²**: {result.adj_r_squared:.4f}
- **F统计量**: {result.f_statistic:.4f}
- **F检验p值**: {result.f_p_value:.4f}

## 系数估计

| 变量 | 系数 | 标准误 | t值 | p值 | 95%置信区间下限 | 95%置信区间上限 |
|------|------|--------|-----|-----|----------------|----------------|
"""
        
        for i, name in enumerate(result.feature_names):
            md += f"| {name} | {result.coefficients[i]:.6f} | {result.std_errors[i]:.6f} | "
            md += f"{result.t_values[i]:.4f} | {result.p_values[i]:.4f} | "
            md += f"{result.conf_int_lower[i]:.6f} | {result.conf_int_upper[i]:.6f} |\n"
        
        md += "\n## 解释\n\n"
        md += f"- 模型的拟合优度R²为 {result.r_squared:.4f}，"
        md += f"表示模型解释了因变量 {result.r_squared*100:.2f}% 的变异。\n"
        md += f"- F统计量为 {result.f_statistic:.4f}，p值为 {result.f_p_value:.4f}，"
        
        if result.f_p_value < 0.05:
            md += "模型整体显著。\n"
        else:
            md += "模型整体不显著。\n"
        
        return md
    
    @staticmethod
    def format_mle(result: Any) -> str:
        """格式化MLE结果为Markdown"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md = f"""# 最大似然估计(MLE)结果

**生成时间**: {timestamp}

## 模型信息

- **观测数量**: {result.n_obs}
- **对数似然值**: {result.log_likelihood:.4f}
- **AIC**: {result.aic:.4f}
- **BIC**: {result.bic:.4f}
- **收敛状态**: {'已收敛' if result.convergence else '未收敛'}

## 参数估计

| 参数 | 估计值 | 标准误 | 95%置信区间下限 | 95%置信区间上限 |
|------|--------|--------|----------------|----------------|
"""
        
        for i, name in enumerate(result.param_names):
            md += f"| {name} | {result.parameters[i]:.6f} | {result.std_errors[i]:.6f} | "
            md += f"{result.conf_int_lower[i]:.6f} | {result.conf_int_upper[i]:.6f} |\n"
        
        md += "\n## 模型选择\n\n"
        md += f"- AIC (赤池信息准则): {result.aic:.4f} - 越小越好\n"
        md += f"- BIC (贝叶斯信息准则): {result.bic:.4f} - 越小越好\n"
        
        return md
    
    @staticmethod
    def format_gmm(result: Any) -> str:
        """格式化GMM结果为Markdown"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md = f"""# 广义矩估计(GMM)结果

**生成时间**: {timestamp}

## 模型信息

- **观测数量**: {result.n_obs}
- **矩条件数量**: {result.n_moments}
- **J统计量**: {result.j_statistic:.4f}
- **J检验p值**: {result.j_p_value:.4f}

## 系数估计

| 变量 | 系数 | 标准误 | t值 | p值 | 95%置信区间下限 | 95%置信区间上限 |
|------|------|--------|-----|-----|----------------|----------------|
"""
        
        for i, name in enumerate(result.feature_names):
            md += f"| {name} | {result.coefficients[i]:.6f} | {result.std_errors[i]:.6f} | "
            md += f"{result.t_values[i]:.4f} | {result.p_values[i]:.4f} | "
            md += f"{result.conf_int_lower[i]:.6f} | {result.conf_int_upper[i]:.6f} |\n"
        
        md += "\n## 过度识别检验\n\n"
        if result.j_p_value < 0.05:
            md += f"- J统计量为 {result.j_statistic:.4f}，p值为 {result.j_p_value:.4f}\n"
            md += "- **警告**: 拒绝过度识别限制的原假设，模型可能存在设定偏误\n"
        else:
            md += f"- J统计量为 {result.j_statistic:.4f}，p值为 {result.j_p_value:.4f}\n"
            md += "- 不能拒绝过度识别限制的原假设，工具变量有效\n"
        
        return md


class TextFormatter:
    """纯文本格式化器"""
    
    @staticmethod
    def format_ols(result: Any) -> str:
        """格式化OLS结果为纯文本"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        txt = f"""OLS回归分析结果
生成时间: {timestamp}
{'='*70}

模型概览
{'='*70}
观测数量: {result.n_obs}
R²: {result.r_squared:.4f}
调整R²: {result.adj_r_squared:.4f}
F统计量: {result.f_statistic:.4f}
F检验p值: {result.f_p_value:.4f}

系数估计
{'='*70}
"""
        
        # 表头
        txt += f"{'变量':<15} {'系数':>12} {'标准误':>12} {'t值':>10} {'p值':>10} "
        txt += f"{'CI下限':>12} {'CI上限':>12}\n"
        txt += "-" * 85 + "\n"
        
        # 数据行
        for i, name in enumerate(result.feature_names):
            txt += f"{name:<15} {result.coefficients[i]:>12.6f} {result.std_errors[i]:>12.6f} "
            txt += f"{result.t_values[i]:>10.4f} {result.p_values[i]:>10.4f} "
            txt += f"{result.conf_int_lower[i]:>12.6f} {result.conf_int_upper[i]:>12.6f}\n"
        
        txt += "\n解释\n"
        txt += "=" * 70 + "\n"
        txt += f"模型的拟合优度R²为 {result.r_squared:.4f}，"
        txt += f"表示模型解释了因变量 {result.r_squared*100:.2f}% 的变异。\n"
        
        return txt
    
    @staticmethod
    def format_mle(result: Any) -> str:
        """格式化MLE结果为纯文本"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        txt = f"""最大似然估计(MLE)结果
生成时间: {timestamp}
{'='*70}

模型信息
{'='*70}
观测数量: {result.n_obs}
对数似然值: {result.log_likelihood:.4f}
AIC: {result.aic:.4f}
BIC: {result.bic:.4f}
收敛状态: {'已收敛' if result.convergence else '未收敛'}

参数估计
{'='*70}
"""
        
        # 表头
        txt += f"{'参数':<15} {'估计值':>12} {'标准误':>12} {'CI下限':>12} {'CI上限':>12}\n"
        txt += "-" * 70 + "\n"
        
        # 数据行
        for i, name in enumerate(result.param_names):
            txt += f"{name:<15} {result.parameters[i]:>12.6f} {result.std_errors[i]:>12.6f} "
            txt += f"{result.conf_int_lower[i]:>12.6f} {result.conf_int_upper[i]:>12.6f}\n"
        
        return txt
    
    @staticmethod
    def format_gmm(result: Any) -> str:
        """格式化GMM结果为纯文本"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        txt = f"""广义矩估计(GMM)结果
生成时间: {timestamp}
{'='*70}

模型信息
{'='*70}
观测数量: {result.n_obs}
矩条件数量: {result.n_moments}
J统计量: {result.j_statistic:.4f}
J检验p值: {result.j_p_value:.4f}

系数估计
{'='*70}
"""
        
        # 表头
        txt += f"{'变量':<15} {'系数':>12} {'标准误':>12} {'t值':>10} {'p值':>10} "
        txt += f"{'CI下限':>12} {'CI上限':>12}\n"
        txt += "-" * 85 + "\n"
        
        # 数据行
        for i, name in enumerate(result.feature_names):
            txt += f"{name:<15} {result.coefficients[i]:>12.6f} {result.std_errors[i]:>12.6f} "
            txt += f"{result.t_values[i]:>10.4f} {result.p_values[i]:>10.4f} "
            txt += f"{result.conf_int_lower[i]:>12.6f} {result.conf_int_upper[i]:>12.6f}\n"
        
        return txt