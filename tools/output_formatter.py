"""Output formatter for adapter results.

``OutputFormatter`` exposes the stable call shape used by 95+ sites across
the adapter layer: ``format_<name>_result(result, format_type)``. Under the
hood every call routes through a single dispatcher: markdown goes to
``MarkdownFormatter`` (which holds the real table-rendering logic); any other
format falls back to a readable ``str(model.model_dump())``, so passing
``format_type="text"`` no longer silently drops fields.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

# Public API name → MarkdownFormatter method name. Keeping the map explicit
# (rather than generating names from a rule) avoids breaking callers on the
# handful of shortened method names (``format_exp_smoothing`` vs
# ``format_exponential_smoothing_result``).
_MARKDOWN_METHOD: dict[str, str] = {
    "format_ols_result": "format_ols",
    "format_mle_result": "format_mle",
    "format_gmm_result": "format_gmm",
    "format_arima_result": "format_arima",
    "format_exponential_smoothing_result": "format_exp_smoothing",
    "format_garch_result": "format_garch",
    "format_unit_root_test_result": "format_unit_root",
    "format_var_result": "format_var",
    "format_cointegration_result": "format_cointegration",
    "format_vecm_result": "format_vecm",
    "format_dynamic_panel_result": "format_dynamic_panel",
}


def _as_plain(result: Any) -> Any:
    """Best-effort dict conversion for Pydantic v1 and v2 models."""
    for attr in ("model_dump", "dict"):
        converter = getattr(result, attr, None)
        if callable(converter):
            return converter()
    return result


def _dispatcher(markdown_method: str) -> Callable[..., str]:
    def dispatch(result: Any, format_type: str = "markdown") -> str:
        if format_type.lower() == "markdown":
            return getattr(MarkdownFormatter, markdown_method)(result)
        return str(_as_plain(result))

    dispatch.__doc__ = f"Dispatch ``{markdown_method}`` result to the requested format."
    return dispatch


class OutputFormatter:
    """Stable-API facade for markdown/text rendering of adapter results."""

    @staticmethod
    def save_to_file(content: str, file_path: str) -> str:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Result saved to: {file_path}"


for _public, _private in _MARKDOWN_METHOD.items():
    setattr(OutputFormatter, _public, staticmethod(_dispatcher(_private)))


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
        md += f"表示模型解释了因变量 {result.r_squared * 100:.2f}% 的变异。\n"
        md += f"- F统计量为 {result.f_statistic:.4f}，p值为 {result.f_p_value:.4f}，"
        md += "模型整体显著。\n" if result.f_p_value < 0.05 else "模型整体不显著。\n"

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
- **收敛状态**: {"已收敛" if result.convergence else "未收敛"}

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

    @staticmethod
    def format_arima(result: Any) -> str:
        """格式化ARIMA结果为Markdown"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = f"""# ARIMA模型分析结果

**生成时间**: {timestamp}

## 模型信息

- **模型类型**: {result.model_type}
- **观测数量**: {result.n_obs}
- **AIC**: {result.aic:.4f}
- **BIC**: {result.bic:.4f}
- **HQIC**: {result.hqic:.4f}

## 参数估计

| 参数 | 系数 | 标准误 | t值 | p值 | 95%置信区间下限 | 95%置信区间上限 |
|------|------|--------|-----|-----|----------------|----------------|
"""

        for i in range(len(result.coefficients)):
            md += f"| 参数{i + 1} | {result.coefficients[i]:.6f} | "
            md += f"{result.std_errors[i] if result.std_errors else 0:.6f} | "
            md += f"{result.t_values[i] if result.t_values else 0:.4f} | "
            md += f"{result.p_values[i] if result.p_values else 0:.4f} | "
            md += f"{result.conf_int_lower[i] if result.conf_int_lower else 0:.6f} | "
            md += f"{result.conf_int_upper[i] if result.conf_int_upper else 0:.6f} |\n"

        if hasattr(result, "forecast") and result.forecast:
            md += "\n## 预测值\n\n| 步骤 | 预测值 |\n|------|--------|\n"
            for i, val in enumerate(result.forecast):
                md += f"| {i + 1} | {val:.4f} |\n"

        return md

    @staticmethod
    def format_exp_smoothing(result: Any) -> str:
        """格式化指数平滑结果为Markdown"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = f"""# 指数平滑模型分析结果

**生成时间**: {timestamp}

## 模型信息

- **模型类型**: {result.model_type}
- **观测数量**: {result.n_obs}
"""

        if result.smoothing_level is not None:
            md += f"- **水平平滑参数**: {result.smoothing_level:.4f}\n"
        if result.smoothing_trend is not None:
            md += f"- **趋势平滑参数**: {result.smoothing_trend:.4f}\n"
        if result.smoothing_seasonal is not None:
            md += f"- **季节平滑参数**: {result.smoothing_seasonal:.4f}\n"

        md += "\n## 模型统计\n\n"
        if result.aic is not None:
            md += f"- **AIC**: {result.aic:.4f}\n"
        if result.bic is not None:
            md += f"- **BIC**: {result.bic:.4f}\n"
        if result.sse is not None:
            md += f"- **SSE**: {result.sse:.4f}\n"
        if result.mse is not None:
            md += f"- **MSE**: {result.mse:.4f}\n"
        if result.rmse is not None:
            md += f"- **RMSE**: {result.rmse:.4f}\n"
        if result.mae is not None:
            md += f"- **MAE**: {result.mae:.4f}\n"

        if hasattr(result, "forecast") and result.forecast:
            md += "\n## 预测值\n\n| 步骤 | 预测值 |\n|------|--------|\n"
            for i, val in enumerate(result.forecast):
                md += f"| {i + 1} | {val:.4f} |\n"

        return md

    @staticmethod
    def format_garch(result: Any) -> str:
        """格式化GARCH结果为Markdown"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = f"""# GARCH模型分析结果

**生成时间**: {timestamp}

## 模型信息

- **模型类型**: {result.model_type}
- **观测数量**: {result.n_obs}
- **对数似然值**: {result.log_likelihood:.4f}
- **AIC**: {result.aic:.4f}
- **BIC**: {result.bic:.4f}

## 参数估计

| 参数 | 系数 | 标准误 | t值 | p值 |
|------|------|--------|-----|-----|
"""

        for i in range(len(result.coefficients)):
            md += f"| 参数{i + 1} | {result.coefficients[i]:.6f} | "
            md += f"{result.std_errors[i] if result.std_errors else 0:.6f} | "
            md += f"{result.t_values[i] if result.t_values else 0:.4f} | "
            md += f"{result.p_values[i] if result.p_values else 0:.4f} |\n"

        if result.persistence is not None:
            md += f"\n## 波动率持续性\n\n- **持续性参数**: {result.persistence:.4f}\n"

        if hasattr(result, "volatility") and result.volatility:
            md += "\n## 波动率统计\n\n"
            md += f"- **平均波动率**: {sum(result.volatility) / len(result.volatility):.6f}\n"
            md += f"- **最大波动率**: {max(result.volatility):.6f}\n"
            md += f"- **最小波动率**: {min(result.volatility):.6f}\n"

        return md

    @staticmethod
    def format_unit_root(result: Any) -> str:
        """格式化单位根检验结果为Markdown"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = f"""# 单位根检验结果

**生成时间**: {timestamp}

## 检验信息

- **检验类型**: {result.test_type}
- **观测数量**: {result.n_obs}
- **检验统计量**: {result.test_statistic:.4f}
- **p值**: {result.p_value:.4f}
- **是否平稳**: {"是" if result.stationary else "否"}
"""

        if result.lags is not None:
            md += f"- **滞后阶数**: {result.lags}\n"

        if result.critical_values:
            md += "\n## 临界值\n\n"
            for key, value in result.critical_values.items():
                md += f"- **{key}**: {value:.4f}\n"

        md += "\n## 解释\n\n"
        if result.stationary:
            md += "- 序列是平稳的，可以直接进行时间序列分析\n"
        else:
            md += "- 序列是非平稳的，建议进行差分或其他转换\n"

        return md

    @staticmethod
    def format_var(result: Any) -> str:
        """格式化VAR/SVAR结果为Markdown"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = f"""# {result.model_type}模型分析结果

**生成时间**: {timestamp}

## 模型信息

- **变量**: {", ".join(result.variables)}
- **滞后期数**: {result.lags}
- **观测数量**: {result.n_obs}

## 参数估计

| 参数 | 系数 | 标准误 | t值 | p值 |
|------|------|--------|-----|-----|
"""

        for i in range(len(result.coefficients)):
            md += f"| 参数{i + 1} | {result.coefficients[i]:.6f} | "
            md += f"{result.std_errors[i] if result.std_errors else 0:.6f} | "
            md += f"{result.t_values[i] if result.t_values else 0:.4f} | "
            md += f"{result.p_values[i] if result.p_values else 0:.4f} |\n"

        md += "\n## 模型统计\n\n"
        if result.aic is not None:
            md += f"- **AIC**: {result.aic:.4f}\n"
        if result.bic is not None:
            md += f"- **BIC**: {result.bic:.4f}\n"
        if result.fpe is not None:
            md += f"- **FPE**: {result.fpe:.4f}\n"
        if result.hqic is not None:
            md += f"- **HQIC**: {result.hqic:.4f}\n"

        return md

    @staticmethod
    def format_cointegration(result: Any) -> str:
        """格式化协整检验结果为Markdown"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = f"""# 协整检验结果

**生成时间**: {timestamp}

## 检验信息

- **检验类型**: {result.model_type}
- **观测数量**: {result.n_obs}
- **检验统计量**: {result.test_statistic:.4f}
- **p值**: {result.p_value:.4f}
"""

        if result.rank is not None:
            md += f"- **协整秩**: {result.rank}\n"

        if result.critical_values:
            md += "\n## 临界值\n\n"
            for key, value in result.critical_values.items():
                md += f"- **{key}**: {value:.4f}\n"

        return md

    @staticmethod
    def format_vecm(result: Any) -> str:
        """格式化VECM结果为Markdown"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = f"""# VECM模型分析结果

**生成时间**: {timestamp}

## 模型信息

- **协整秩**: {result.coint_rank}
- **观测数量**: {result.n_obs}

## 参数估计

| 参数 | 系数 | 标准误 | t值 | p值 |
|------|------|--------|-----|-----|
"""

        for i in range(len(result.coefficients)):
            md += f"| 参数{i + 1} | {result.coefficients[i]:.6f} | "
            md += f"{result.std_errors[i] if result.std_errors else 0:.6f} | "
            md += f"{result.t_values[i] if result.t_values else 0:.4f} | "
            md += f"{result.p_values[i] if result.p_values else 0:.4f} |\n"

        md += "\n## 模型统计\n\n"
        if result.log_likelihood is not None:
            md += f"- **对数似然值**: {result.log_likelihood:.4f}\n"
        if result.aic is not None:
            md += f"- **AIC**: {result.aic:.4f}\n"
        if result.bic is not None:
            md += f"- **BIC**: {result.bic:.4f}\n"

        return md

    @staticmethod
    def format_dynamic_panel(result: Any) -> str:
        """格式化动态面板模型结果为Markdown"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = f"""# 动态面板模型分析结果

**生成时间**: {timestamp}

## 模型信息

- **模型类型**: {result.model_type}
- **观测数量**: {result.n_obs}
- **个体数量**: {result.n_individuals}
- **时间期数**: {result.n_time_periods}

## 参数估计

| 参数 | 系数 | 标准误 | t值 | p值 |
|------|------|--------|-----|-----|
"""

        for i in range(len(result.coefficients)):
            md += f"| 参数{i + 1} | {result.coefficients[i]:.6f} | "
            md += f"{result.std_errors[i] if result.std_errors else 0:.6f} | "
            md += f"{result.t_values[i] if result.t_values else 0:.4f} | "
            md += f"{result.p_values[i] if result.p_values else 0:.4f} |\n"

        md += "\n## 模型统计\n\n"
        if result.instruments is not None:
            md += f"- **工具变量数量**: {result.instruments}\n"
        if result.j_statistic is not None:
            md += f"- **J统计量**: {result.j_statistic:.4f}\n"
        if result.j_p_value is not None:
            md += f"- **J统计量p值**: {result.j_p_value:.4f}\n"

        return md
