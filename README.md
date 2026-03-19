# aigroup-econ-mcp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-2.0.8-brightgreen.svg)](https://github.com/jackdark425/aigroup-econ-mcp)
[![Tools](https://img.shields.io/badge/Tools-66-brightgreen.svg)](https://github.com/jackdark425/aigroup-econ-mcp)

> Econometrics MCP server for regression, causal inference, time series, panel data, machine learning, and broader statistical analysis workflows.

## Overview

`aigroup-econ-mcp` is a professional econometrics-oriented MCP server designed to help AI assistants and MCP clients perform structured quantitative analysis.

It covers:

- parameter estimation and regression analysis
- causal inference workflows
- microeconometrics and panel data
- time series and volatility models
- machine learning for econometric tasks
- spatial econometrics, decomposition, and inference tools

## Highlights

- **66 professional tools** across core econometrics domains
- **Multiple input formats** including CSV, JSON, TXT, and Excel
- **Multiple output formats** including JSON, Markdown, HTML, LaTeX, and text
- **Support for MCP clients** such as RooCode, Claude-compatible tools, and other MCP hosts
- **Broad method coverage** from OLS and IV to ARIMA, GARCH, GAM, and causal forests
- **Designed for research and applied analysis** rather than narrow single-task workflows

## Tool Groups

The server currently groups its 66 tools across the following categories:

- **Basic parametric estimation** — OLS, MLE, GMM
- **Causal inference** — DID, IV, PSM, fixed/random effects, RDD, synthetic control, event study, and more
- **Decomposition analysis** — Oaxaca-Blinder, ANOVA, time-series decomposition
- **Machine learning** — random forest, gradient boosting, SVM, neural networks, clustering, DML, causal forest
- **Microeconometrics** — logit, probit, multinomial logit, Poisson, negative binomial, Tobit, Heckman
- **Missing data handling** — simple imputation and MICE
- **Model diagnostics and robust inference** — specification tests, GLS, WLS, robust errors, regularization, simultaneous equations
- **Nonparametric methods** — kernel regression, quantile regression, spline regression, GAM
- **Spatial econometrics** — weights matrices, Moran's I, Geary's C, LISA, spatial regression, GWR
- **Statistical inference** — bootstrap and permutation tests
- **Time series and panel data** — ARIMA, exponential smoothing, GARCH, unit-root tests, VAR/SVAR, cointegration, dynamic panel, panel VAR, structural breaks, time-varying parameter models

## Quick Start

### Requirements

- Python >= 3.10
- `uvx` recommended for easiest usage, or `pip`

### Run with uvx

```bash
uvx aigroup-econ-mcp
```

If `uvx` keeps using an older cached build:

```bash
uvx --no-cache aigroup-econ-mcp
```

### Install with pip

```bash
pip install aigroup-econ-mcp
aigroup-econ-mcp
```

## MCP Client Configuration

### Claude-compatible MCP clients / RooCode / similar tools

```json
{
  "mcpServers": {
    "aigroup-econ-mcp": {
      "command": "uvx",
      "args": ["aigroup-econ-mcp"]
    }
  }
}
```

## Input & Output Support

### Supported input formats

- CSV
- JSON
- TXT
- Excel (`.xlsx`, `.xls`)

Typical usage patterns:

- direct structured data input
- raw file content input
- local file path input

### Supported output formats

- `json`
- `markdown`
- `html`
- `latex`
- `text`

## Example Use Cases

- OLS and generalized regression modeling
- difference-in-differences and instrumental variable analysis
- matching and regression discontinuity workflows
- random forest / gradient boosting / causal forest analysis
- ARIMA, GARCH, VAR, and cointegration modeling
- panel diagnostics and dynamic panel estimation

## Project Structure

```text
aigroup-econ-mcp/
├── econometrics/
├── tools/
├── resources/
├── prompts/
├── cli.py
├── server.py
└── pyproject.toml
```

## Development

```bash
uv sync
uv run pytest
```

Useful development commands:

```bash
uv run black .
uv run isort .
```

## Troubleshooting

### uvx cache issue

If a newer published version does not seem to load, try one of the following:

```bash
uvx --no-cache aigroup-econ-mcp
uv cache clean
```

The repository also includes helper scripts such as:

- `clear_uvx_cache.bat`
- `clear_uvx_cache.sh`
- `clear_uvx_cache.py`

## License & Usage

This project is released under the **MIT License**.

You may use, copy, modify, merge, publish, distribute, sublicense, and sell copies of this software, including in academic, research, internal, and commercial environments, provided that the original copyright notice and license text are preserved.

Please keep in mind:

- the software is provided **"AS IS"**, without warranty of any kind
- you must retain the relevant copyright and permission notice in copies or substantial portions of the software
- statistical results still depend on data quality, assumptions, and correct methodological choices by the user

See the full text in [LICENSE](LICENSE).

## Acknowledgments

### Core Scientific Ecosystem

- **statsmodels** — statistical modeling foundations
- **pandas** — data manipulation and tabular workflows
- **scikit-learn** — machine learning components
- **linearmodels** — panel data and econometric modeling support
- **arch** — volatility and ARCH/GARCH modeling

### Community & Protocol Ecosystem

- **Model Context Protocol** — MCP integration model
- The broader econometrics and open-source scientific computing community

## Support

- Issues: https://github.com/jackdark425/aigroup-econ-mcp/issues
- Repository: https://github.com/jackdark425/aigroup-econ-mcp
- PyPI publishing guide: [PYPI_PUBLISH_GUIDE.md](PYPI_PUBLISH_GUIDE.md)
