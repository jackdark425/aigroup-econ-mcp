# aigroup-econ-mcp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-2.0.10-brightgreen.svg)](https://github.com/jackdark425/aigroup-econ-mcp)
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
- **Multiple output formats** including JSON, Markdown, and text
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

### macOS users: install libomp for XGBoost-backed tools

A subset of machine-learning tools links against `xgboost`, which requires the
OpenMP runtime (`libomp.dylib`) at import time. Without it, these four tools
return a `tool_unavailable` payload even though the server itself starts fine:

- `ml_kmeans_clustering`
- `ml_hierarchical_clustering`
- `ml_double_machine_learning`
- `ml_causal_forest`

Install once with Homebrew:

```bash
brew install libomp
```

Linux / Windows users are unaffected — the xgboost wheels bundle or locate
OpenMP automatically.

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

- `json` (default — structured Pydantic result serialized)
- `markdown` (human-readable tables and coefficient stars)
- `text` (compact `str(model.model_dump())` fallback)

## Example Use Cases

- OLS and generalized regression modeling
- difference-in-differences and instrumental variable analysis
- matching and regression discontinuity workflows
- random forest / gradient boosting / causal forest analysis
- ARIMA, GARCH, VAR, and cointegration modeling
- panel diagnostics and dynamic panel estimation

## Calling tools

Every tool accepts parameters either **inline** (direct lists) or **from a
file** via `file_path`. The server returns a JSON string; on failure the
payload has a uniform `{"ok": false, "error": {...}}` shape.

### Inline: OLS regression

```json
{
  "name": "basic_parametric_estimation_ols",
  "arguments": {
    "y_data": [1.0, 2.1, 2.9, 4.1, 5.0, 5.9, 7.1, 8.0, 8.9, 10.1],
    "x_data": [[1.0], [2.0], [3.0], [4.0], [5.0],
               [6.0], [7.0], [8.0], [9.0], [10.0]],
    "output_format": "json"
  }
}
```

### File-based: causal DID from a CSV

Your CSV first column is the dependent variable (`y_data`), the rest are
covariates (`x_data`). For domain-specific keys like `treatment`,
`time_period`, `outcome`, supply a `.json` file:

```json
// policy.json
{"treatment": [0,0,1,1,0,0,1,1],
 "time_period": [0,1,0,1,0,1,0,1],
 "outcome":    [4.1,4.8,3.9,5.6,4.0,4.7,3.8,5.5]}
```

```json
{
  "name": "causal_difference_in_differences",
  "arguments": {"file_path": "policy.json", "output_format": "markdown"}
}
```

### Time series: ARIMA with forecast

```json
{
  "name": "time_series_arima_model",
  "arguments": {
    "data": [/* monthly observations */],
    "order": [1, 1, 1],
    "forecast_steps": 6,
    "output_format": "markdown"
  }
}
```

### Interpreting `fit_warnings`

Several models return a `fit_warnings` array. A non-empty value means the
statistic was a fallback placeholder — treat the associated numbers with
care. For example, a Cox regression with singular Hessian:

```json
{
  "coefficients": [...],
  "std_errors": [1.0, 1.0, 1.0],
  "p_values":   [1.0, 1.0, 1.0],
  "fit_warnings": [
    "Hessian inversion failed; std_errors are placeholder 1.0 — Z/p values below are not real"
  ]
}
```

Seeing `p=1.0` with no warning means "not significant"; the same with a
warning means "could not compute".

### Debug mode

Set `AIGROUP_ECON_MCP_DEBUG=1` before launching the server to include
Python tracebacks inside the structured error payload — useful when
developing new MCP clients.

## Project Structure

```text
aigroup-econ-mcp/
├── aigroup_econ_mcp/       # MCP server + CLI
│   ├── cli.py              #   argparse entry point
│   ├── server.py           #   FastMCP wire-up
│   ├── registry.py         #   ToolSpec registry
│   ├── _registrations.py   #   all 66 tools registered here
│   └── errors.py
├── tools/                  # adapter layer (I/O + formatting)
├── econometrics/           # algorithms
├── resources/
├── prompts/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PUBLISHING.md
│   └── TESTING.md
├── tests/
├── CHANGELOG.md
└── pyproject.toml
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for layer boundaries and how
to add a new tool.

## Development

```bash
uv sync
uv run pytest           # full suite, ~12 s (271 tests)
uv run pytest -m "not slow"   # fast iteration, ~1.5 s (264 tests)
uv run ruff check .
uv run ruff format .
```

All 66 registered tools are covered by a four-tier test pyramid:
registration-shape → smoke → mathematical-correctness (known-DGP) →
real MCP stdio protocol. See [`docs/VERIFICATION.md`](docs/VERIFICATION.md)
for the testing paradigm, ground-truth DGPs, tolerance rationale, and
the record of real bugs this approach has surfaced (9 to date). See
[`docs/TESTING.md`](docs/TESTING.md) for quick-run commands and the
test-file layout.

## Troubleshooting

### uvx resolves an old version

`uvx` caches per-version, so if a published release is not picked up:

```bash
uvx --refresh aigroup-econ-mcp
# or
uv cache clean
```

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
