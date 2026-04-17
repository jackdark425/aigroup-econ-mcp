"""Data loaders for adapter inputs.

Two shapes are supported:

* **Structured** data (:meth:`DataLoader.load_from_file`) — the typical case
  for regression, causal, time-series-panel adapters. Returns
  ``{"y_data": [...], "x_data": [[...]], "feature_names": [...]}``.
* **Flat 1-D** data (:meth:`DataLoader.load_flat`) — a single numeric column,
  used by MLE and other univariate tools. Returns ``{"data": [...]}``.

Both accept ``.txt``, ``.json``, ``.csv``, ``.xlsx``/``.xls``. The loaders are
stateless ``@staticmethod``s so adapters can call them without instantiation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_SUFFIXES = frozenset({".txt", ".json", ".csv", ".xlsx", ".xls"})


class DataLoader:
    """Load structured tabular data (first column y, remaining columns X)."""

    @staticmethod
    def load_from_file(file_path: str) -> dict[str, Any]:
        path = _validated_path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            return DataLoader._parse_matrix(_read_numeric_matrix(path))
        if suffix == ".json":
            return DataLoader._from_json(path)
        if suffix == ".csv":
            return DataLoader._from_dataframe(pd.read_csv(path))
        if suffix in {".xlsx", ".xls"}:
            return DataLoader._from_dataframe(pd.read_excel(path))
        raise ValueError(f"unsupported file format: {suffix!r}")

    @staticmethod
    def load_flat(file_path: str) -> dict[str, Any]:
        """Load a single numeric column. Used by MLE and univariate tools."""
        path = _validated_path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            return {"data": _read_flat_floats(path)}
        if suffix == ".json":
            with path.open(encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                return {"data": loaded}
            if isinstance(loaded, dict) and "data" in loaded:
                return {"data": loaded["data"]}
            raise ValueError("JSON must be a list or an object with a 'data' key")
        if suffix == ".csv":
            return {"data": pd.read_csv(path).iloc[:, 0].tolist()}
        if suffix in {".xlsx", ".xls"}:
            return {"data": pd.read_excel(path).iloc[:, 0].tolist()}
        raise ValueError(f"unsupported file format: {suffix!r}")

    # --- internal helpers ----------------------------------------------------

    @staticmethod
    def _from_json(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if "y_data" in data and "x_data" in data:
            return {
                "y_data": data["y_data"],
                "x_data": data["x_data"],
                "feature_names": data.get("feature_names"),
            }
        if "data" in data:
            return DataLoader._parse_matrix(data["data"])
        raise ValueError("JSON must contain either ('y_data' and 'x_data') or 'data'")

    @staticmethod
    def _from_dataframe(df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            raise ValueError("data frame is empty")
        if df.shape[1] < 2:
            raise ValueError("data must have a dependent column and at least one regressor")
        return {
            "y_data": df.iloc[:, 0].tolist(),
            "x_data": df.iloc[:, 1:].values.tolist(),
            "feature_names": df.columns[1:].tolist(),
        }

    @staticmethod
    def _parse_matrix(data: list[list[float]]) -> dict[str, Any]:
        if not data:
            raise ValueError("data matrix is empty")
        if len(data[0]) < 2:
            raise ValueError("data must have a dependent column and at least one regressor")
        return {
            "y_data": [row[0] for row in data],
            "x_data": [row[1:] for row in data],
            "feature_names": [f"X{i + 1}" for i in range(len(data[0]) - 1)],
        }


def merge_file_data(file_path: str | None, **defaults: Any) -> dict[str, Any]:
    """Overlay file data onto caller-supplied defaults.

    Adapter functions uniformly accept both direct-arg and file-path inputs:

        if file_path:
            d = DataLoader.load_from_file(file_path)
            y_data = d.get("y_data", y_data)
            x_data = d.get("x_data", x_data)

    This helper collapses the pattern to one call and keeps the "file wins
    when present, otherwise keep caller default" semantics:

        merged = merge_file_data(file_path, y_data=y_data, x_data=x_data)
        y_data, x_data = merged["y_data"], merged["x_data"]

    When ``file_path`` is ``None`` or empty the defaults are returned as-is —
    no file I/O occurs.
    """
    if not file_path:
        return defaults
    loaded = DataLoader.load_from_file(file_path)
    return {key: loaded.get(key, default) for key, default in defaults.items()}


# Backwards-compat alias — a single external import point exists in
# ``econometrics_adapter`` and may also exist in downstream code. The thin
# shim keeps the name available without duplicating logic.
class MLEDataLoader:
    """Deprecated. Use :meth:`DataLoader.load_flat` directly."""

    @staticmethod
    def load_from_file(file_path: str) -> dict[str, Any]:
        return DataLoader.load_flat(file_path)


def _validated_path(file_path: str) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")
    return path


def _read_numeric_matrix(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()])
    if not rows:
        raise ValueError("file contains no numeric rows")
    return rows


def _read_flat_floats(path: Path) -> list[float]:
    values: list[float] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                values.append(float(line.split()[0]))
    return values
