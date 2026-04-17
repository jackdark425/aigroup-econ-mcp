"""Wheel-level packaging regression guard (slow).

:func:`tests.test_manifest.test_every_string_target_resolves_to_a_callable`
runs against the *source tree*, so it cannot catch a class of bugs where a
business module is silently dropped from the built wheel — e.g. the 2.0.9
release where ``pyproject.toml``'s ``exclude = "**/*_test.py"`` accidentally
stripped :mod:`econometrics.statistical_inference.permutation_test` and
:mod:`econometrics.causal_inference.causal_identification_strategy.hausman_test`
and broke 15 of 66 tools in a published-to-PyPI artifact.

This test builds a wheel from the current tree with ``python -m build`` and
inspects the wheel's ``RECORD`` file (the authoritative list of installed
paths) to confirm every adapter target and every imported ``econometrics.*``
submodule is present. The regression was always a *file-shipping* bug, never
a runtime bug, so file-shipping is what we verify — no full dep install.

Marked ``slow`` because ``python -m build`` takes ~10 s.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Modules whose absence from the wheel was the 2.0.9 regression. If either
# of these disappears again, the wheel is broken even if every source-tree
# test still passes.
REQUIRED_ECONOMETRICS_MODULES = [
    "econometrics/statistical_inference/permutation.py",
    "econometrics/causal_inference/causal_identification_strategy/hausman_specification.py",
]


@pytest.fixture(scope="module")
def wheel_paths(tmp_path_factory: pytest.TempPathFactory) -> set[str]:
    """Build the wheel once per test-module run and expose its RECORD paths."""
    dist_dir = tmp_path_factory.mktemp("dist")
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir), str(REPO_ROOT)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    wheels = list(dist_dir.glob("aigroup_econ_mcp-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    with zipfile.ZipFile(wheels[0]) as zf:
        return set(zf.namelist())


@pytest.mark.slow
def test_built_wheel_contains_every_adapter_and_business_module(
    wheel_paths: set[str],
) -> None:
    for package in ("aigroup_econ_mcp", "econometrics", "tools"):
        init = f"{package}/__init__.py"
        assert init in wheel_paths, f"wheel missing {init!r} — top-level package not shipped"

    for module in REQUIRED_ECONOMETRICS_MODULES:
        assert module in wheel_paths, (
            f"wheel is missing {module!r} — this is the 2.0.9 regression pattern "
            f"(pyproject exclude eating business files). Wheel contents:\n"
            + "\n".join(sorted(wheel_paths))
        )

    from aigroup_econ_mcp._registrations import _MANIFEST

    missing: list[str] = []
    for _name, _group, target, _desc in _MANIFEST:
        if not isinstance(target, str):
            continue
        module_path, _, _ = target.partition(":")
        wheel_path = module_path.replace(".", "/") + ".py"
        if wheel_path not in wheel_paths:
            missing.append(f"{target!r} — expected wheel entry {wheel_path!r}")

    assert not missing, (
        "manifest targets refer to modules missing from the wheel:\n  "
        + "\n  ".join(missing)
    )


@pytest.mark.slow
def test_built_wheel_has_no_test_files(wheel_paths: set[str]) -> None:
    leaked_tests = [
        p for p in wheel_paths
        if p.startswith(("tests/", "econometrics/tests/"))
        or Path(p).name.startswith("test_")
    ]
    assert not leaked_tests, (
        "wheel shipped test files — release hygiene regression:\n"
        + "\n".join(leaked_tests)
    )
