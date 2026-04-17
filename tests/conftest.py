"""Pytest bootstrap.

macOS + xgboost + user-scope Homebrew gotcha: xgboost's
``libxgboost.dylib`` is linked with ``@rpath/libomp.dylib`` resolved
against macOS's default search path (``/opt/homebrew/opt/libomp/lib/``).
On a non-admin Homebrew install that lives under ``~/homebrew/``, the
dynamic linker can't find libomp at import time and the whole ML
adapter registers as ``tool_unavailable``.

Setting ``DYLD_LIBRARY_PATH`` via ``os.environ`` inside conftest is
too late — dyld reads that variable only at process start. So we
symlink ``libomp.dylib`` next to ``libxgboost.dylib`` inside the
venv, which xgboost *does* search via ``@loader_path/``.

On Linux CI (``libomp-dev`` handled by apt) and on Intel-Mac admin
Homebrew (default path resolves), this is a no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_libomp_near_xgboost() -> None:
    if sys.platform != "darwin":
        return
    try:
        import xgboost  # noqa: F401  — if import already works we're done
        return
    except Exception:
        pass

    candidates = [
        Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
        Path("/usr/local/opt/libomp/lib/libomp.dylib"),
        Path.home() / "homebrew" / "opt" / "libomp" / "lib" / "libomp.dylib",
    ]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        return

    # Find xgboost's own lib directory and link libomp.dylib inside it.
    site_packages = next(
        (Path(p) for p in sys.path if p.endswith("site-packages")), None
    )
    if site_packages is None:
        return
    xgboost_lib = site_packages / "xgboost" / "lib"
    if not xgboost_lib.exists():
        return

    target = xgboost_lib / "libomp.dylib"
    if not target.exists():
        try:
            target.symlink_to(source)
        except OSError:
            # e.g. read-only filesystem or permission error — non-fatal
            pass


_ensure_libomp_near_xgboost()
