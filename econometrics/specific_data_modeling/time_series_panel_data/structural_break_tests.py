"""
结构突变检验实现（Chow检验、Quandt-Andrews检验、Bai-Perron检验）
"""

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import f as _f_dist


class StructuralBreakResult(BaseModel):
    """结构突变检验结果"""

    test_type: str = Field(..., description="检验类型")
    test_statistic: float = Field(..., description="检验统计量")
    p_value: float | None = Field(None, description="p值")
    break_points: list[int] | None = Field(None, description="断点位置")
    critical_value: float | None = Field(None, description="临界值")
    n_breaks: int | None = Field(None, description="断点数量")
    n_obs: int = Field(..., description="观测数量")


def _mean_only_ssr(segment: np.ndarray) -> float:
    """SSR for a constant-only (``y = μ + ε``) model on ``segment``."""
    if segment.size == 0:
        return 0.0
    return float(np.sum((segment - segment.mean()) ** 2))


def chow_test(data: list[float], break_point: int) -> StructuralBreakResult:
    """Chow F-test for a single structural break under a constant-only model.

    F = ((SSR_r − SSR_u) / k) / (SSR_u / (n − 2k))   with k = 1.
    p = 1 − F_CDF(F, k, n − 2k).
    """
    arr = np.asarray(data, dtype=np.float64)
    n = arr.size
    k = 1
    if n <= 2 * k or not (k <= break_point < n - k):
        raise ValueError(
            f"Chow test requires break_point in [{k}, {n - k}) and n > 2; "
            f"got break_point={break_point}, n={n}"
        )
    seg1, seg2 = arr[:break_point], arr[break_point:]
    ssr_unrestricted = _mean_only_ssr(seg1) + _mean_only_ssr(seg2)
    ssr_restricted = _mean_only_ssr(arr)
    denominator_df = n - 2 * k
    if ssr_unrestricted <= 1e-12 or denominator_df <= 0:
        # Degenerate residuals → no signal to distinguish from perfect split.
        return StructuralBreakResult(
            test_type="Chow Test",
            test_statistic=float("inf"),
            p_value=0.0,
            break_points=[break_point],
            n_breaks=1,
            n_obs=n,
        )
    f_stat = ((ssr_restricted - ssr_unrestricted) / k) / (
        ssr_unrestricted / denominator_df
    )
    p_value = float(1 - _f_dist.cdf(max(f_stat, 0.0), k, denominator_df))
    return StructuralBreakResult(
        test_type="Chow Test",
        test_statistic=float(f_stat),
        p_value=p_value,
        break_points=[break_point],
        n_breaks=1,
        n_obs=n,
    )


def quandt_andrews_test(data: list[float]) -> StructuralBreakResult:
    """Quandt-Andrews (sup-F) test: sweep interior break-points, report the
    largest Chow F statistic observed and its location.

    Uses the Andrews (1993) 15%/85% trimming so near-boundary instability
    doesn't dominate.
    """
    arr = np.asarray(data, dtype=np.float64)
    n = arr.size
    if n < 10:
        raise ValueError("Quandt-Andrews requires at least 10 observations")
    trim = max(1, int(0.15 * n))
    best_stat = -np.inf
    best_bp = trim
    best_p = 1.0
    for bp in range(trim, n - trim):
        chow = chow_test(list(arr), bp)
        if chow.test_statistic > best_stat:
            best_stat = chow.test_statistic
            best_bp = bp
            best_p = chow.p_value
    return StructuralBreakResult(
        test_type="Quandt-Andrews Test",
        test_statistic=float(best_stat),
        p_value=float(best_p),
        break_points=[best_bp],
        n_breaks=1,
        n_obs=n,
    )


def bai_perron_test(data: list[float], max_breaks: int = 5) -> StructuralBreakResult:
    """Bai-Perron multiple-break detection via repeated sup-F scan.

    Each iteration locates the next strongest break in the largest remaining
    segment; stops when no further break reaches p < 0.05 or ``max_breaks``
    is reached.
    """
    arr = np.asarray(data, dtype=np.float64)
    n = arr.size
    breaks: list[int] = []
    last_p = 1.0
    last_stat = 0.0

    def _segments(sorted_breaks: list[int]) -> list[tuple[int, int]]:
        cuts = [0, *sorted_breaks, n]
        return list(zip(cuts[:-1], cuts[1:], strict=True))

    for _ in range(max_breaks):
        candidates = []
        for start, stop in _segments(sorted(breaks)):
            length = stop - start
            if length < 10:
                continue
            sub = arr[start:stop].tolist()
            qa = quandt_andrews_test(sub)
            # qa.break_points are local to [start, stop); convert to global index
            global_bp = start + qa.break_points[0]
            candidates.append((qa.test_statistic, qa.p_value, global_bp))
        if not candidates:
            break
        candidates.sort(reverse=True)
        best_stat, best_p, best_bp = candidates[0]
        if best_p > 0.05:
            break
        breaks.append(best_bp)
        breaks.sort()
        last_p, last_stat = best_p, best_stat

    return StructuralBreakResult(
        test_type="Bai-Perron Test",
        test_statistic=float(last_stat),
        p_value=float(last_p),
        break_points=breaks,
        n_breaks=len(breaks),
        n_obs=n,
    )
