"""Shared statistics for the recorded-run analyses (packet 13).

Pure functions, numpy only. Everything is deterministic given the seed.

paired_bootstrap_mean_diff : percentile bootstrap of a paired mean difference over whole cases.
clopper_pearson_upper      : exact binomial (Clopper-Pearson) upper confidence bound.
bootstrap_ratio            : percentile bootstrap of sum(num)/sum(den) over whole cases
                             (whole-task resampling of an output-level pass rate).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    ci_low: float
    ci_high: float
    n_cases: int
    n_resamples: int
    seed: int
    n_nonzero_cases: int
    top_case_share: float  # |d| of the largest-magnitude case / sum |d| (0 if all zero)

    def as_dict(self) -> dict:
        return {
            "point_estimate": self.point,
            "ci95_low": self.ci_low,
            "ci95_high": self.ci_high,
            "n_cases": self.n_cases,
            "n_resamples": self.n_resamples,
            "seed": self.seed,
            "n_cases_with_nonzero_diff": self.n_nonzero_cases,
            "top_case_share_of_abs_diff": self.top_case_share,
        }


def paired_bootstrap_mean_diff(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, seed: int = 0
) -> BootstrapResult:
    """Percentile bootstrap of mean(a - b) resampling whole paired cases with replacement.

    a, b: per-case values of the two conditions, aligned (same case order). The point estimate is
    the plain mean difference; the CI is the 2.5/97.5 percentile of the resampled mean difference.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"paired inputs must be 1-D and aligned, got {a.shape} vs {b.shape}")
    n = a.shape[0]
    if n == 0:
        raise ValueError("no cases")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    abs_d = np.abs(d)
    total = float(abs_d.sum())
    top_share = float(abs_d.max() / total) if total > 0 else 0.0
    return BootstrapResult(
        point=float(d.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        n_cases=n,
        n_resamples=n_resamples,
        seed=seed,
        n_nonzero_cases=int((d != 0).sum()),
        top_case_share=top_share,
    )


def bootstrap_ratio(
    num: np.ndarray, den: np.ndarray, n_resamples: int = 10_000, seed: int = 0
) -> dict:
    """Percentile bootstrap of sum(num)/sum(den) resampling whole cases (tasks) with replacement.

    Used for the output-level pass rate (num = correct outputs per task, den = outputs per task).
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    if num.shape != den.shape or num.ndim != 1 or num.shape[0] == 0:
        raise ValueError("num/den must be aligned non-empty 1-D arrays")
    n = num.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    ratios = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return {
        "point_estimate": float(num.sum() / den.sum()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n_cases": n,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P[X <= k] for X ~ Binomial(n, p), computed in log space (exact up to float rounding)."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    return min(1.0, total)


def clopper_pearson_upper(x: int, n: int, confidence: float = 0.95, two_sided: bool = True) -> float:
    """Exact (Clopper-Pearson) upper confidence bound for a binomial proportion.

    two_sided=True gives the upper end of the central (1-confidence)/2 interval, the usual
    "95% CI" convention; two_sided=False gives the one-sided bound at tail = 1-confidence.
    For x == 0 the bound has the closed form 1 - tail**(1/n); for x > 0 it is the root of
    P[X <= x | p] = tail, found by bisection on the exact binomial CDF.
    """
    if n <= 0 or x < 0 or x > n:
        raise ValueError(f"invalid x={x}, n={n}")
    tail = (1.0 - confidence) / 2.0 if two_sided else (1.0 - confidence)
    if x == n:
        return 1.0
    if x == 0:
        return 1.0 - tail ** (1.0 / n)
    lo, hi = x / n, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _binom_cdf(x, n, mid) > tail:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
