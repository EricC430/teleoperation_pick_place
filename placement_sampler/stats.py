"""One-sample Kolmogorov-Smirnov test against a uniform distribution.

Spec: docs/specs/S2_placement_sampler.md 6 -- print a p-value for "is r^2
uniform over U(r_in^2, r_out^2)?". stdlib only (no scipy in this repo).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def ks_uniform_pvalue(values: Sequence[float], lo: float, hi: float) -> float:
    """p-value of the KS test that `values` are drawn from U(lo, hi).

    Small p (say < 0.05) => reject uniformity. Uses the asymptotic Kolmogorov
    distribution with Stephens' small-sample correction; fine for n >= ~20.
    """
    n = len(values)
    if n == 0:
        raise ValueError("need at least one value")
    if hi <= lo:
        raise ValueError(f"need lo < hi, got {lo}, {hi}")

    u = sorted((v - lo) / (hi - lo) for v in values)
    d = 0.0
    for i, x in enumerate(u):
        xc = min(1.0, max(0.0, x))
        d = max(d, (i + 1) / n - xc, xc - i / n)

    sqrt_n = math.sqrt(n)
    lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d
    p = 2.0 * sum(
        (-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101)
    )
    return max(0.0, min(1.0, p))
