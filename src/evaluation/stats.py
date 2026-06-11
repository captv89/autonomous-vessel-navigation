"""
Statistical utilities for benchmark reporting.

Rates get Wilson score intervals (well-behaved at 0%/100% and small n,
unlike the normal approximation); continuous metrics get percentile
bootstrap intervals with a fixed seed so reports are reproducible.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

Z95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z95
                    ) -> Dict[str, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return {"rate": 0.0, "low": 0.0, "high": 1.0, "n": 0}
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {"rate": round(p, 4), "low": round(max(0.0, center - half), 4),
            "high": round(min(1.0, center + half), 4), "n": n}


def bootstrap_mean(values: Sequence[float], n_resamples: int = 2000,
                   seed: int = 0) -> Optional[Dict[str, float]]:
    """95% percentile bootstrap interval for a mean (seeded)."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        m = float(arr[0])
        return {"mean": round(m, 4), "low": round(m, 4),
                "high": round(m, 4), "n": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    return {"mean": round(float(arr.mean()), 4),
            "low": round(float(np.percentile(means, 2.5)), 4),
            "high": round(float(np.percentile(means, 97.5)), 4),
            "n": len(arr)}


def fmt_rate(ci: Dict[str, float]) -> str:
    return f"{ci['rate']:.1%} [{ci['low']:.1%}, {ci['high']:.1%}]"


def fmt_mean(ci: Optional[Dict[str, float]], digits: int = 2) -> str:
    if ci is None:
        return "-"
    return (f"{ci['mean']:.{digits}f} "
            f"[{ci['low']:.{digits}f}, {ci['high']:.{digits}f}]")
