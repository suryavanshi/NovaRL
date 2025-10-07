"""Utility helpers for NovaRL."""

from .advantages import (
    AdvantageEstimate,
    AdvantageEstimator,
    GAEAdvantageEstimator,
    normalize_advantages,
)
from .stats import compute_discounted_returns, compute_gae, explained_variance
from .timing import RateTracker

__all__ = [
    "AdvantageEstimate",
    "AdvantageEstimator",
    "GAEAdvantageEstimator",
    "RateTracker",
    "compute_discounted_returns",
    "compute_gae",
    "explained_variance",
    "normalize_advantages",
]
