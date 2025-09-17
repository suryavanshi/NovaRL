"""Reward modules."""

from .fake.basic import IdentityRewardManager, NoisyRewardManager

__all__ = ["IdentityRewardManager", "NoisyRewardManager"]
